import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from src.ecg_bsa.config import get_config
from src.ecg_bsa.dataset import BiosignalDataset
from src.ecg_bsa.models import get_model

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn1d", "resnet"], default="cnn1d")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--target-class", type=int, default=6)
    parser.add_argument("--classes", default=None)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--leads", default=None)
    parser.add_argument("--max-check", type=int, default=4000)
    parser.add_argument("--qrs", action="store_true")
    parser.add_argument("--gradcam-only", action="store_true")
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--attention-values", action="store_true")
    parser.add_argument("--attention-summary", action="store_true")
    parser.add_argument("--notes-name", default="xai_clinical_notes.csv")
    return parser.parse_args()


def parse_number_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_model(config, checkpoint, device):
    model = get_model(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_target_layer(model, model_name):
    if model_name == "cnn1d":
        return model.conv5
    return model.block4.conv2


class GradCam1D:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.activations = None
        self.gradients = None

        self.forward_hook = layer.register_forward_hook(self.save_activation)
        self.backward_hook = layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __call__(self, x, class_idx):
        self.model.zero_grad()

        with torch.backends.cudnn.flags(enabled=False):
            output = self.model(x)
            score = output[0, class_idx]
            score.backward()

        # average gradients over time
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def attention_map(model, x):
    with torch.no_grad():
        _ = model(x)

    weights = model.attention.last_weights

    weights = weights.transpose(1, 2)
    weights = F.interpolate(weights, size=x.shape[-1], mode="linear", align_corners=False)
    weights = weights.squeeze().detach().cpu().numpy()

    if weights.max() > weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights = np.zeros_like(weights)

    return weights


def attention_values(model, x):
    with torch.no_grad():
        _ = model(x)

    weights = model.attention.last_weights
    return weights.squeeze().detach().cpu().numpy()


def find_examples(model, dataset, device, class_idx, max_check):
    correct = None
    wrong = None

    for i in range(min(len(dataset), max_check)):
        item = dataset[i]
        x = item["signal"].unsqueeze(0).to(device)
        y = item["label"].numpy()

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0]

        pred = (prob >= 0.5).astype(np.float32)
        true_value = y[class_idx]
        pred_value = pred[class_idx]

        if correct is None and true_value == 1 and pred_value == 1:
            correct = (i, item, prob)

        if wrong is None and true_value != pred_value:
            wrong = (i, item, prob)

        if correct is not None and wrong is not None:
            break

    return correct, wrong


def qrs_peaks(lead_signal, fs):
    signal = np.abs(lead_signal - np.median(lead_signal))
    distance = int(0.25 * fs)
    height = np.percentile(signal, 92)
    prominence = max(np.std(signal), 0.1)
    peaks, _ = find_peaks(signal, distance=distance, height=height, prominence=prominence)

    if len(peaks) < 2:
        height = np.percentile(signal, 88)
        peaks, _ = find_peaks(signal, distance=distance, height=height)

    return peaks


def plot_xai(signal, heatmap, title, out_path, lead_idx, fs, show_qrs):
    lead_signal = signal[lead_idx]
    t = np.arange(len(lead_signal))
    lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"Lead {lead_idx + 1}"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, lead_signal, color="black", linewidth=1)
    ax.imshow(
        heatmap.reshape(1, -1),
        aspect="auto",
        cmap="jet",
        alpha=0.35,
        extent=[0, len(lead_signal), lead_signal.min(), lead_signal.max()],
    )

    if show_qrs:
        for peak in qrs_peaks(lead_signal, fs):
            ax.axvline(peak, color="red", linestyle="--", linewidth=1.3, alpha=0.9)
        peaks = qrs_peaks(lead_signal, fs)
        ax.scatter(
            peaks,
            lead_signal[peaks],
            color="red",
            marker="v",
            s=35,
            zorder=5,
            label="QRS marker",
        )

    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel(lead_name)
    if show_qrs:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_attention_values(weights, title, out_path):
    steps = np.arange(1, len(weights) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, weights, color="#1f77b4", linewidth=1.5)
    ax.fill_between(steps, weights, color="#1f77b4", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("BiGRU time step")
    ax.set_ylabel("Attention weight")
    ax.grid(alpha=0.3)

    top_indices = np.argsort(weights)[-5:]
    ax.scatter(steps[top_indices], weights[top_indices], color="#d62728", s=35, zorder=5, label="Top weights")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_attention_values(weights, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "attention_weight"])

        for idx, value in enumerate(weights, start=1):
            writer.writerow([idx, float(value)])


def plot_attention_summary(curves, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))

    for label, weights in curves:
        steps = np.arange(1, len(weights) + 1)
        ax.plot(steps, weights, linewidth=1.4, label=label)

    ax.set_title(title)
    ax.set_xlabel("BiGRU time step")
    ax.set_ylabel("Attention weight")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_attention_summary_csv(curves, out_path):
    max_len = max(len(weights) for _, weights in curves)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step"] + [label for label, _ in curves])

        for idx in range(max_len):
            row = [idx + 1]

            for _, weights in curves:
                if idx < len(weights):
                    row.append(float(weights[idx]))
                else:
                    row.append("")

            writer.writerow(row)


def run_attention_summary(model, save_name, dataset, device, class_list, out_dir, max_check):
    curves = []

    for class_idx in class_list:
        correct, _ = find_examples(model, dataset, device, class_idx, max_check)

        if correct is None:
            print(f"class{class_idx + 1}: no correct positive example found for attention summary.")
            continue

        idx, item, prob = correct
        x = item["signal"].unsqueeze(0).to(device)
        weights = attention_values(model, x)
        label = f"Class {class_idx + 1} (p={prob[class_idx]:.2f})"
        curves.append((label, weights))
        print(f"class{class_idx + 1} attention summary example: sample {idx}")

    if len(curves) == 0:
        return []

    plot_attention_summary(
        curves,
        "Attention Weights Across Classes",
        os.path.join(out_dir, f"{save_name}_all_classes_attention_values.png"),
    )
    save_attention_summary_csv(
        curves,
        os.path.join(out_dir, f"{save_name}_all_classes_attention_values.csv"),
    )

    return [[save_name, "all", "correct", "-", "-", "-", f"{save_name}_all_classes_attention_values.png; {save_name}_all_classes_attention_values.csv", ""]]


def explain_sample(model, model_name, save_name, item, prob, class_idx, lead_idx, out_dir, tag, fs, show_qrs, gradcam_only=False, attention_only=False, attention_values_only=False):
    signal = item["signal"].numpy()
    x = item["signal"].unsqueeze(0).to(next(model.parameters()).device)
    lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"lead{lead_idx + 1}"
    lead_tag = lead_name.replace(" ", "").replace("/", "")
    class_name = f"class{class_idx + 1}"

    if model_name == "cnn1d" and attention_values_only:
        weights = attention_values(model, x)
        base_name = f"{save_name}_{class_name}_{tag}_attention_values"
        plot_attention_values(
            weights,
            f"{save_name} Attention Weights ({tag}, {class_name}, prob={prob[class_idx]:.3f})",
            os.path.join(out_dir, f"{base_name}.png"),
        )
        save_attention_values(weights, os.path.join(out_dir, f"{base_name}.csv"))
        return

    if not attention_only:
        gradcam = GradCam1D(model, get_target_layer(model, model_name))
        cam = gradcam(x, class_idx)
        gradcam.remove()

        plot_xai(
            signal,
            cam,
            f"{save_name} Grad-CAM ({tag}, {class_name}, {lead_name}, prob={prob[class_idx]:.3f})",
            os.path.join(out_dir, f"{save_name}_{class_name}_{tag}_{lead_tag}_gradcam.png"),
            lead_idx,
            fs,
            show_qrs,
        )

    if model_name == "cnn1d" and not gradcam_only:
        attn = attention_map(model, x)
        plot_xai(
            signal,
            attn,
            f"{save_name} Attention ({tag}, {class_name}, {lead_name}, prob={prob[class_idx]:.3f})",
            os.path.join(out_dir, f"{save_name}_{class_name}_{tag}_{lead_tag}_attention.png"),
            lead_idx,
            fs,
            show_qrs,
        )


def write_xai_notes(out_dir, rows, notes_name):
    path = os.path.join(out_dir, notes_name)
    exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(
                [
                    "model",
                    "class",
                    "example_type",
                    "lead",
                    "sample_idx",
                    "probability",
                    "plot_files",
                    "clinical_alignment_notes",
                ]
            )

        for row in rows:
            writer.writerow(row)


def run_for_class(model, model_name, save_name, dataset, device, class_idx, lead_idx, out_dir, max_check, fs, show_qrs, gradcam_only=False, attention_only=False, attention_values_only=False):
    rows = []
    correct, wrong = find_examples(model, dataset, device, class_idx, max_check)
    class_name = f"class{class_idx + 1}"
    lead_name = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f"Lead {lead_idx + 1}"
    lead_tag = lead_name.replace(" ", "").replace("/", "")

    if correct is not None:
        idx, item, prob = correct
        print(f"{class_name} correct example: sample {idx}, {lead_name}")
        explain_sample(model, model_name, save_name, item, prob, class_idx, lead_idx, out_dir, "correct", fs, show_qrs, gradcam_only, attention_only, attention_values_only)

        files = []
        if attention_values_only:
            files.append(f"{save_name}_{class_name}_correct_attention_values.png")
            files.append(f"{save_name}_{class_name}_correct_attention_values.csv")
        elif not attention_only:
            files.append(f"{save_name}_{class_name}_correct_{lead_tag}_gradcam.png")
        if model_name == "cnn1d" and not gradcam_only and not attention_values_only:
            files.append(f"{save_name}_{class_name}_correct_{lead_tag}_attention.png")

        rows.append([save_name, class_idx + 1, "correct", lead_name, idx, prob[class_idx], "; ".join(files), ""])
    else:
        print(f"{class_name}: no correct positive example found.")

    if wrong is not None:
        idx, item, prob = wrong
        print(f"{class_name} wrong example: sample {idx}, {lead_name}")
        explain_sample(model, model_name, save_name, item, prob, class_idx, lead_idx, out_dir, "wrong", fs, show_qrs, gradcam_only, attention_only, attention_values_only)

        files = []
        if attention_values_only:
            files.append(f"{save_name}_{class_name}_wrong_attention_values.png")
            files.append(f"{save_name}_{class_name}_wrong_attention_values.csv")
        elif not attention_only:
            files.append(f"{save_name}_{class_name}_wrong_{lead_tag}_gradcam.png")
        if model_name == "cnn1d" and not gradcam_only and not attention_values_only:
            files.append(f"{save_name}_{class_name}_wrong_{lead_tag}_attention.png")

        rows.append([save_name, class_idx + 1, "wrong", lead_name, idx, prob[class_idx], "; ".join(files), ""])
    else:
        print(f"{class_name}: no wrong example found.")

    return rows


def main():
    args = parse_args()
    config = get_config()
    config["model"]["name"] = args.model

    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(config["paths"]["checkpoints"], f"{args.model}_fold0.pt")

    save_name = args.output_name
    if save_name is None:
        save_name = args.model

    out_dir = os.path.join(config["paths"]["outputs"], "xai")
    os.makedirs(out_dir, exist_ok=True)

    if args.leads is not None:
        lead_list = [lead - 1 for lead in parse_number_list(args.leads)]
    else:
        lead_list = [args.lead - 1]

    dataset = BiosignalDataset(config)
    model = load_model(config, checkpoint, device)

    if args.all_classes:
        class_list = list(range(config["dataset"]["num_classes"]))
    elif args.classes is not None:
        class_list = [class_num - 1 for class_num in parse_number_list(args.classes)]
    else:
        class_list = [args.target_class - 1]

    fs = config["preprocess"]["downsampled_rate"]
    note_rows = []

    if args.attention_summary:
        note_rows.extend(run_attention_summary(model, save_name, dataset, device, class_list, out_dir, args.max_check))
    else:
        for class_idx in class_list:
            for lead_idx in lead_list:
                note_rows.extend(
                    run_for_class(
                        model,
                        args.model,
                        save_name,
                        dataset,
                        device,
                        class_idx,
                        lead_idx,
                        out_dir,
                        args.max_check,
                        fs,
                        args.qrs,
                        args.gradcam_only,
                        args.attention_only,
                        args.attention_values,
                    )
                )

    write_xai_notes(out_dir, note_rows, args.notes_name)

    print(f"Saved XAI plots to {out_dir}")


if __name__ == "__main__":
    main()
