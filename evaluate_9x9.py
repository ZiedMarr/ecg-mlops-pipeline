import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_bsa.config import get_config
from ecg_bsa.dataset import BiosignalDataset
from ecg_bsa.models import get_model
from ecg_bsa.training import evaluate, make_splits


CLASS_NAMES = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
MODEL_TITLES = {
    "cnn1d": "CNN-BiGRU-Attention",
    "resnet": "ResNet",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="recording_eval")
    parser.add_argument("--models", default="cnn1d,resnet")
    return parser.parse_args()


def add_matrix(total, matrix):
    if total is None:
        total = np.zeros_like(np.asarray(matrix, dtype=int))

    return total + np.asarray(matrix, dtype=int)


def save_matrix_csv(matrix, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)

        for idx, row in enumerate(matrix):
            writer.writerow([CLASS_NAMES[idx]] + [int(value) for value in row])


def save_matrix_plot(matrix, title, out_path):
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted positive class")
    ax.set_ylabel("True positive class")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = norm_matrix[row_idx, col_idx]
            color = "white" if value > 0.55 else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def evaluate_model(model_name, experiment_name, config, device):
    config["model"]["name"] = model_name
    config["training"]["recording_eval"] = True
    config["training"]["experiment_name"] = experiment_name

    splits = make_splits(config)
    total_matrix = None

    for fold, split in enumerate(splits):
        print(f"{model_name}: fold {fold + 1}")

        _, test_sids = split
        test_dataset = BiosignalDataset(config, test_sids)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        checkpoint = os.path.join(
            config["paths"]["checkpoints"],
            f"{model_name}_{experiment_name}_fold{fold}.pt",
        )

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        model = get_model(config).to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()

        metrics = evaluate(
            model,
            test_loader,
            device,
            criterion=None,
            threshold_tuning=False,
            recording_eval=True,
        )

        total_matrix = add_matrix(total_matrix, metrics["recording_pair_matrix"])

    return total_matrix


def main():
    args = parse_args()
    config = get_config()
    device = config["training"]["device"]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(config["paths"]["plots"], exist_ok=True)
    os.makedirs(config["paths"]["results"], exist_ok=True)

    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        matrix = evaluate_model(model_name, args.experiment_name, config, device)
        base_name = f"{model_name}_{args.experiment_name}_9x9_class_matrix"

        csv_path = os.path.join(config["paths"]["results"], f"{base_name}.csv")
        png_path = os.path.join(config["paths"]["plots"], f"{base_name}.png")

        save_matrix_csv(matrix, csv_path)
        title = f"{MODEL_TITLES.get(model_name, model_name)} 9x9 Confusion Matrix"
        save_matrix_plot(matrix, title, png_path)

        print(f"Saved {csv_path}")
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
