# training.py

import csv
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
import wandb


from src.ecg_bsa.config import STFT_PARAMS
from src.ecg_bsa.dataset import BiosignalDataset
from src.ecg_bsa.models import get_model
from src.ecg_bsa.transforms import stft_transform
from utils import (
    check_no_leakage,
    get_subject_ids,
    group_kfold_split,
    kfold_split_indices,
    lmso_split,
    loso_split,
)

SPECTRAL_MODELS = {"cnn_vit"}


def make_transform(config):
    if config["model"]["name"] in SPECTRAL_MODELS:
        return stft_transform(**STFT_PARAMS)

    return None


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def macro_auc(y_true, y_prob):
    aucs = []

    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]

        if len(np.unique(class_true)) < 2:
            continue

        aucs.append(roc_auc_score(class_true, y_prob[:, class_idx]))

    if len(aucs) == 0:
        return None

    return float(np.mean(aucs))


def tune_thresholds(y_true, y_prob):
    thresholds = []
    tuned_pred = np.zeros_like(y_prob, dtype=np.float32)

    for class_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_score = -1

        for threshold in np.arange(0.1, 0.91, 0.05):
            pred = (y_prob[:, class_idx] >= threshold).astype(np.float32)
            score = f1_score(y_true[:, class_idx], pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds.append(best_threshold)
        tuned_pred[:, class_idx] = (y_prob[:, class_idx] >= best_threshold).astype(np.float32)

    return thresholds, tuned_pred


def multilabel_confusion_counts(y_true, y_pred):
    counts = []

    for class_idx in range(y_true.shape[1]):
        true_class = y_true[:, class_idx]
        pred_class = y_pred[:, class_idx]

        tp = int(np.sum((true_class == 1) & (pred_class == 1)))
        fp = int(np.sum((true_class == 0) & (pred_class == 1)))
        fn = int(np.sum((true_class == 1) & (pred_class == 0)))
        tn = int(np.sum((true_class == 0) & (pred_class == 0)))

        counts.append(
            {
                "class": class_idx + 1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    return counts


def class_pair_matrix(y_true, y_pred):
    num_classes = y_true.shape[1]
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for idx in range(y_true.shape[0]):
        true_classes = np.where(y_true[idx] == 1)[0]
        pred_classes = np.where(y_pred[idx] == 1)[0]

        for true_idx in true_classes:
            for pred_idx in pred_classes:
                matrix[true_idx, pred_idx] += 1

    return matrix.tolist()


def recording_metrics(subject_ids, y_true, y_prob):
    grouped = {}

    for idx, subject_id in enumerate(subject_ids):
        if subject_id not in grouped:
            grouped[subject_id] = {"labels": y_true[idx], "probs": []}

        grouped[subject_id]["probs"].append(y_prob[idx])

    rec_true = []
    rec_prob = []

    for item in grouped.values():
        rec_true.append(item["labels"])
        rec_prob.append(np.mean(item["probs"], axis=0))

    rec_true = np.asarray(rec_true, dtype=np.float32)
    rec_prob = np.asarray(rec_prob, dtype=np.float32)
    rec_pred = (rec_prob >= 0.5).astype(np.float32)

    rec_prob_for_loss = np.clip(rec_prob, 1e-7, 1 - 1e-7)
    loss = F.binary_cross_entropy(
        torch.tensor(rec_prob_for_loss, dtype=torch.float32),
        torch.tensor(rec_true, dtype=torch.float32),
    )

    return {
        "recording_val_loss": float(loss.item()),
        "recording_macro_f1": float(f1_score(rec_true, rec_pred, average="macro", zero_division=0)),
        "recording_micro_f1": float(f1_score(rec_true, rec_pred, average="micro", zero_division=0)),
        "recording_macro_auc": macro_auc(rec_true, rec_prob),
        "recording_confusion": multilabel_confusion_counts(rec_true, rec_pred),
        "recording_pair_matrix": class_pair_matrix(rec_true, rec_pred),
    }


def evaluate(model, loader, device, criterion=None, threshold_tuning=False, recording_eval=False):
    model.eval()
    true_labels = []
    probabilities = []
    subject_ids = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)

            outputs = model(x)

            if criterion is not None:
                loss = criterion(outputs, y)
                total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu()

            true_labels.append(y.cpu().numpy())
            probabilities.append(probs.numpy())
            subject_ids.extend(batch["subject_id"])

    y_true = np.concatenate(true_labels, axis=0)
    y_prob = np.concatenate(probabilities, axis=0)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    thresholds = [0.5] * y_true.shape[1]

    if threshold_tuning:
        thresholds, y_pred = tune_thresholds(y_true, y_prob)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_auc": macro_auc(y_true, y_prob),
        "per_class_f1": per_class_f1.tolist(),
        "confusion": multilabel_confusion_counts(y_true, y_pred),
        "pair_matrix": class_pair_matrix(y_true, y_pred),
        "thresholds": thresholds,
    }

    if criterion is not None:
        metrics["val_loss"] = float(total_loss / len(loader))

    if recording_eval:
        metrics.update(recording_metrics(subject_ids, y_true, y_prob))

    return metrics


def get_class_weights(dataset, device, num_classes, mode="full"):
    label_counts = np.zeros(num_classes, dtype=np.float32)

    for item in dataset:
        label_counts += item["label"].numpy()

    total = len(dataset)
    neg_counts = total - label_counts
    weights = neg_counts / np.maximum(label_counts, 1)

    if mode == "sqrt":
        weights = np.sqrt(weights)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_optimizer(model, config):
    name = config["training"].get("optimizer", "adam")
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def set_learning_rate(optimizer, config, epoch):
    scheduler = config["training"].get("scheduler", "none")

    if scheduler != "warmup_cosine":
        return optimizer.param_groups[0]["lr"]

    base_lr = config["training"]["learning_rate"]
    total_epochs = config["training"]["epochs"]
    warmup_epochs = config["training"].get("warmup_epochs", 0)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        if total_epochs == warmup_epochs:
            progress = 1
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)

        progress = min(max(progress, 0), 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def result_name(config):
    name = config["model"]["name"]
    experiment_name = config["training"].get("experiment_name")

    if experiment_name:
        return f"{name}_{experiment_name}"

    if config["training"].get("class_weights", False):
        return f"{name}_weighted"

    return name


def save_results(results, config):
    results_path = config["paths"]["results"]
    model_name = result_name(config)
    json_path = os.path.join(results_path, f"{model_name}_results.json")
    csv_path = os.path.join(results_path, f"{model_name}_summary.csv")
    metrics = [
        "loss",
        "val_loss",
        "recording_val_loss",
        "macro_f1",
        "micro_f1",
        "macro_auc",
        "recording_macro_f1",
        "recording_micro_f1",
        "recording_macro_auc",
    ]

    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=4)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold",
                "loss",
                "val_loss",
                "recording_val_loss",
                "macro_f1",
                "micro_f1",
                "macro_auc",
                "recording_macro_f1",
                "recording_micro_f1",
                "recording_macro_auc",
            ]
        )

        for fold_result in results:
            final = fold_result["final"]
            writer.writerow(
                [
                    fold_result["fold"],
                    final["loss"],
                    final.get("val_loss"),
                    final.get("recording_val_loss"),
                    final["macro_f1"],
                    final["micro_f1"],
                    final["macro_auc"],
                    final.get("recording_macro_f1"),
                    final.get("recording_micro_f1"),
                    final.get("recording_macro_auc"),
                ]
            )

        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])

        for metric in metrics:
            values = [
                fold_result["final"][metric]
                for fold_result in results
                if metric in fold_result["final"] and fold_result["final"][metric] is not None
            ]

            if len(values) == 0:
                continue

            writer.writerow(
                [
                    metric,
                    float(np.mean(values)),
                    float(np.std(values)),
                ]
            )

        writer.writerow([])
        writer.writerow(["class", "tn", "fp", "fn", "tp"])

        confusion_totals = {}

        for fold_result in results:
            confusion_key = "recording_confusion" if config["training"].get("recording_eval", False) else "confusion"

            for item in fold_result["final"].get(confusion_key, []):
                class_idx = item["class"]

                if class_idx not in confusion_totals:
                    confusion_totals[class_idx] = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

                for key in ["tn", "fp", "fn", "tp"]:
                    confusion_totals[class_idx][key] += item[key]

        for class_idx in sorted(confusion_totals):
            item = confusion_totals[class_idx]
            writer.writerow([class_idx, item["tn"], item["fp"], item["fn"], item["tp"]])


def make_splits(config):
    protocol = config["evaluation"]["protocol"]
    num_folds = config["evaluation"]["num_folds"]

    if protocol in ["loso", "lmso", "group_kfold"]:
        subject_ids = get_subject_ids(config["paths"]["processed_data"])

        if protocol == "loso":
            return loso_split(subject_ids)

        if protocol == "lmso":
            return lmso_split(subject_ids, k=num_folds)

        return group_kfold_split(subject_ids, k=num_folds)

    if protocol == "kfold":
        full_dataset = BiosignalDataset(config, subject_ids=None)
        return kfold_split_indices(len(full_dataset), num_folds)

    raise ValueError(f"Unknown evaluation protocol: {protocol}")


def run_experiment(config):
    device = config["training"]["device"]
    protocol = config["evaluation"]["protocol"]
    splits = make_splits(config)
    transform = make_transform(config)
    results = []

    for fold, split in enumerate(splits):
        print(f"\nFold {fold + 1}")

        if protocol in ["loso", "lmso", "group_kfold"]:
            train_sids, test_sids = split
            check_no_leakage(train_sids, test_sids)

            train_dataset = BiosignalDataset(
                config,
                train_sids, transform=transform,
                augment=config["training"].get("use_augmentation", False),
            )

        else:
            train_idx, test_idx = split
            full_dataset = BiosignalDataset(config, subject_ids=None, transform=transform)
            train_dataset = Subset(full_dataset, train_idx)
            test_dataset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        model = get_model(config).to(device)

        dummy_signal = torch.randn(
            config["dataset"]["input_channels"],
            config["dataset"]["segment_length"],
        )

        if transform is not None:
            dummy = torch.stack([transform(dummy_signal) for _ in range(2)]).to(device)
        else:
            dummy = dummy_signal.unsqueeze(0).repeat(2, 1, 1).to(device)

        try:
            _ = model(dummy)
        except Exception as exc:
            raise RuntimeError(f"Model forward failed: {exc}") from exc

        optimizer = make_optimizer(model, config)

        if config["training"].get("class_weights", False):
            class_weights = get_class_weights(
                train_dataset,
                device,
                config["dataset"]["num_classes"],
                mode=config["training"].get("class_weight_mode", "full"),
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()

        val_criterion = nn.BCEWithLogitsLoss()
        fold_history = []
        best_loss = None
        best_state = None
        best_result = None
        no_improve = 0
        use_early_stopping = config["training"].get("early_stopping", False)
        early_stopping_metric = config["training"].get("early_stopping_metric", "val_loss")
        patience = config["training"].get("patience", 15)

        for epoch in range(config["training"]["epochs"]):
            lr = set_learning_rate(optimizer, config, epoch + 1)
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(
                model,
                test_loader,
                device,
                val_criterion,
                threshold_tuning=config["training"].get("threshold_tuning", False),
                recording_eval=config["training"].get("recording_eval", False),
            )

            print(
                f"Epoch {epoch + 1}: "
                f"Loss={loss:.4f}, "
                f"Val-Loss={metrics['val_loss']:.4f}, "
                f"Macro-F1={metrics['macro_f1']:.4f}, "
                f"Macro-AUC={metrics['macro_auc']}"
            )

            if config["training"].get("recording_eval", False):
                print(
                    f"Recording: "
                    f"Val-Loss={metrics['recording_val_loss']:.4f}, "
                    f"Macro-F1={metrics['recording_macro_f1']:.4f}, "
                    f"Macro-AUC={metrics['recording_macro_auc']}"
                )

            fold_history.append(
                {
                    "epoch": epoch + 1,
                    "learning_rate": float(lr),
                    "loss": float(loss),
                    **metrics,
                }
            )

            if use_early_stopping:
                val_loss = metrics[early_stopping_metric]

                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_result = fold_history[-1]
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if use_early_stopping and best_state is not None:
            model.load_state_dict(best_state)
            final_result = best_result
        else:
            final_result = fold_history[-1]

        torch.save(
            model.state_dict(),
            os.path.join(
                config["paths"]["checkpoints"],
                f"{result_name(config)}_fold{fold}.pt",
            ),
        )

        results.append(
            {
                "fold": fold + 1,
                "final": final_result,
                "stopped_epoch": fold_history[-1]["epoch"],
                "history": fold_history,
            }
        )
        save_results(results, config)

    final_f1 = [fold["final"]["macro_f1"] for fold in results]
    final_auc = [
        fold["final"]["macro_auc"]
        for fold in results
        if fold["final"]["macro_auc"] is not None
    ]

    print("\nFinal Results")
    print(f"Mean Macro-F1: {sum(final_f1) / len(final_f1):.4f}")

    if len(final_auc) > 0:
        print(f"Mean Macro-AUC: {sum(final_auc) / len(final_auc):.4f}")

    save_results(results, config)

    return results
