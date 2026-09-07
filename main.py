# main.py

import argparse
import os

from ecg_bsa.config import get_config
from ecg_bsa.training import run_experiment
from utils import create_folders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn1d", "resnet", "cnn_vit"],
        help="Model to train. If not set, config.py is used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs. If not set, config.py is used.",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class-weighted BCE loss.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["full", "sqrt"],
        help="How strong the positive class weights should be.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay used by Adam.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate used by Adam.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        help="Optimizer used for training.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "warmup_cosine"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        help="Number of warmup epochs.",
    )
    parser.add_argument(
        "--threshold-tuning",
        action="store_true",
        help="Tune one decision threshold per class on the held-out fold.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use simple training data augmentation.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop training when validation loss stops improving.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=["val_loss", "recording_val_loss"],
        help="Metric used for early stopping.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Patience used for early stopping.",
    )
    parser.add_argument(
        "--recording-eval",
        action="store_true",
        help="Average segment predictions per recording during evaluation.",
    )
    parser.add_argument(
        "--protocol",
        choices=["kfold", "group_kfold", "loso", "lmso"],
        help="Evaluation protocol.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Extra name used for saved results and checkpoints.",
    )
    return parser.parse_args()


def processed_data_exists(config):
    data_path = config["paths"]["processed_data"]

    if not os.path.isdir(data_path):
        return False

    return any(file.endswith(".npy") for file in os.listdir(data_path))


def main():
    args = parse_args()
    config = get_config()

    if args.model is not None:
        config["model"]["name"] = args.model

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.class_weights:
        config["training"]["class_weights"] = True

    if args.class_weight_mode is not None:
        config["training"]["class_weight_mode"] = args.class_weight_mode

    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    if args.optimizer is not None:
        config["training"]["optimizer"] = args.optimizer

    if args.scheduler is not None:
        config["training"]["scheduler"] = args.scheduler

    if args.warmup_epochs is not None:
        config["training"]["warmup_epochs"] = args.warmup_epochs

    if args.threshold_tuning:
        config["training"]["threshold_tuning"] = True

    if args.augment:
        config["training"]["use_augmentation"] = True

    if args.early_stopping:
        config["training"]["early_stopping"] = True

    if args.early_stopping_metric is not None:
        config["training"]["early_stopping_metric"] = args.early_stopping_metric

    if args.patience is not None:
        config["training"]["patience"] = args.patience

    if args.recording_eval:
        config["training"]["recording_eval"] = True

    if args.protocol is not None:
        config["evaluation"]["protocol"] = args.protocol

    if args.experiment_name is not None:
        config["training"]["experiment_name"] = args.experiment_name

    create_folders(config)

    if not processed_data_exists(config):
        from src.ecg_bsa.preprocessing import preprocess_dataset

        preprocess_dataset(config)

    run_experiment(config)


if __name__ == "__main__":
    main()
