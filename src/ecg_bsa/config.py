# config.py

STFT_PARAMS = {"n_fft": 64, "hop_length": 16, "win_length": 64}


def get_config():
    config = {
        "wandb": {
            "EXPERIMENT_NAME": "cnn_vit" ,
            "PROJECT_NAME": "ecg-classification"
        },
        "paths": {
            "raw_data": "./data/raw/",
            "processed_data": "./data/processed/",
            "outputs": "./outputs/",
            "checkpoints": "./outputs/models/",
            "results": "./outputs/results/",
            "plots": "./outputs/plots/"
        },
        "raw_dataset": {
            "num_subjects" : 6877,
            "name" : "Physionet Challenge China 2018"

        },

        "dataset": {
            "input_channels": 12,
            "segment_length": 1500,
            "num_classes": 9,
            "sampling_rate" : 500
        },

        "model": {
            "name": "cnn_vit"
        },

        "preprocess": {
            "powerline" : 50,
            "lowcut" : 0.5,
            "highcut" :45 ,
            "downsampled_rate" : 250,
            "overlap_ratio" : 0.5
        },

        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "scheduler": "none",
            "warmup_epochs": 0,
            "device": "cpu",  # change to "cpu" if needed
            "class_weights": False,
            "class_weight_mode": "full",
            "weight_decay": 0,
            "threshold_tuning": False,
            "experiment_name": None,
            "early_stopping": False,
            "early_stopping_metric": "val_loss",
            "patience": 15,
            "recording_eval": False,
            "use_augmentation": False,
            "augmentation": {
                "noise_std": 0.01,
                "scale_range": 0.1,
                "shift_samples": 25
            }
        },

        "evaluation": {
            "protocol": "group_kfold",
            "num_folds": 10
        }
    }

    return config
