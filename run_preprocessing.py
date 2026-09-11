# run_preprocessing.py (separate entry point, not called from main.py)
from ecg_bsa.config import get_config
from ecg_bsa.pipeline_preprocessing import preprocess_dataset
import json, hashlib
import os

def compute_fingerprint(params: dict) -> str:
    """
    Deterministic short hash of the parameters that define a processed
    dataset version. Same params -> same fingerprint, always.
    """
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

def main():
    config = get_config()

    # compute parameters for save
    params = {
    "dataset": config["dataset"],
    "preprocess": config["preprocess"],
    "raw_dataset": config["raw_dataset"],
    }
    fingerprint = compute_fingerprint(params)
    save_path = os.path.join(config["paths"]["processed_data"], fingerprint)
    


    result = preprocess_dataset(
        segment_size=config["dataset"]["segment_length"],
        overlap_ratio=config["preprocess"]["overlap_ratio"],
        num_pat=config["raw_dataset"]["num_subjects"],
        sampling_rate=config["dataset"]["sampling_rate"],
        powerline=config["preprocess"]["powerline"],
        lowcut=config["preprocess"]["lowcut"],
        highcut=config["preprocess"]["highcut"],
        downsampled_rate=config["preprocess"]["downsampled_rate"],
        raw_path=config["paths"]["raw_data"],
        save_path=save_path,
        expected_channels=config["dataset"]["input_channels"],
        expected_segment_length=config["dataset"]["segment_length"],
        expected_num_classes=config["dataset"]["num_classes"],
)
    manifest = {
    "fingerprint": fingerprint,
    "params": params,
    "processed": result["processed"],
    "skipped": result["skipped"],
    }
    with open(os.path.join(save_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()