from pathlib import Path
import pandas as pd
from ecg_bsa.preprocessing import form_subject_dict, preprocessing_pipeline, segmentation, save_subject_file, validate_processed_files

TEST_DIR = Path(__file__).parent          # tests/integration/
DATA_DIR = TEST_DIR.parent / "data"       # tests/data/


def test_form_subject_dict_correct_labels_and_signal():
    signals_path = DATA_DIR / "Training_WFDB"
    ref_path = DATA_DIR / "REFERENCE.csv"
    subject_number = 1  # adjust to match a real subject_id in your fixture set

    fake_config = {"dataset": {"num_classes": 9}}

    result = form_subject_dict(
        signals_path=str(signals_path),
        ref_path=str(ref_path),
        subject_number=subject_number,
        config=fake_config,
    )

    reference_df = pd.read_csv(ref_path)
    expected_row = reference_df.iloc[subject_number - 1]
    expected_labels = expected_row[["First_label", "Second_label", "Third_label"]].dropna().astype(int).tolist()

    assert result["subject_id"] == f"A{subject_number:04d}"
    assert result["signals"].ndim == 2
    assert result["signals"].shape[1] == 12  # 12-lead ECG

    for label in expected_labels:
        assert result["labels"][label - 1] == 1
    assert result["labels"].sum() == len(expected_labels)

def test_full_pipeline_round_trip(tmp_path):
    signals_path = DATA_DIR / "Training_WFDB"
    ref_path = DATA_DIR / "REFERENCE.csv"
    subject_number = 1  # reuse whatever real subject you confirmed above

    config = {
        "dataset": {
            "sampling_rate": 500,
            "input_channels": 12,
            "segment_length": 1500,
            "num_classes": 9,
        },
        "preprocess": {
            "powerline": 50,
            "lowcut": 0.5,
            "highcut": 45,
            "downsampled_rate": 250,
            "overlap_ratio": 0.5,
        },
        "paths": {
            "processed_data": str(tmp_path),
        },
    }

    subj = form_subject_dict(
        signals_path=str(signals_path),
        ref_path=str(ref_path),
        subject_number=subject_number,
        config=config,
    )
    subj["signals"] = preprocessing_pipeline(subj["signals"], config=config)
    segments = segmentation(subj, config["dataset"]["segment_length"], config["preprocess"]["overlap_ratio"])

    segments_saved = save_subject_file(segments, config=config)
    assert segments_saved == len(segments)
    assert segments_saved > 0

    saved_files = list(tmp_path.glob("*.npy"))
    assert len(saved_files) == 1

    valid_files, invalid_files = validate_processed_files(str(tmp_path), config)
    assert valid_files == 1
    assert invalid_files == 0