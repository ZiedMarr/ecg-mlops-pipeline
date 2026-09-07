from pathlib import Path
import pandas as pd
from preprocessing import form_subject_dict

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