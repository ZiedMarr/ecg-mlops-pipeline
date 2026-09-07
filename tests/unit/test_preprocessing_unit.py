
import numpy as np
from ecg_bsa.preprocessing import segmentation, get_labels, _z_score_norm, _downsample
import pytest

def make_signal(n_samples, n_channels, seed=0):
    """
    Build a synthetic (N_samples, C_channels) signal for preprocessing unit tests.
    Deterministic (seeded) so tests are reproducible.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_channels))

def make_subj_dict(n_samples, n_channels, seed=0, subject_id= "12", label="3"):
    """
    Build subject dictionnary using a synthetic signal created by make_signal()
    """
    signal = make_signal(n_samples, n_channels, seed)
    return {"subject_id": subject_id, "signals" : signal, "labels": label }


def test_segmentation_normal_case():
    n_samples = 1000
    n_channels = 3
    segment_size = 200
    overlap_ratio = 0.5

    subj = make_subj_dict(n_samples=n_samples, n_channels=n_channels)

    result = segmentation(subj, segment_size=segment_size, overlap_ratio=overlap_ratio)

    step_size = int(segment_size * (1 - overlap_ratio))
    expected_num_segments = (n_samples - segment_size) // step_size + 1

    assert len(result) == expected_num_segments

    for data_point in result:
        assert data_point["signals"].shape == (n_channels, segment_size)
        assert data_point["subject_id"] == subj["subject_id"]
        assert data_point["labels"] == subj["labels"]

def test_segmentation_invalid_overlap_ratio_raises():
    subj = make_subj_dict(n_samples=1000, n_channels=3)

    with pytest.raises(ValueError):
        segmentation(subj, segment_size=200, overlap_ratio=1.0)

import pandas as pd

def test_get_labels_all_three_present():
    reference_df = pd.DataFrame({
        "First_label": [3],
        "Second_label": [1],
        "Third_label": [5],
    })
    fake_config = {"dataset": {"num_classes": 9}}

    result = get_labels(reference_df, idx=1, config=fake_config)

    expected = np.zeros(9, dtype=np.float32)
    expected[3 - 1] = 1
    expected[1 - 1] = 1
    expected[5 - 1] = 1

    assert result.shape == (9,)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)

def test_get_labels_missing_second_and_third():
    reference_df = pd.DataFrame({
        "First_label": [4],
        "Second_label": [np.nan],
        "Third_label": [np.nan],
    })
    fake_config = {"dataset": {"num_classes": 9}}

    result = get_labels(reference_df, idx=1, config=fake_config)

    expected = np.zeros(9, dtype=np.float32)
    expected[4 - 1] = 1

    assert result.shape == (9,)
    np.testing.assert_array_equal(result, expected)

from math import gcd, ceil

def test_downsample_output_length_and_shape():
    n_samples = 500
    n_channels = 4
    signal = make_signal(n_samples=n_samples, n_channels=n_channels)

    fake_config = {
        "dataset": {"sampling_rate": 500},
        "preprocess": {"downsampled_rate": 250},
    }

    result = _downsample(signal, config=fake_config)

    original_fs = fake_config["dataset"]["sampling_rate"]
    target_fs = fake_config["preprocess"]["downsampled_rate"]
    g = gcd(original_fs, target_fs)
    up = target_fs // g
    down = original_fs // g
    expected_length = ceil(n_samples * up / down)

    assert result.shape == (expected_length, n_channels)

def test_z_score_norm_normal_channel():
    signal = make_signal(n_samples=1000, n_channels=1)

    result = _z_score_norm(signal.copy())

    assert np.isclose(result[:, 0].mean(), 0.0, atol=1e-8)
    assert np.isclose(result[:, 0].std(), 1.0, atol=1e-8)


def test_z_score_norm_zero_std_channel():
    n_samples = 100
    constant_signal = np.full((n_samples, 1), fill_value=5.0)

    result = _z_score_norm(constant_signal.copy())

    assert np.all(result[:, 0] == 0)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))

