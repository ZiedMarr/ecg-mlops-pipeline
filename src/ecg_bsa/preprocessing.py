# preprocessing.py

import os
import numpy as np
import wfdb
import pandas as pd
import neurokit2 as nk
from ecg_bsa.config import get_config
from math import gcd
from scipy.signal import resample_poly


cfg = get_config()
sampling_rate = cfg["dataset"]["sampling_rate"]


def form_subject_dict(signals_path, reference_df, subject_number, expected_num_classes):
    subject_id = f"A{subject_number:04d}"
    sig_file_path = os.path.join(signals_path, subject_id)
    signals, _ = unpack_signal(sig_file_path)
    labels = get_labels(reference_df, subject_number, expected_num_classes)
    return {"subject_id": subject_id, "signals": signals, "labels": labels}



def clear_processed_files(save_path):
    os.makedirs(save_path, exist_ok=True)

    for file in os.listdir(save_path):
        if file.endswith(".npy"):
            os.remove(os.path.join(save_path, file))


def unpack_signal(file_path):
    """
    parameters: 
        file_path : path of the .hea file without .hea
    returns:
        signals from the .mat file as an np array of shape (N, C)
    """
        
    # Load record 
    record = wfdb.rdrecord(file_path)
    # Access the signal data as a NumPy array
    signals = record.p_signal        # physical units (mV, etc.)
    # signals = record.d_signal      # digital (raw) values
    # Metadata from the .hea file
    sig_len = record.sig_len  # number of samples

    return signals, sig_len


def get_labels(reference_df, idx, expected_num_classes):
    row = reference_df.iloc[idx - 1]
    labels = row[["First_label", "Second_label", "Third_label"]].dropna().astype(int).tolist()
    multi_hot_labels = np.zeros(expected_num_classes, dtype=np.float32)
    for label in labels:
        multi_hot_labels[label - 1] = 1
    return multi_hot_labels


def preprocessing_pipeline(signal, filter_params):
    filtered = _notch(signal, filter_params)
    filtered = _butterworth(filtered, filter_params)
    down_sampled = _downsample(filtered, filter_params)
    return _z_score_norm(down_sampled)



def _baseline_wander_remove(signal, config=None):
    if config is None:
        config = cfg
    sampling_rate = config["dataset"]["sampling_rate"]
     # removing Baseline Wander
    filtered = np.stack([nk.signal_filter(signal[:,c], sampling_rate, method='savgol') for c in range(signal.shape[1])], axis=1)
    return filtered


def _notch(signal, filter_params):
    sampling_rate = filter_params["sampling_rate"]
    powerline = filter_params["powerline"]
    return np.stack([nk.signal_filter(signal[:, c], sampling_rate, method='powerline', powerline=powerline)
                      for c in range(signal.shape[1])], axis=1)


def _butterworth(signal, filter_params):
    sampling_rate = filter_params["sampling_rate"]
    lowcut = filter_params["lowcut"]
    highcut = filter_params["highcut"]
    return np.stack([nk.signal_filter(signal[:, c], sampling_rate, lowcut=lowcut, highcut=highcut, method='butterworth')
                      for c in range(signal.shape[1])], axis=1)


def _downsample(signal, filter_params):
    original_fs = filter_params["sampling_rate"]
    target_fs = filter_params["downsampled_rate"]
    g = gcd(original_fs, target_fs)
    up, down = target_fs // g, original_fs // g
    return resample_poly(signal, up, down, axis=0)


def _z_score_norm(signal):
    #each lead normalized independently
    for ch in range(signal.shape[1]):
        mean = signal[:, ch].mean()
        std = signal[:, ch].std()
        if std == 0:
            signal[:, ch] = 0
        else:
            signal[:, ch] = (signal[:, ch] - mean) / std

    return signal


def segmentation(subject_dict : dict, segment_size=None, overlap_ratio=None):
    """
    segments the signals of a subject. Segment_size : to be found in config

    parameteres:
        subject_dict: gets a subject directory, with "signals" : (N, C)
    returns: 
        dicts_list: a list containing dictionnaries with "signals": segment of the original signal (N, C)
    
    """
    dicts_list = []
    
    signals = subject_dict["signals"]
    if segment_size is None:
        segment_size = cfg["dataset"]["segment_length"]
    if overlap_ratio is None:
        overlap_ratio = cfg["preprocess"]["overlap_ratio"]

    step_size = int(segment_size * (1 - overlap_ratio))
    if step_size <= 0:
        raise ValueError("overlap_ratio must be less than 1.")

    if signals.shape[0] < segment_size:
        pad_length = segment_size - signals.shape[0]
        signals = np.pad(signals, ((0, pad_length), (0, 0)), mode="constant")

    #split signal array into overlapping segments
    segments_list = [signals[start:start+segment_size, : ] for start in range(0, signals.shape[0] - segment_size + 1, step_size) ] 

    for segment in segments_list:
        segment = segment.T
        data_point = {"subject_id" :  subject_dict["subject_id"] , "signals" : segment , "labels" : subject_dict["labels"]}
        dicts_list.append(data_point)
    
    return dicts_list


def save_subject_file(subject_dict, config=None):
    if config is None:
        config = cfg
    save_path = config["paths"]["processed_data"]
    os.makedirs(save_path, exist_ok=True)

    if len(subject_dict) == 0:
        return 0

    subject_id = subject_dict[0]["subject_id"]
    signals = np.stack([data_point["signals"] for data_point in subject_dict])
    labels = np.array([data_point["labels"] for data_point in subject_dict])

    np.save(os.path.join(save_path, f"{subject_id}.npy"), {
        "signals": signals,
        "labels": labels,
        "subject_id": subject_id
    })

    return len(subject_dict)


def validate_processed_files(save_path, expected_channels = 12,
        expected_segment_length=1500,
        expected_num_classes = 9):
    valid_files = 0
    invalid_files = 0

    for file in os.listdir(save_path):
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(save_path, file)
        sample = np.load(file_path, allow_pickle=True).item()
        signals = sample["signals"]
        labels = sample["labels"]

        valid_signals = signals.ndim == 3 and signals.shape[1:] == (expected_channels, expected_segment_length)
        valid_labels = labels.ndim == 2 and labels.shape[0] == signals.shape[0] and labels.shape[1] == expected_num_classes

        if valid_signals and valid_labels:
            valid_files += 1
        else:
            invalid_files += 1
            print(f"Invalid processed file: {file}")
            print(f"  signals shape: {signals.shape}, expected: (num_segments, {expected_channels}, {expected_segment_length})")
            print(f"  labels shape: {labels.shape}, expected: (num_segments, {expected_num_classes})")

    print("Processed output validation completed")
    print(f"Valid processed files: {valid_files}")
    print(f"Invalid processed files: {invalid_files}")

    return valid_files, invalid_files


def print_processed_label_statistics(save_path, num_classes):
    label_segment_counts = np.zeros(num_classes, dtype=int)
    total_segments = 0

    for file in os.listdir(save_path):
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(save_path, file)
        sample = np.load(file_path, allow_pickle=True).item()
        labels = sample["labels"]

        total_segments += labels.shape[0]
        label_segment_counts += labels.sum(axis=0).astype(int)

    print("Processed label statistics")
    print(f"Total segments: {total_segments}")
    for label_idx, segment_count in enumerate(label_segment_counts, start=1):
        print(f"Label {label_idx}: {segment_count} segments")
