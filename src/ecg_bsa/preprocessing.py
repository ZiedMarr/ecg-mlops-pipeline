# preprocessing.py

import os
import numpy as np
import wfdb
import pandas as pd
import neurokit2 as nk
from src.ecg_bsa.config import get_config
from math import gcd
from scipy.signal import resample_poly


cfg = get_config()
sampling_rate = cfg["dataset"]["sampling_rate"]


def preprocess_dataset(config):
    #set paths
    raw_path = config["paths"]["raw_data"]
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")
    reference_df = pd.read_csv(ref_path)
    save_path = config["paths"]["processed_data"]
    clear_processed_files(save_path)

    #set number of subjects
    num_pat = config["raw_dataset"]["num_subjects"]
    segment_size = config["dataset"]["segment_length"]
    overlap_ratio = config["preprocess"]["overlap_ratio"]

    print(f"Segment length: {segment_size} samples")
    print(f"Using overlap ratio: {overlap_ratio}")

    subjects_processed = 0
    subjects_skipped = 0
    total_segments = 0

    for i in range(1, num_pat +1):
        subject_id = f"A{i:04d}"
        print(f"Processing {subject_id} ({i}/{num_pat})")
        try:
            subject_dict = form_subject_dict(signals_path=signals_path, reference_df=reference_df, subject_number=i, config=config)
            subject_dict["signals"] = preprocessing_pipeline(subject_dict["signals"], config=config)
            segmented_list = segmentation(subject_dict=subject_dict, segment_size=segment_size, overlap_ratio=overlap_ratio)
            segments_saved = save_subject_file(segmented_list, config=config)
        except Exception as e:
            print(f"Skipping {subject_id}: {e}")
            subjects_skipped += 1
            continue

        if segments_saved == 0:
            subjects_skipped += 1
            print(f"Skipping {subject_id}: no segments created")
            continue

        subjects_processed += 1
        total_segments += segments_saved

    print("Pre-processing completed")
    print(f"Subjects processed: {subjects_processed}")
    print(f"Subjects skipped: {subjects_skipped}")
    print(f"Total segments saved: {total_segments}")
    validate_processed_files(save_path, config)
    print_processed_label_statistics(save_path, config)


def form_subject_dict(signals_path, reference_df=None, subject_number=None, config=None, ref_path=None):
    """
    returns: 
        a directory that contains : subject_id, signals, labels
    """
    #set file name
    subject_id = f"A{subject_number:04d}"
    #set signal file path
    sig_file_path = os.path.join(signals_path, subject_id)
    #get signals
    signals, _ = unpack_signal(sig_file_path)
    #get labels
    if isinstance(reference_df, str):
        reference_df = pd.read_csv(reference_df)
    elif reference_df is None:
        reference_df = pd.read_csv(ref_path)
    labels = get_labels(reference_df, subject_number, config=config)
    # form subject dict:
    subject_dict = { "subject_id" :  subject_id , "signals" : signals , "labels" : labels}
        
    return subject_dict


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


def get_labels(reference_df, idx, config=None):
    """
    for an index (corresponds to a subject), returns the labels as a multi-hot vector
    """
    if config is None:
        config = cfg
    #get the labels
    row = reference_df.iloc[idx-1]
    labels = row[["First_label", "Second_label", "Third_label"]].dropna().astype(int).tolist()
    multi_hot_labels = np.zeros(config["dataset"]["num_classes"], dtype=np.float32)

    for label in labels:
        multi_hot_labels[label - 1] = 1
    
    return  multi_hot_labels


def preprocessing_pipeline(signal, config=None):
    #TODO : test
    """
    Applies Filters channel-wise
    """
    if config is None:
        config = cfg
    # Notch Filter : removing Powerline 
    filtered = _notch(signal, config)
    # bandpass butterworth filter
    filtered = _butterworth(filtered, config)
    #downsample signal
    down_sampeled = _downsample(filtered, config)
    #z_score_norm
    normalized = _z_score_norm(down_sampeled)
    

    return normalized


def _baseline_wander_remove(signal, config=None):
    if config is None:
        config = cfg
    sampling_rate = config["dataset"]["sampling_rate"]
     # removing Baseline Wander
    filtered = np.stack([nk.signal_filter(signal[:,c], sampling_rate, method='savgol') for c in range(signal.shape[1])], axis=1)
    return filtered


def _notch(signal, config=None):
    if config is None:
        config = cfg
    sampling_rate = config["dataset"]["sampling_rate"]
    powerline = config["preprocess"]["powerline"]
     # Notch Filter : removing Powerline 
    filtered = np.stack([nk.signal_filter(signal[:,c], sampling_rate, method='powerline', powerline=powerline) for c in range(signal.shape[1])], axis=1)
    return filtered


def _butterworth(signal, config=None):
    if config is None:
        config = cfg
    sampling_rate = config["dataset"]["sampling_rate"]
    lowcut = config["preprocess"]["lowcut"]
    highcut = config["preprocess"]["highcut"]
    # bandpass butterworth filter
    filtered = np.stack([nk.signal_filter(signal[:,c], sampling_rate, lowcut=lowcut, highcut=highcut, method='butterworth') for c in range(signal.shape[1])], axis=1)
    return filtered


def _downsample(signal, config=None):
    if config is None:
        config = cfg
    #get the fs rates
    original_fs = config["dataset"]["sampling_rate"]
    target_fs = config["preprocess"]["downsampled_rate"]
    # greatest common denominator
    g = gcd(original_fs, target_fs)
    # get the downsampling coefficients
    up = target_fs // g
    down = original_fs // g
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


def validate_processed_files(save_path, config):
    expected_channels = config["dataset"]["input_channels"]
    expected_segment_length = config["dataset"]["segment_length"]
    expected_num_classes = config["dataset"]["num_classes"]
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


def print_processed_label_statistics(save_path, config):
    num_classes = config["dataset"]["num_classes"]
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
