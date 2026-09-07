from preprocessing import segmentation, unpack_signal, preprocessing_pipeline, form_subject_dict
import numpy as np
import torch
import os
from  matplotlib import pyplot as plt
from config import get_config
from transforms import stft_transform
from config import STFT_PARAMS
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytest

from transforms import stft_transform

FS = 250  # sampling rate AFTER downsampling in your preprocessing pipeline

# Candidate configs to compare: (n_fft, hop_length, win_length)
CONFIGS = [
    (128, 32, 128),   # current default
    (64, 16, 64),     # shorter window, literature-typical range
    (32, 8, 32),      # very short window, finest time localization
]

cfg = get_config()
sampling_rate = cfg["dataset"]["sampling_rate"]


def set_data_paths() :
    #set paths
    
    raw_path = cfg["paths"]["raw_data"]
    signals_path = os.path.join(raw_path, "Training_WFDB")
    ref_path = os.path.join(raw_path, "REFERENCE.csv")

    #set number of subjects
    num_pat = cfg["raw_dataset"]["num_subjects"]

    return signals_path, ref_path, num_pat

signals_path, ref_path, num_pat = set_data_paths()

def test_segmentation():
    signals = np.asarray([[1, 2],[3, 2],[5, 6], [3,5]])
    subj = {"subject_id" :  "12" , "signals" : signals , "labels" : "label"}
    di_list = segmentation(subj)
    print(di_list)

def test_unpack_signal() : 
    file_path = "./data/raw/Training_WFDB/A6851"
    record, _  = unpack_signal(file_path= file_path)
    print(record.shape)

def visualize_signal(signal, channel, preprocess=False):
    title = "raw"
    if preprocess:
        signal = preprocessing_pipeline(signal=signal)
        title = "preprocessed"
    t = np.arange(len(signal[:,channel])) / sampling_rate
    plt.figure()
    plt.plot(t, signal[:, channel])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.show(block= False)

"""
Compare STFT parameter choices on an already-loaded ECG segment.

Usage (e.g. from your test file, after loading a segment):

    sample_signal = torch.tensor(segments[0]["signals"])  # (12, segment_length)
    plot_stft_comparison(sample_signal, lead_idx=1)
"""




def plot_stft_comparison(sample_signal, lead_idx=1, fs=FS, configs=CONFIGS, save_path="./tests/data/stft_comparison.png"):
    """
    sample_signal: (12, T) tensor for one segment.
    """
    lead_signal = sample_signal[lead_idx]
    t_axis = np.arange(len(lead_signal)) / fs

    n_configs = len(configs)
    fig, axes = plt.subplots(n_configs + 1, 1, figsize=(10, 3 * (n_configs + 1)))

    axes[0].plot(t_axis, lead_signal.numpy(), linewidth=0.8)
    axes[0].set_title(f"Raw lead {lead_idx} waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    for i, (n_fft, hop, win) in enumerate(configs, start=1):
        transform = stft_transform(n_fft=n_fft, hop_length=hop, win_length=win)
        spec = transform(sample_signal)  # (12, F, T)
        spec_lead = spec[lead_idx].numpy()

        freqs = np.linspace(0, fs / 2, spec_lead.shape[0])
        times = np.linspace(0, len(lead_signal) / fs, spec_lead.shape[1])

        win_ms = win / fs * 1000
        hop_ms = hop / fs * 1000
        df = fs / n_fft

        ax = axes[i]
        im = ax.pcolormesh(times, freqs, spec_lead, shading="auto", cmap="magma")
        ax.set_ylim(0, 50)  # ECG-relevant band only
        ax.set_title(
            f"n_fft={n_fft}, hop={hop} ({hop_ms:.0f} ms), win={win} ({win_ms:.0f} ms), "
            f"\u0394f={df:.1f} Hz, shape={spec_lead.shape}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="log-magnitude")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    """
    subj = form_subject_dict(signals_path, ref_path, 1222)
    visualize_signal(subj["signals"], 11, True)
    visualize_signal(subj["signals"], 11, False)
    plt.pause(0.001)
    input("Press Enter to close...")
    """
    subj = form_subject_dict(signals_path, ref_path, 1222)
    subj["signals"] = preprocessing_pipeline(signal=subj["signals"])
    segments = segmentation(subject_dict=subj)

    sample_signal = torch.tensor(segments[0]["signals"])  # (12, segment_length)
    plot_stft_comparison(sample_signal)
    """
    transform = stft_transform(**STFT_PARAMS)
    out = transform(sample_signal)
    print(out.shape)  # (12, F, T)
    """

