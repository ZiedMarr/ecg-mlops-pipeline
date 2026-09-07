# transforms.py

import torch


def stft_transform(n_fft, hop_length, win_length):
    window = torch.hann_window(win_length)
    eps = 1e-8

    def transform(signal):
        spec = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window.to(signal.device, signal.dtype),
            center=True,
            return_complex=True,
        )
        return torch.log(spec.abs() + eps)

    return transform
