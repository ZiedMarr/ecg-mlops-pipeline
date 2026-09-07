
import torch
from src.ecg_bsa.config import get_config
from src.ecg_bsa.models import get_model
from src.ecg_bsa.transforms import stft_transform
from src.ecg_bsa.config import STFT_PARAMS

config = get_config()
config["model"]["name"] = "cnn_vit"
config["training"]["device"] = "cpu"

# Build model
model = get_model(config)
model.eval()

# Build the same transform training.py would use
transform = stft_transform(**STFT_PARAMS)

# Fake a small batch of raw signals: (batch, 12, 1500)
batch_size = 2
raw_signals = torch.randn(batch_size, 12, 1500)

# Apply transform per-sample (mirrors what Dataset.__getitem__ does)
spectrograms = torch.stack([transform(sig) for sig in raw_signals])
print("Spectrogram batch shape:", spectrograms.shape)  # expect (2, 12, 65, 47)

# Forward pass
with torch.no_grad():
    out = model(spectrograms)

print("Output shape:", out.shape)   # expect (2, 9)
print("Output values:", out)
print("Any NaN:", torch.isnan(out).any().item())