# models.py

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        self.last_weights = None

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        self.last_weights = weights.detach()
        return torch.sum(weights * x, dim=1)


class CNNBiGRUAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config["dataset"]["input_channels"]
        num_classes = config["dataset"]["num_classes"]

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(32)

        self.conv4 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.bn5 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = Attention(hidden_size=64)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = x.transpose(1, 2)
        x, _ = self.bigru(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual
        x = self.relu(x)

        return x


class ResNet1D(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config["dataset"]["input_channels"]
        num_classes = config["dataset"]["num_classes"]

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

        self.block1 = ResBlock(16, 16)
        self.block2 = ResBlock(16, 32, stride=2)
        self.block3 = ResBlock(32, 64, stride=2)
        self.block4 = ResBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
class CNNStem(nn.Module):
    """Downsamples (B, 12, F, T) spectrogram into a coarse feature grid."""
    def __init__(self, in_channels=12, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, embed_dim, F', T')

class CNNViT(nn.Module):
    def __init__(
        self,
        config,
        input_shape=(33, 94),   # (F, T) from your STFT output — pass this in instead of num_tokens
        embed_dim=128,
        num_heads=4,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        # TODO: derive input_shape from STFT_PARAMS
        super().__init__()
        
        in_channels = config["dataset"]["input_channels"]
        num_classes = config["dataset"]["num_classes"]


        self.stem = CNNStem(in_channels, embed_dim)

        # Dummy forward pass to infer num_tokens from actual (F, T)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            out = self.stem(dummy)
            num_tokens = out.shape[2] * out.shape[3]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)


def get_model(config):
    model_name = config["model"]["name"]

    if model_name == "cnn1d":
        return CNNBiGRUAttention(config)

    if model_name == "resnet":
        return ResNet1D(config)
    
    if model_name== "cnn_vit":
        return CNNViT(config) 

    raise ValueError(f"Unknown model name: {model_name}")
