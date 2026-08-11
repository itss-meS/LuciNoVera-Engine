import torch
import torch.nn as nn


class NoiseEstimator(nn.Module):
    """Estimates a per-image degradation embedding from shallow features."""

    def __init__(self, in_channels, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        f = self.net(x).flatten(1)
        return self.fc(f)


class FiLM(nn.Module):
    """Feature-wise linear modulation: injects the noise embedding into feature maps."""

    def __init__(self, embed_dim, channels):
        super().__init__()
        self.to_scale = nn.Linear(embed_dim, channels)
        self.to_shift = nn.Linear(embed_dim, channels)

    def forward(self, feat, embed):
        scale = torch.tanh(self.to_scale(embed)).unsqueeze(-1).unsqueeze(-1)
        shift = torch.tanh(self.to_shift(embed)).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + scale) + shift
