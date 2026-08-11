import torch
import torch.nn as nn

from src.model.blocks import RRDB, WindowAttentionBlock, PixelShuffleUpsample
from src.model.noise_estimator import NoiseEstimator, FiLM


CONFIGS = {
    "small": dict(channels=32, n_rrdb=6, attn_every=3, window=8, heads=2),
    "medium": dict(channels=64, n_rrdb=12, attn_every=3, window=8, heads=4),
    "large": dict(channels=96, n_rrdb=20, attn_every=3, window=8, heads=6),
}


class RestorationNet(nn.Module):
    def __init__(self, config="medium", scale=2, in_channels=1, embed_dim=32):
        super().__init__()
        cfg = CONFIGS[config] if isinstance(config, str) else config
        C = cfg["channels"]
        self.scale = scale

        self.shallow = nn.Conv2d(in_channels, C, 3, 1, 1)
        self.noise_est = NoiseEstimator(C, embed_dim=embed_dim)
        self.film = FiLM(embed_dim, C)

        blocks = []
        for i in range(cfg["n_rrdb"]):
            blocks.append(RRDB(C))
            if (i + 1) % cfg["attn_every"] == 0:
                blocks.append(WindowAttentionBlock(C, window=cfg["window"], heads=cfg["heads"]))
        self.trunk = nn.ModuleList(blocks)
        self.trunk_conv = nn.Conv2d(C, C, 3, 1, 1)

        up_layers = []
        remaining = scale
        while remaining > 1:
            up_layers.append(PixelShuffleUpsample(C, scale=2))
            remaining //= 2
        self.upsample = nn.Sequential(*up_layers) if up_layers else nn.Identity()

        self.restore_head = nn.Conv2d(C, in_channels, 3, 1, 1)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(C, in_channels, 3, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        f0 = self.shallow(x)
        embed = self.noise_est(f0)
        feat = self.film(f0, embed)

        for block in self.trunk:
            feat = block(feat)
        feat = self.trunk_conv(feat)
        feat = feat + f0

        feat = self.upsample(feat)
        restored = self.restore_head(feat)
        confidence = self.confidence_head(feat)
        return restored, confidence

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
