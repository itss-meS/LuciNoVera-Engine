import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, channels, growth=32):
        super().__init__()
        self.c1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.c3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.c4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.c5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, channels, growth=32):
        super().__init__()
        self.d1 = DenseBlock(channels, growth)
        self.d2 = DenseBlock(channels, growth)
        self.d3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.d1(x)
        out = self.d2(out)
        out = self.d3(out)
        return x + out * 0.2


class WindowAttention(nn.Module):
    def __init__(self, channels, window=8, heads=4):
        super().__init__()
        self.window = window
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        xp = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x
        Hp, Wp = xp.shape[2], xp.shape[3]

        qkv = self.qkv(xp)
        qkv = qkv.view(B, 3, self.heads, C // self.heads, Hp // w, w, Wp // w, w)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3).reshape(3, -1, self.heads, w * w, C // self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.view(B, Hp // w, Wp // w, self.heads, w, w, C // self.heads)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return x + out


class WindowAttentionBlock(nn.Module):
    def __init__(self, channels, window=8, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = WindowAttention(channels, window, heads)

    def forward(self, x):
        return self.attn(self.norm(x))


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))
