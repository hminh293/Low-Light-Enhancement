import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic blocks
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = ConvBlock(channels, channels)

    def forward(self, x):
        return x + self.conv(x)


# -----------------------------
# MIRNetV2 (simplified)
# -----------------------------
class MIRNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super().__init__()

        self.entry = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        self.res1 = ResidualBlock(num_features)
        self.res2 = ResidualBlock(num_features)
        self.res3 = ResidualBlock(num_features)

        self.exit = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.entry(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)
        out = self.exit(feat)
        return torch.clamp(out, 0.0, 1.0)
