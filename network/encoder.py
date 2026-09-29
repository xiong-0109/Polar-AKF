import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleCircularConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=(0, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 0, 1, 1), mode="circular"))


class PolarEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            AngleCircularConv2d(in_channels, base_channels),
            nn.GELU(),
            AngleCircularConv2d(base_channels, base_channels * 2),
            nn.GELU(),
            AngleCircularConv2d(base_channels * 2, base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeasurementEncoder(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

