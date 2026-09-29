import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarFrontend(nn.Module):
    def __init__(
        self,
        angle_bins: int,
        radius_bins: int,
        radius_max: float,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.angle_bins = int(angle_bins)
        self.radius_bins = int(radius_bins)
        self.radius_max = float(radius_max)
        self.align_corners = bool(align_corners)

        theta = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=self.angle_bins + 1,
            dtype=torch.float32,
        )[:-1]
        radius = torch.linspace(
            0.0,
            self.radius_max,
            steps=self.radius_bins,
            dtype=torch.float32,
        )
        theta_grid, radius_grid = torch.meshgrid(theta, radius, indexing="ij")
        self.register_buffer("theta_grid", theta_grid, persistent=False)
        self.register_buffer("radius_grid", radius_grid, persistent=False)

    def _to_polar(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = x.shape
        if height != width:
            raise ValueError(
                f"polar projection requires square images, got {height}x{width}"
            )
        max_radius = min(float(width - 1), float(height - 1)) * 0.5
        if self.radius_max > max_radius + 1e-6:
            raise ValueError(
                f"polar_radius_max={self.radius_max} exceeds valid radius {max_radius}"
            )

        center_x = (float(width) - 1.0) * 0.5
        center_y = (float(height) - 1.0) * 0.5
        x_coords = center_x + self.radius_grid * torch.cos(self.theta_grid)
        y_coords = center_y + self.radius_grid * torch.sin(self.theta_grid)
        x_norm = 2.0 * x_coords / float(width - 1) - 1.0
        y_norm = 2.0 * y_coords / float(height - 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).to(x)
        grid = grid.unsqueeze(0).expand(batch_size * sequence_length, -1, -1, -1)

        flat_x = x.reshape(batch_size * sequence_length, channels, height, width)
        polar = F.grid_sample(
            flat_x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        return polar.reshape(
            batch_size,
            sequence_length,
            channels,
            self.angle_bins,
            self.radius_bins,
        )

    @staticmethod
    def _build_residual_pair(polar_raw: torch.Tensor) -> torch.Tensor:
        block_mean = polar_raw.mean(dim=1, keepdim=True)
        angle_main = block_mean.mean(dim=-1, keepdim=True)
        radial_main = block_mean.mean(dim=-2, keepdim=True)
        global_mean = block_mean.mean(dim=(-2, -1), keepdim=True)
        separable_reference = angle_main + radial_main - global_mean
        residual = polar_raw - separable_reference

        previous = torch.empty_like(residual)
        previous[:, 0] = residual[:, 0]
        if residual.shape[1] > 1:
            previous[:, 1:] = residual[:, :-1]
        return torch.cat((previous, residual), dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "expected [batch, sequence, channel, height, width], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[2] != 1:
            raise ValueError(f"expected one left-nostril channel, got {x.shape[2]}")
        return self._build_residual_pair(self._to_polar(x))

