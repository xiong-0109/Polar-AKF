from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .configuration import load_config, validate_config
from .decoder import WaveformDecoder
from .dynamics import StateConditionedDynamics
from .encoder import MeasurementEncoder, PolarEncoder
from .filter import AdaptiveKalmanFilter
from .frontend import PolarFrontend


class PolarAKF(nn.Module):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = dict(
            load_config()
            if config is None
            else validate_config(config)
        )

        self.polar_frontend = PolarFrontend(
            angle_bins=self.config["polar_angle_bins"],
            radius_bins=self.config["polar_radius_bins"],
            radius_max=self.config["polar_radius_max"],
        )
        self.encoder = PolarEncoder(
            in_channels=2,
            feature_dim=self.config["feature_dim"],
            base_channels=self.config["encoder_base_channels"],
        )
        self.measurement_encoder = MeasurementEncoder(
            feature_dim=self.config["feature_dim"],
            latent_dim=self.config["latent_dim"],
        )
        self.state_conditioned_dynamics = StateConditionedDynamics(
            state_dim=self.config["latent_dim"],
            hidden_dim=self.config["hidden_dim"],
            delta_t=self.config["delta_t"],
        )
        self.latent_filter = AdaptiveKalmanFilter(
            state_dim=self.config["latent_dim"],
            q_min=self.config["q_min"],
            r_min=self.config["r_min"],
            initial_q=self.config["initial_q"],
            initial_r=self.config["initial_r"],
            alpha_q=self.config["alpha_q"],
            alpha_r=self.config["alpha_r"],
            eps=self.config["eps"],
        )
        self.waveform_decoder = WaveformDecoder(self.config["latent_dim"])

    def _validate_input(self, x: torch.Tensor) -> None:
        expected = (
            self.config["sequence_length"],
            1,
            self.config["input_size"],
            self.config["input_size"],
        )
        if not torch.is_tensor(x):
            raise TypeError("input must be a torch.Tensor")
        if x.dtype != torch.float32:
            raise TypeError(f"input dtype must be torch.float32, got {x.dtype}")
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected [batch, {', '.join(map(str, expected))}], "
                f"got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        polar_pair = self.polar_frontend(x)
        batch_size, sequence_length = polar_pair.shape[:2]
        flat = polar_pair.reshape(
            batch_size * sequence_length, *polar_pair.shape[2:]
        )
        features = self.encoder(flat).reshape(
            batch_size, sequence_length, self.config["feature_dim"]
        )
        measurements = self.measurement_encoder(features)
        posterior = self.latent_filter(
            measurement_sequence=measurements,
            dynamics=self.state_conditioned_dynamics,
        )
        return self.waveform_decoder(posterior).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        if x.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self._forward_fp32(x.float())
        return self._forward_fp32(x)

