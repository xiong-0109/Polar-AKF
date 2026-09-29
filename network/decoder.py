import torch
import torch.nn as nn


class WaveformDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, state_code: torch.Tensor) -> torch.Tensor:
        return self.linear(state_code)

