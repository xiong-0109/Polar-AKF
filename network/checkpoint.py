from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn


def load_polar_akf_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint must be a non-empty state_dict mapping")
    if not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise TypeError("checkpoint must contain only string keys and tensor values")

    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    supplied_keys = set(state)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint keys do not match model; missing={missing}, unexpected={unexpected}"
        )

    for key, expected_tensor in expected_state.items():
        supplied_tensor = state[key]
        if tuple(supplied_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"shape mismatch for {key}: expected {tuple(expected_tensor.shape)}, "
                f"got {tuple(supplied_tensor.shape)}"
            )
        if supplied_tensor.dtype != expected_tensor.dtype:
            raise TypeError(
                f"dtype mismatch for {key}: expected {expected_tensor.dtype}, "
                f"got {supplied_tensor.dtype}"
            )
        if not supplied_tensor.is_floating_point():
            raise TypeError(f"checkpoint tensor {key} must be floating point")
        if not torch.isfinite(supplied_tensor).all():
            raise ValueError(f"checkpoint tensor {key} contains NaN or Inf")

    model.load_state_dict(state, strict=True)

