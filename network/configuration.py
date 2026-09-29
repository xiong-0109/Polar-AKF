from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "polar_akf.yaml"
)

_INTEGER_KEYS = {
    "sequence_length",
    "input_size",
    "polar_angle_bins",
    "polar_radius_bins",
    "encoder_base_channels",
    "feature_dim",
    "hidden_dim",
    "latent_dim",
}
_FLOAT_KEYS = {
    "polar_radius_max",
    "delta_t",
    "q_min",
    "r_min",
    "initial_q",
    "initial_r",
    "alpha_q",
    "alpha_r",
    "eps",
}
_EXPECTED_KEYS = _INTEGER_KEYS | _FLOAT_KEYS


def validate_config(config: Mapping[str, Any]) -> dict[str, int | float]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a YAML mapping")
    supplied = set(config)
    missing = sorted(_EXPECTED_KEYS - supplied)
    unexpected = sorted(supplied - _EXPECTED_KEYS)
    if missing or unexpected:
        raise ValueError(
            f"config keys do not match; missing={missing}, "
            f"unexpected={unexpected}"
        )

    validated: dict[str, int | float] = {}
    for key in _INTEGER_KEYS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        validated[key] = value
    for key in _FLOAT_KEYS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric")
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{key} must be positive")
        validated[key] = value

    if validated["initial_q"] < validated["q_min"]:
        raise ValueError("initial_q must be at least q_min")
    if validated["initial_r"] < validated["r_min"]:
        raise ValueError("initial_r must be at least r_min")
    if not 0.0 <= validated["alpha_q"] <= 1.0:
        raise ValueError("alpha_q must be in [0, 1]")
    if not 0.0 <= validated["alpha_r"] <= 1.0:
        raise ValueError("alpha_r must be in [0, 1]")
    return validated


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> Mapping[str, int | float]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return MappingProxyType(validate_config(loaded))

