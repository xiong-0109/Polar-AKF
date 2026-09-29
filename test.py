import argparse
from pathlib import Path

import numpy as np
import torch

from network import (
    DEFAULT_CONFIG_PATH,
    PolarAKF,
    load_config,
    load_polar_akf_checkpoint,
)


DEFAULT_INPUT_PATH = Path("data/input/prepared_left_nostril.npy")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/polar_akf.pt")
DEFAULT_OUTPUT_PATH = Path("results/predictions/prediction.npy")


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_input(path: Path, sequence_length: int, input_size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32:
        raise TypeError(f"input dtype must be float32, got {array.dtype}")
    expected_shape = (sequence_length, 1, input_size, input_size)
    if array.shape != expected_shape:
        raise ValueError(
            f"input shape must be {list(expected_shape)}, got {tuple(array.shape)}"
        )
    if not np.isfinite(array).all():
        raise ValueError("input contains NaN or Inf")
    return array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Polar-AKF on a prepared sequence.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = _device_from_argument(args.device)
    config = load_config(args.config)
    input_array = _load_input(
        args.input,
        sequence_length=int(config["sequence_length"]),
        input_size=int(config["input_size"]),
    )

    model = PolarAKF(config)
    load_polar_akf_checkpoint(
        model,
        args.checkpoint,
        map_location="cpu",
    )
    model.to(device).eval()
    tensor = torch.from_numpy(input_array).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor)[0].cpu().numpy().astype(np.float32, copy=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction, allow_pickle=False)
    print(f"saved standardized waveform {prediction.shape} to {args.output}")


if __name__ == "__main__":
    main()

