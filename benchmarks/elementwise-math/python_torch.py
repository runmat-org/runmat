#!/usr/bin/env python3
import os
import torch


def resolve_device() -> torch.device:
    override = os.environ.get("TORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    points = int(os.environ.get("ELM_POINTS", 5_000_001))
    points = max(points, 2)
    device = resolve_device()

    x = torch.linspace(0.0, 4.0 * torch.pi, steps=points, dtype=torch.float32, device=device)
    y0 = torch.sin(x) * torch.exp(-x / 10.0)
    y1 = y0 * torch.cos(x / 4.0) + 0.25 * torch.square(y0)
    y2 = torch.tanh(y1) + 0.1 * y1
    print(f"RESULT_ok device={device}")


if __name__ == "__main__":
    main()

