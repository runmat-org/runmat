#!/usr/bin/env python3
import os
from typing import Optional, Tuple

import torch


def _shape_from_tshirt(label: Optional[str]) -> Optional[Tuple[int, int]]:
    if not label:
        return None
    normalized = label.strip().lower()
    table = {
        "xs": (2048, 16),
        "xsmall": (2048, 16),
        "x-small": (2048, 16),
        "s": (32768, 128),
        "small": (32768, 128),
        "m": (131072, 512),
        "medium": (131072, 512),
        "l": (524288, 2048),
        "large": (524288, 2048),
        "xl-lite": (1_000_000, 1536),
        "xllite": (1_000_000, 1536),
        "xl_lite": (1_000_000, 1536),
        "xl-mid": (1_500_000, 3072),
        "xlmid": (1_500_000, 3072),
        "xl_mid": (1_500_000, 3072),
        "xl": (2_000_000, 4096),
        "xlarge": (2_000_000, 4096),
        "full": (2_000_000, 4096),
    }
    return table.get(normalized)


def resolve_shape(default_m: int, default_t: int) -> Tuple[int, int]:
    tshirt = os.environ.get("IIR_TSHIRT")
    resolved = _shape_from_tshirt(tshirt)
    if resolved:
        return resolved
    m = int(os.environ.get("IIR_M", default_m))
    t = int(os.environ.get("IIR_T", default_t))
    return m, t


def main() -> None:
    torch.manual_seed(0)
    override = os.environ.get("TORCH_DEVICE")
    if override:
        device = torch.device(override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, T = resolve_shape(2_000_000, 4096)
    alpha = torch.tensor(0.98, dtype=torch.float32, device=device)
    beta = torch.tensor(0.02, dtype=torch.float32, device=device)

    X = torch.rand((M, T), device=device, dtype=torch.float32)
    Y = torch.zeros((M, 1), device=device, dtype=torch.float32)

    for t in range(T):
        Y = alpha * Y + beta * X[:, t:t+1]

    mean_y = Y.mean().double().item()
    dev = "cuda" if device.type == "cuda" else "cpu"
    print(f"RESULT_ok MEAN={mean_y:.6e} device={dev}")


if __name__ == "__main__":
    main()