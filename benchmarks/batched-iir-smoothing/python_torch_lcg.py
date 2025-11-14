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
    M, T = resolve_shape(2_000_000, 4096)
    alpha = torch.tensor(0.98, dtype=torch.float32)
    beta = torch.tensor(0.02, dtype=torch.float32)

    rid = torch.arange(M, dtype=torch.float64).view(M, 1)
    modulus = torch.tensor(4294967296.0, dtype=torch.float64)
    Y = torch.zeros((M, 1), dtype=torch.float32)

    for t in range(T):
        salt = float(t) * float(M)
        idx = rid + salt
        state = torch.fmod(1664525.0 * idx + 1013904223.0, modulus)
        x = (state.float() / 4294967296.0).to(torch.float32)
        Y = alpha * Y + beta * x

    mean_y = float(Y.double().mean())
    print(f"RESULT_ok MEAN={mean_y:.6e} device=cpu")


if __name__ == "__main__":
    main()