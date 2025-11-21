#!/usr/bin/env python3
import os
from typing import Optional, Tuple

import numpy as np


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
    np.random.seed(0)
    M, T = resolve_shape(2_000_000, 4096)
    alpha, beta = np.float32(0.98), np.float32(0.02)

    X = np.random.rand(M, T).astype(np.float32)
    Y = np.zeros((M, 1), dtype=np.float32)

    for t in range(T):
        Y = alpha * Y + beta * X[:, t:t+1]

    mean_y = Y.mean(dtype=np.float64)
    print(f"RESULT_ok MEAN={mean_y:.6e}")

if __name__ == "__main__":
    main() 