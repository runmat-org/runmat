#!/usr/bin/env python3
import os
import numpy as np


def _shape_from_tshirt(label: str):
    mapping = {
        "s": (128, 512, 200),
        "small": (128, 512, 200),
        "m": (128, 2048, 200),
        "medium": (128, 2048, 200),
        "l": (256, 4096, 200),
        "large": (256, 4096, 200),
    }
    return mapping.get(label)


def resolve_params(default_p: int, default_c: int, default_t: int):
    tshirt = os.environ.get("NLMS_TSHIRT")
    if tshirt:
        normalized = tshirt.strip().lower()
        resolved = _shape_from_tshirt(normalized)
        if resolved:
            return resolved
    p = int(os.environ.get("NLMS_P", default_p))
    c = int(os.environ.get("NLMS_C", default_c))
    t = int(os.environ.get("NLMS_T", default_t))
    return p, c, t


def main() -> None:
    np.random.seed(0)
    p, C, T = resolve_params(128, 2048, 200)
    mu = np.float32(0.5)
    eps0 = np.float32(1e-3)

    W = np.zeros((p, C), dtype=np.float32)

    for t in range(T):
        # Deterministic per-index LCG using uint32 to match MATLAB/RunMat exactly
        rid = np.arange(p, dtype=np.uint32)[:, None]
        cid = np.arange(C, dtype=np.uint32)[None, :]
        salt = np.uint32(t) * np.uint32(p * C)
        idx = rid * np.uint32(C) + cid + salt + np.uint32(0)
        state = (np.uint32(1664525) * idx + np.uint32(1013904223)).astype(np.uint32)
        x = (state.astype(np.float32) / np.float32(4294967296.0)).astype(np.float32)
        d = np.sum(x * x, axis=0)
        y = np.sum(x * W, axis=0)
        e = d - y
        nx = np.sum(x ** 2, axis=0) + eps0
        W = W + mu * x * (e / nx)

    mse = np.mean((d - np.sum(x * W, axis=0)) ** 2, dtype=np.float64)
    print(f"RESULT_ok MSE={mse:.6e}")


if __name__ == "__main__":
    main()