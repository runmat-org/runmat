#!/usr/bin/env python3
import os
from typing import Optional, Tuple

import numpy as np
import torch


def _shape_from_tshirt(label: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not label:
        return None
    normalized = label.strip().lower()
    mapping = {
        "s": (128, 512, 200),
        "small": (128, 512, 200),
        "m": (128, 2048, 200),
        "medium": (128, 2048, 200),
        "l": (256, 4096, 200),
        "large": (256, 4096, 200),
    }
    return mapping.get(normalized)


def resolve_params(default_p: int, default_c: int, default_t: int) -> Tuple[int, int, int]:
    resolved = _shape_from_tshirt(os.environ.get("NLMS_TSHIRT"))
    if resolved:
        return resolved
    p = int(os.environ.get("NLMS_P", default_p))
    c = int(os.environ.get("NLMS_C", default_c))
    t = int(os.environ.get("NLMS_T", default_t))
    return p, c, t


def lcg_matrix(p: int, c: int, t: int) -> np.ndarray:
    rid = np.arange(p, dtype=np.uint32)[:, None]
    cid = np.arange(c, dtype=np.uint32)[None, :]
    salt = np.uint32(t) * np.uint32(p * c)
    idx = rid * np.uint32(c) + cid + salt + np.uint32(0)
    state = (np.uint32(1664525) * idx + np.uint32(1013904223)).astype(np.uint32)
    return (state.astype(np.float32) / np.float32(4294967296.0)).astype(np.float32)


def main() -> None:
    torch.manual_seed(0)
    override = os.environ.get("TORCH_DEVICE")
    if override:
        device = torch.device(override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, C, T = resolve_params(128, 2048, 200)
    mu = torch.tensor(0.5, dtype=torch.float32, device=device)
    eps0 = torch.tensor(1e-3, dtype=torch.float32, device=device)

    W = torch.zeros((p, C), device=device, dtype=torch.float32)

    for t in range(T):
        x_np = lcg_matrix(p, C, t)
        x = torch.from_numpy(x_np).to(device)
        d = torch.sum(x * x, dim=0)
        y = torch.sum(x * W, dim=0)
        e = d - y
        nx = torch.sum(x.pow(2), dim=0) + eps0
        scale = (e / nx).unsqueeze(0)
        W = W + mu * x * scale

    mse = torch.mean((d - torch.sum(x * W, dim=0)).pow(2)).double().item()
    dev = "cuda" if device.type == "cuda" else "cpu"
    print(f"RESULT_ok MSE={mse:.6e} device={dev}")


if __name__ == "__main__":
    main()

