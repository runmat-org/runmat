#!/usr/bin/env python3
import os
import numpy as np

def main() -> None:
    np.random.seed(0)
    B, H, W = 16, 2160, 3840
    # Env overrides for suite
    B = int(os.environ.get("IMG_B", B))
    H = int(os.environ.get("IMG_H", H))
    W = int(os.environ.get("IMG_W", W))
    gain, bias, gamma, eps0 = np.float32(1.0123), np.float32(-0.02), np.float32(1.8), np.float32(1e-6)

    # Deterministic pseudo-random field (LCG-based)
    bid = np.arange(B, dtype=np.uint32)[:, None, None]
    yid = np.arange(H, dtype=np.uint32)[None, :, None]
    xid = np.arange(W, dtype=np.uint32)[None, None, :]
    strideHW = np.uint32(H * W)
    strideW = np.uint32(W)
    seed32 = np.uint32(0)
    idx = bid * strideHW + yid * strideW + xid + seed32
    state = (np.uint32(1664525) * idx + np.uint32(1013904223)).astype(np.uint32)
    imgs = (state.astype(np.float32) / np.float32(2.0**32))
    mu = imgs.mean(axis=(1, 2), keepdims=True)
    sigma = np.sqrt(((imgs - mu) ** 2).mean(axis=(1, 2), keepdims=True) + eps0)

    out = ((imgs - mu) / sigma) * gain + bias
    # Clamp to avoid fractional power of negative numbers producing NaN
    out = np.maximum(out, 0.0)
    out = out ** gamma
    mse = ((out - imgs) ** 2).mean(dtype=np.float64)
    print(f"RESULT_ok MSE={mse:.6e}")

if __name__ == "__main__":
    main()