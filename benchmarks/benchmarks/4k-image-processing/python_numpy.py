#!/usr/bin/env python3
import numpy as np

def main() -> None:
    np.random.seed(0)
    B, H, W = 16, 2160, 3840
    gain, bias, gamma, eps0 = np.float32(1.0123), np.float32(-0.02), np.float32(1.8), np.float32(1e-6)

    imgs = np.random.rand(B, H, W).astype(np.float32)
    mu = imgs.mean(axis=(1, 2), keepdims=True)
    sigma = np.sqrt(((imgs - mu) ** 2).mean(axis=(1, 2), keepdims=True) + eps0)

    out = ((imgs - mu) / sigma) * gain + bias
    out = out ** gamma
    mse = ((out - imgs) ** 2).mean(dtype=np.float64)
    print(f"RESULT_ok MSE={mse:.6e}")

if __name__ == "__main__":
    main()