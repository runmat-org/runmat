#!/usr/bin/env python3
import os
import numpy as np

def main() -> None:
    np.random.seed(0)
    p, C, T = 128, 2048, 200
    p = int(os.environ.get("NLMS_P", p))
    C = int(os.environ.get("NLMS_C", C))
    T = int(os.environ.get("NLMS_T", T))
    mu = np.float32(0.5)
    eps0 = np.float32(1e-3)

    W = np.zeros((p, C), dtype=np.float32)

    for _ in range(T):
        x = np.random.rand(p, C).astype(np.float32)
        d = np.sum(x * x, axis=0)
        y = np.sum(x * W, axis=0)
        e = d - y
        nx = np.sum(x ** 2, axis=0) + eps0
        W = W + mu * x * (e / nx)

    mse = np.mean((d - np.sum(x * W, axis=0)) ** 2, dtype=np.float64)
    print(f"RESULT_ok MSE={mse:.6e}")


if __name__ == "__main__":
    main()