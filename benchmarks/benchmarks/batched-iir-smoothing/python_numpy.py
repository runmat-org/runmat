#!/usr/bin/env python3
import numpy as np

def main() -> None:
    np.random.seed(0)
    M, T = 2_000_000, 4096
    alpha, beta = np.float32(0.98), np.float32(0.02)

    X = np.random.rand(M, T).astype(np.float32)
    Y = np.zeros((M, 1), dtype=np.float32)

    for t in range(T):
        Y = alpha * Y + beta * X[:, t:t+1]

    mean_y = Y.mean(dtype=np.float64)
    print(f"RESULT_ok MEAN={mean_y:.6e}")

if __name__ == "__main__":
    main() 