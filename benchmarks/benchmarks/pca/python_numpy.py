#!/usr/bin/env python3
import os
import numpy as np

def main() -> None:
    np.random.seed(0)
    n, d, k, iters = 200_000, 1024, 8, 15
    # Env overrides for sweeps
    n = int(os.environ.get("PCA_N", n))
    d = int(os.environ.get("PCA_D", d))
    k = int(os.environ.get("PCA_K", k))
    iters = int(os.environ.get("PCA_ITERS", iters))

    A = np.random.rand(n, d).astype(np.float32)
    mu = A.mean(axis=0, keepdims=True)
    A = A - mu
    G = (A.T @ A) / np.float32(n - 1)

    Q = np.random.rand(d, k).astype(np.float32)
    Q, _ = np.linalg.qr(Q, mode='reduced')

    for _ in range(iters):
        Q = G @ Q
        Q, _ = np.linalg.qr(Q, mode='reduced')

    Lambda = np.diag(Q.T @ G @ Q).astype(np.float64)
    explained = Lambda / np.sum(np.diag(G).astype(np.float64))
    print(f"RESULT_ok EXPLAINED1={explained[0]:.4f} TOPK_SUM={Lambda.sum():.6e}")


if __name__ == "__main__":
    main()