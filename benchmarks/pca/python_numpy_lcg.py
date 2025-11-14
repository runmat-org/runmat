#!/usr/bin/env python3
import os
import numpy as np

def _lcg_matrix(rows: int, cols: int, seed: float, offset: float = 0.0) -> np.ndarray:
    rid = np.arange(rows, dtype=np.float64)[:, None]
    cid = np.arange(cols, dtype=np.float64)[None, :]
    idx = rid * np.float64(cols) + cid + seed + offset
    state = np.mod(1664525.0 * idx + 1013904223.0, 4294967296.0)
    return (state.astype(np.float32) / np.float32(4294967296.0))


def main() -> None:
    n, d, k, iters = 200_000, 1024, 8, 15
    n = int(os.environ.get("PCA_N", n))
    d = int(os.environ.get("PCA_D", d))
    k = int(os.environ.get("PCA_K", k))
    iters = int(os.environ.get("PCA_ITERS", iters))

    A = _lcg_matrix(n, d, 0.0)
    mu = A.mean(axis=0, keepdims=True)
    A = A - mu
    G = (A.T @ A) / np.float32(n - 1)

    offset = float(n * d)
    Q = _lcg_matrix(d, k, 0.0, offset)
    Q, _ = np.linalg.qr(Q, mode='reduced')

    for _ in range(iters):
        Q = G @ Q
        Q, _ = np.linalg.qr(Q, mode='reduced')

    Lambda = np.diag(Q.T @ G @ Q).astype(np.float64)
    explained = Lambda / np.sum(np.diag(G).astype(np.float64))
    print(f"RESULT_ok EXPLAINED1={explained[0]:.4f} TOPK_SUM={Lambda.sum():.6e}")


if __name__ == "__main__":
    main()