#!/usr/bin/env python3
import os
import torch


def lcg_matrix(rows: int, cols: int, seed: float, offset: float = 0.0) -> torch.Tensor:
    rid = torch.arange(rows, dtype=torch.float64).view(rows, 1)
    cid = torch.arange(cols, dtype=torch.float64).view(1, cols)
    idx = rid * float(cols) + cid + seed + offset
    state = torch.fmod(1664525.0 * idx + 1013904223.0, 4294967296.0)
    return (state.float() / 4294967296.0)


def main() -> None:
    n, d, k, iters = 200_000, 1024, 8, 15
    n = int(os.environ.get("PCA_N", n))
    d = int(os.environ.get("PCA_D", d))
    k = int(os.environ.get("PCA_K", k))
    iters = int(os.environ.get("PCA_ITERS", iters))

    A = lcg_matrix(n, d, 0.0).to(torch.float32)
    mu = A.mean(dim=0, keepdim=True)
    A = A - mu
    G = (A.t() @ A) / float(n - 1)

    offset = float(n * d)
    Q = lcg_matrix(d, k, 0.0, offset).to(torch.float32)
    Q, _ = torch.linalg.qr(Q, mode='reduced')

    for _ in range(iters):
        Q = G @ Q
        Q, _ = torch.linalg.qr(Q, mode='reduced')

    Lambda = torch.diag(Q.t() @ G @ Q).double()
    explained = Lambda / torch.sum(torch.diag(G).double())
    print(f"RESULT_ok EXPLAINED1={float(explained[0]):.4f} TOPK_SUM={float(Lambda.sum()):.6e} device=cpu")


if __name__ == "__main__":
    main()