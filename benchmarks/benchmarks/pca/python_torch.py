#!/usr/bin/env python3
import os
import torch

def main() -> None:
    torch.manual_seed(0)
    override = os.environ.get("TORCH_DEVICE")
    if override:
        device = torch.device(override)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    n, d, k, iters = 200_000, 1024, 8, 15
    # Env overrides for sweeps
    n = int(os.environ.get("PCA_N", n))
    d = int(os.environ.get("PCA_D", d))
    k = int(os.environ.get("PCA_K", k))
    iters = int(os.environ.get("PCA_ITERS", iters))

    A = torch.rand((n, d), device=device, dtype=torch.float32)
    mu = A.mean(dim=0, keepdim=True)
    A = A - mu
    G = (A.t() @ A) / float(n - 1)

    Q = torch.rand((d, k), device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Q, mode='reduced')

    for _ in range(iters):
        Q = G @ Q
        Q, _ = torch.linalg.qr(Q, mode='reduced')

    Lambda = torch.diag(Q.t() @ G @ Q).double()
    explained = Lambda / torch.sum(torch.diag(G).double())
    devname = 'mps' if device.type == 'mps' else ('cuda' if device.type == 'cuda' else 'cpu')
    print(f"RESULT_ok EXPLAINED1={float(explained[0]):.4f} TOPK_SUM={float(Lambda.sum()):.6e} device={devname}")


if __name__ == "__main__":
    main()