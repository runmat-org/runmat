#!/usr/bin/env python3
import torch

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, d, k, iters = 200_000, 1024, 8, 15

    A = torch.rand((n, d), device=device, dtype=torch.float32)
    mu = A.mean(dim=0, keepdim=True)
    A = A - mu
    G = (A.t() @ A) / float(n - 1)

    Q = torch.rand((d, k), device=device, dtype=torch.float32)
    Q = Q / (torch.norm(Q, dim=0, keepdim=True) + 1e-12)

    for _ in range(iters):
        Q = G @ Q
        Q = Q / (torch.norm(Q, dim=0, keepdim=True) + 1e-12)

    Lambda = torch.diag(Q.t() @ G @ Q).double()
    explained = Lambda / torch.sum(torch.diag(G).double())
    print(f"RESULT_ok EXPLAINED1={float(explained[0]):.4f} TOPK_SUM={float(Lambda.sum()):.6e} device={'cuda' if device.type=='cuda' else 'cpu'}")


if __name__ == "__main__":
    main()