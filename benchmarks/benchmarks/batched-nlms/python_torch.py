#!/usr/bin/env python3
import torch

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, C, T = 128, 2048, 200
    mu = torch.tensor(0.5, dtype=torch.float32, device=device)
    eps0 = torch.tensor(1e-3, dtype=torch.float32, device=device)

    W = torch.zeros((p, C), device=device, dtype=torch.float32)

    for _ in range(T):
        x = torch.rand((p, C), device=device, dtype=torch.float32)
        d = torch.sum(x * x, dim=0)
        y = torch.sum(x * W, dim=0)
        e = d - y
        nx = torch.sum(x.pow(2), dim=0) + eps0
        scale = (e / nx).unsqueeze(0)  # shape (1, C)
        W = W + mu * x * scale

    mse = torch.mean((d - torch.sum(x * W, dim=0)).pow(2)).double().item()
    dev = "cuda" if device.type == "cuda" else "cpu"
    print(f"RESULT_ok MSE={mse:.6e} device={dev}")


if __name__ == "__main__":
    main()
