#!/usr/bin/env python3
import torch

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, T = 2_000_000, 4096
    alpha = torch.tensor(0.98, dtype=torch.float32, device=device)
    beta = torch.tensor(0.02, dtype=torch.float32, device=device)

    X = torch.rand((M, T), device=device, dtype=torch.float32)
    Y = torch.zeros((M, 1), device=device, dtype=torch.float32)

    for t in range(T):
        Y = alpha * Y + beta * X[:, t:t+1]

    mean_y = Y.mean().double().item()
    dev = "cuda" if device.type == "cuda" else "cpu"
    print(f"RESULT_ok MEAN={mean_y:.6e} device={dev}")


if __name__ == "__main__":
    main()