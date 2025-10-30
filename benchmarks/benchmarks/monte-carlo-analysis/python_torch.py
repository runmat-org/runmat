#!/usr/bin/env python3
import torch

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, T = 10_000_000, 256
    S0, mu, sigma = 100.0, 0.05, 0.20
    dt, K = 1.0 / 252.0, 100.0

    S = torch.full((M, 1), S0, device=device, dtype=torch.float32)
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device, dtype=torch.float32))
    drift = torch.tensor((mu - 0.5 * sigma * sigma) * dt, device=device, dtype=torch.float32)
    scale = torch.tensor(sigma, device=device, dtype=torch.float32) * sqrt_dt

    for _ in range(T):
        Z = torch.randn((M, 1), device=device, dtype=torch.float32)
        S = S * torch.exp(drift + scale * Z)

    payoff = torch.clamp(S - K, min=0.0)
    price = payoff.mean() * torch.exp(torch.tensor(-mu * T * dt, device=device, dtype=torch.float32))
    print(f"RESULT_ok PRICE={float(price):.6f} device={'cuda' if device.type=='cuda' else 'cpu'}")


if __name__ == "__main__":
    main()