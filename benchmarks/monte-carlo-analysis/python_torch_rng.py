#!/usr/bin/env python3
import os
import torch

def resolve_device() -> torch.device:
    override = os.environ.get("TORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(0)
    device = resolve_device()
    M, T = 10_000_000, 256
    M = int(os.environ.get("MC_M", M))
    T = int(os.environ.get("MC_T", T))
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
    print(f"RESULT_ok PRICE={float(price):.6f} device={device.type}")


if __name__ == "__main__":
    main()