#!/usr/bin/env python3
import os
import torch


def main() -> None:
    M, T = 10_000_000, 256
    M = int(os.environ.get("MC_M", M))
    T = int(os.environ.get("MC_T", T))
    S0, mu, sigma = 100.0, 0.05, 0.20
    dt, K = 1.0 / 252.0, 100.0

    S = torch.full((M, 1), S0, dtype=torch.float32)
    sqrt_dt = torch.sqrt(torch.tensor(dt, dtype=torch.float32))
    drift = torch.tensor((mu - 0.5 * sigma * sigma) * dt, dtype=torch.float32)
    scale = torch.tensor(sigma, dtype=torch.float32) * sqrt_dt

    rid = torch.arange(M, dtype=torch.float64).view(M, 1)
    two_m = float(M) * 2.0
    for t in range(T):
        salt = float(t) * two_m
        idx1 = rid + salt
        idx2 = rid + salt + float(M)
        state1 = torch.fmod(1664525.0 * idx1 + 1013904223.0, 4294967296.0)
        state2 = torch.fmod(1664525.0 * idx2 + 1013904223.0, 4294967296.0)
        u1 = torch.clamp(state1 / 4294967296.0, min=1.0 / 4294967296.0)
        u2 = state2 / 4294967296.0
        r = torch.sqrt(-2.0 * torch.log(u1))
        theta = 2.0 * torch.pi * u2
        Z = (r * torch.cos(theta)).to(torch.float32)
        S = S * torch.exp(drift + scale * Z)

    payoff = torch.clamp(S - K, min=0.0)
    discount = torch.exp(torch.tensor(-mu * T * dt, dtype=torch.float32))
    price = payoff.mean() * discount
    print(f"RESULT_ok PRICE={float(price):.6f} device=cpu")


if __name__ == "__main__":
    main()