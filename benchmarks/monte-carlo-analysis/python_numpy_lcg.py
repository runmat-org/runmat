#!/usr/bin/env python3
import os
import numpy as np

def main() -> None:
    M, T = 10_000_000, 256
    M = int(os.environ.get("MC_M", M))
    T = int(os.environ.get("MC_T", T))
    S0, mu, sigma = 100.0, 0.05, 0.20
    dt, K = 1.0 / 252.0, 100.0

    S = np.full((M, 1), S0, dtype=np.float32)
    sqrt_dt = np.sqrt(np.float32(dt))
    drift = np.float32((mu - 0.5 * sigma * sigma) * dt)
    scale = np.float32(sigma) * sqrt_dt

    rid = np.arange(M, dtype=np.float64)[:, None]
    two_m = float(M) * 2.0
    for t in range(T):
        salt = float(t) * two_m
        idx1 = rid + salt
        idx2 = rid + salt + float(M)
        state1 = np.mod(1664525.0 * idx1 + 1013904223.0, 4294967296.0)
        state2 = np.mod(1664525.0 * idx2 + 1013904223.0, 4294967296.0)
        u1 = np.maximum(state1 / 4294967296.0, 1.0 / 4294967296.0)
        u2 = state2 / 4294967296.0
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        Z = (r * np.cos(theta)).astype(np.float32)
        S = S * np.exp(drift + scale * Z)

    payoff = np.maximum(S - K, 0.0)
    price = payoff.mean() * np.exp(-mu * T * dt)
    print(f"RESULT_ok PRICE={float(price):.6f}")


if __name__ == "__main__":
    main()