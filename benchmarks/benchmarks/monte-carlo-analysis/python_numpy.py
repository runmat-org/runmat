#!/usr/bin/env python3
import numpy as np

def main() -> None:
    np.random.seed(0)
    M, T = 10_000_000, 256
    S0, mu, sigma = 100.0, 0.05, 0.20
    dt, K = 1.0 / 252.0, 100.0

    S = np.full((M, 1), S0, dtype=np.float32)
    sqrt_dt = np.sqrt(np.float32(dt))
    drift = np.float32((mu - 0.5 * sigma * sigma) * dt)
    scale = np.float32(sigma) * sqrt_dt

    for _ in range(T):
        Z = np.random.randn(M, 1).astype(np.float32)
        S = S * np.exp(drift + scale * Z)

    payoff = np.maximum(S - K, 0.0)
    price = payoff.mean() * np.exp(-mu * T * dt)
    print(f"RESULT_ok PRICE={float(price):.6f}")


if __name__ == "__main__":
    main()