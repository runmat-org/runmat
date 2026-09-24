"""European call, exact terminal GBM; shared float64 fixture, CPU only."""
import math
import os
import time

import numpy as np


def price_paths(z, mode):
    if mode == "loop":
        payoff = np.empty(len(z), dtype=np.float64)
        for i in range(len(z)):
            payoff[i] = math.exp(-0.03) * max(100 * math.exp(0.01 + 0.2 * float(z[i])) - 100, 0)
    elif mode == "vectorized":
        payoff = math.exp(-0.03) * np.maximum(100 * np.exp(0.01 + 0.2 * z) - 100, 0)
    else:
        raise ValueError("MC_MODE must be loop or vectorized")
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(len(z)))


def main():
    z = np.fromfile(os.environ["MC_FIXTURE"], dtype="<f8")
    if os.environ.get('MC_PRECISION') == 'float32':
        z = z.astype(np.float32)
    mode = os.environ.get("MC_MODE", "vectorized")
    repeats = int(os.environ.get("MC_REPEATS", "10"))
    warmups = int(os.environ.get("MC_WARMUPS", "3"))
    for rep in range(1, 1 + warmups + repeats):
        start = time.perf_counter()
        price, stderr = price_paths(z, mode)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"SAMPLE rep={rep} ms={elapsed:.12g} price={price:.17g} stderr={stderr:.17g}")


if __name__ == "__main__":
    main()
