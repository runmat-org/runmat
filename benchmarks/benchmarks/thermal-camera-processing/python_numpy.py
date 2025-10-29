#!/usr/bin/env python3
import numpy as np

def main() -> None:
    np.random.seed(0)
    B, H, W = 16, 1024, 1024

    raw = np.random.rand(B, H, W).astype(np.float32)

    dark = 0.02 + 0.01 * np.random.rand(H, W).astype(np.float32)
    ffc = 0.98 + 0.04 * np.random.rand(H, W).astype(np.float32)
    gain = 1.50 + 0.50 * np.random.rand(H, W).astype(np.float32)
    offset = -0.05 + 0.10 * np.random.rand(H, W).astype(np.float32)

    lin = (raw - dark) * ffc
    radiance = lin * gain + offset
    radiance = np.maximum(radiance, 0.0)

    tempK = 273.15 + 80.0 * np.log1p(radiance)

    mean_temp = float(tempK.mean())
    print(f"RESULT_ok MEAN_TEMP={mean_temp:.6f}")


if __name__ == "__main__":
    main()