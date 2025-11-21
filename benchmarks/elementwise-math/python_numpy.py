#!/usr/bin/env python3
import os
import numpy as np


def main() -> None:
    default_points = 5_000_001
    points = int(os.environ.get("ELM_POINTS", default_points))
    points = max(points, 2)

    x = np.linspace(0.0, 4.0 * np.pi, points, dtype=np.float32)
    y0 = np.sin(x) * np.exp(-x / 10.0)
    y1 = y0 * np.cos(x / 4.0) + 0.25 * np.square(y0)
    y2 = np.tanh(y1) + 0.1 * y1
    print("RESULT_ok")


if __name__ == "__main__":
    main()

