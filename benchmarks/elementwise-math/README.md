# Elementwise Math Benchmark


This benchmark exercises an elementwise sequence representative of an arbitrary
image/signal pre-processing. For each vector `x` we compute:


```matlab:runnable
y0 = sin(x) .* exp(-x / single(10));
y1 = y0 .* cos(x / 4) + single(0.25) .* (y0 .^ 2);
y2 = tanh(y1) + single(0.1) .* y1;
```

The scripts scale the number of samples via `ELM_POINTS` (default
5,000,001). Every implementation prints `RESULT_ok`.

---

## Results

![RunMat is up to 144x faster](https://web.runmatstatic.com/elementwise-math_speedup-b.svg)


### Elementwise Math Perf Sweep (points)
| points | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 1M   | 145.15 | 856.41  |   72.39 | 0.50× | 5.90× |
| 2M   | 149.75 | 901.05  |   79.49 | 0.53× | 6.02× |
| 5M   | 145.14 | 1111.16 |  119.45 | 0.82× | 7.66× |
| 10M  | 143.39 | 1377.43 |  154.38 | 1.08× | 9.61× |
| 100M | 144.81 | 16,404.22 | 1,073.09 | 7.41× | 113.28× |
| 200M | 156.94 | 16,558.98 | 2,114.66 | 13.47× | 105.51× |
| 500M | 137.58 | 17,882.11 | 5,026.94 | 36.54× | 129.97× |
| 1B | 144.40 | 20,841.42 | 11,931.93 | 82.63× | 144.34× |

*M = 10⁶ elements, B = 10⁹ elements.*


---


Full sources: 

- [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/elementwise-math/runmat.m) – RunMat / MATLAB syntax implementation.
- [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/elementwise-math/python_numpy.py) – NumPy implementation.
- [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/elementwise-math/python_torch.py) – PyTorch implementation (uses MPS/CUDA when available).


---

## Why RunMat is fast (accelerate + fusion)

RunMat fuses elementwise stages and keeps tensors resident on device between steps, while random number generation and updates execute in large, coalesced kernels—a strong fit for GPUs. For the big picture on fusion and residency, see the [Introduction to RunMat on the GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md) document.

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).
