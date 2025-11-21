# Elementwise Math Benchmark


This benchmark exercises an elementwise sequence representative of an arbitrary
image/signal pre-processing. For each vector `x` we compute:


```
y0 = sin(x) .* exp(-x / single(10));
y1 = y0 .* cos(x / 4) + single(0.25) .* (y0 .^ 2);
y2 = tanh(y1) + single(0.1) .* y1;
```

The scripts scale the number of samples via `ELM_POINTS` (default
5,000,001). Every implementation prints `RESULT_ok`.

---

## Results

![RunMat is up to 100x faster](https://web.runmatstatic.com/elementwise-math_speedup.svg)

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
