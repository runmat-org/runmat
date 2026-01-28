# Image Preprocessing, GPU‑accelerated

If you ship geospatial or vision workloads, you’ve likely written this stage countless times: standardize each 4K tile, apply a small radiometric correction, gamma‑correct, and run a quick QC metric. On CPUs this is fine—until the batch grows and your wall‑clock explodes. 

In this article, we benchmark the performance of RunMat against NumPy, and PyTorch.

The math is deliberately simple and realistic: compute a per‑image mean and standard deviation, normalize, apply a modest gain/bias and a gamma curve, then validate with a mean‑squared error.

---

## Results

![RunMat is 10x faster than NumPy](https://web.runmatstatic.com/4k-image-processing_speedup-b.svg)

### 4K Image Pipeline Perf Sweep (B = batch size)
| B | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|---|---:|---:|---:|---:|---:|
| 4  | 142.97 | 801.29 | 500.34 | 3.50× | 5.60× |
| 8  | 212.77 | 808.92 | 939.27 | 4.41× | 3.80× |
| 16 | 241.56 | 907.73 | 1783.47 | 7.38× | 3.76× |
| 32 | 389.25 | 1141.92 | 3605.95 | 9.26× | 2.93× |
| 64 | 683.54 | 1203.20 | 6958.28 | 10.18× | 1.76× |
---

## Core implementation in RunMat (MATLAB-syntax)

We'll use a simple pipeline: compute a per‑image mean and standard deviation, normalize, apply a modest gain/bias and a gamma curve, then validate with a mean‑squared error.

```matlab:runnable
rng(0); B=16; H=2160; W=3840;
gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

imgs = rand(B, H, W, 'single');
mu = mean(imgs, [2 3]);
sigma = sqrt(mean((imgs - mu).^2, [2 3]) + eps0);
out = ((imgs - mu) ./ sigma) * gain + bias;
out = out .^ gamma;
mse = mean((out - imgs).^2, 'all');

fprintf('RESULT_ok MSE=%.6e\n', double(mse));
```

Full sources:
- RunMat / Octave: [`runmat_rng.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/4k-image-processing/runmat_rng.m)
- Python (NumPy): [`python_numpy_rng.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/4k-image-processing/python_numpy_rng.py)
- Python (PyTorch): [`python_torch_rng.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/4k-image-processing/python_torch_rng.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/4k-image-processing/julia.jl)

Note: MATLAB’s license agreement restricts usage of their runtime for benchmarking, so we do not include MATLAB runs. If you have numbers, consider sharing them on GitHub Discussions.

---

## Why RunMat is fast (accelerate + fusion)

RunMat fuses elementwise stages and keeps tensors resident on device between steps, while random number generation and updates execute in large, coalesced kernels—a strong fit for GPUs. For the big picture on fusion and residency, see the [Introduction to RunMat on the GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md) document.

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).