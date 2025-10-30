# RunMat Benchmark: Batched IIR Smoothing (EMA) at Scale

If you operate IoT/telemetry or time‑series analytics, you’ve seen this pattern: millions of independent channels, each updated with a simple first‑order IIR (exponential moving average). CPUs struggle when the channel count and horizon grow. This benchmark compares RunMat against NumPy, PyTorch, and Julia on a large batched EMA.

In this article, we benchmark the performance of RunMat against NumPy, PyTorch, and Julia.

We update `M` channels over `T` steps with `Y = alpha .* Y + beta .* X(:, t)`, then report `mean(Y)`.

---

## Results

![Relative speed (higher is better), normalized to NumPy = 1×](../../results/batched_iir_smoothing_bar.png)

---

## Core implementation in RunMat (MATLAB‑syntax)

```matlab
rng(0);
M = 2_000_000; T = 4096;
alpha = single(0.98); beta = single(0.02);

X = rand(M, T, 'single');
Y = zeros(M, 1, 'single');

for t = 1:T
  Y = alpha .* Y + beta .* X(:, t);
end

mean_y = mean(Y);
fprintf('RESULT_ok MEAN=%.6e\n', double(mean_y));
```

Full sources:
- RunMat / Octave: [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-iir-smoothing/runmat.m)
- Python (NumPy): [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-iir-smoothing/python_numpy.py)
- Python (PyTorch): [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-iir-smoothing/python_torch.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-iir-smoothing/julia.jl)

---

## Why RunMat is fast (accelerate + fusion)

RunMat keeps the per‑step elementwise work on device and fuses map‑like chains, while the time loop parallelizes across channels—an ideal fit for GPUs. See the introduction for a deeper dive into fusion and device residency: [Introduction to RunMat GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md)

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).
