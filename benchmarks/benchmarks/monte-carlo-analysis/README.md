# RunMat Benchmark: Monte Carlo GBM Risk Simulation

Geometric Brownian Motion (GBM) is a staple in intraday risk and options pricing. At realistic scales (millions of paths, hundreds of steps), throughput wins matter. This benchmark compares RunMat against NumPy, PyTorch, and Julia for a batched GBM simulation with a simple call option payoff.

We simulate `M` paths over `T` steps:

```
S ← S .* exp((μ − ½σ²)Δt + σ√Δt · Z),    Z ~ N(0, 1)
payoff = max(S − K, 0)
price  = mean(payoff) · exp(−μ T Δt)
```

---

## Results

![Relative speed (higher is better), normalized to NumPy = 1×](../../results/monte_carlo_analysis_bar.png)

---

## Core implementation in RunMat (MATLAB-syntax)

```matlab
rng(0);
M = 10_000_000; T = 256;
S0 = single(100); mu = single(0.05); sigma = single(0.20);
dt = single(1.0/252.0); K = single(100.0);

S = ones(M, 1, 'single') * S0;
sqrt_dt = sqrt(dt);
drift = (mu - 0.5 * sigma^2) * dt;
scale = sigma * sqrt_dt;

for t = 1:T
  Z = randn(M, 1, 'single');
  S = S .* exp(drift + scale .* Z);
end

payoff = max(S - K, 0);
price  = mean(payoff) * exp(-mu * T * dt);
fprintf('RESULT_ok PRICE=%.6f\n', double(price));
```

Full sources:
- RunMat / Octave: [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/monte-carlo-analysis/runmat.m)
- Python (NumPy): [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/monte-carlo-analysis/python_numpy.py)
- Python (PyTorch): [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/monte-carlo-analysis/python_torch.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/monte-carlo-analysis/julia.jl)

---

## Why RunMat is fast (accelerate + fusion)

RunMat fuses elementwise stages and keeps tensors resident on device between steps, while random number generation and updates execute in large, coalesced kernels—a strong fit for GPUs. For the big picture on fusion and residency, see the [Introduction to RunMat on the GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md) document.

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).
