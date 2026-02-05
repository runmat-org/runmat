# Monte Carlo GBM Risk Simulation

Geometric Brownian Motion (GBM) is a staple in intraday risk and options pricing. At realistic scales (millions of paths, hundreds of steps), throughput wins matter. This benchmark compares RunMat against NumPy, and PyTorch for a batched GBM simulation with a simple call option payoff.

We simulate `M` paths over `T` steps:

```matlab:runnable
M = 1000000; T = 256;
drift = (mu - 0.5 * sigma^2) * dt;
scale = sigma * sqrt(dt);

for t = 1:T
  Z = randn(M, 1, 'single');
  S = S .* exp(drift + scale .* Z);
end

payoff = max(S - K, 0);
price  = mean(payoff, 'all') * exp(-mu * T * dt);
```

---

## Results

![RunMat is up to 131x faster than NumPy](https://web.runmatstatic.com/monte-carlo-analysis_speedup-b.svg)

### Monte Carlo Perf Sweep 
| Simulation Paths (M) | RunMat (ms) | PyTorch (ms) | NumPy (ms) | NumPy ÷ RunMat | PyTorch ÷ RunMat |
|--------------------:|-----------:|-------------:|-----------:|---------------:|-----------------:|
| 250k   | 108.58 |   824.42 |  4,065.87 | 37.44× | 7.59× |
| 500k   | 136.10 |   900.11 |  8,206.56 | 60.30× | 6.61× |
| 1M     | 188.00 |   894.32 | 16,092.49 | 85.60× | 4.76× |
| 2M     | 297.65 | 1,108.80 | 32,304.64 |108.53× | 3.73× |
| 5M     | 607.36 | 1,697.59 | 79,894.98 |131.55× | 2.80× |

*250k = 250,000 paths, 1M = 1,000,000 paths, etc.*

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
- RunMat / Octave: [`runmat_rng.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/monte-carlo-analysis/runmat_rng.m)
- Python (NumPy): [`python_numpy_rng.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/monte-carlo-analysis/python_numpy_rng.py)
- Python (PyTorch): [`python_torch_rng.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/monte-carlo-analysis/python_torch_rng.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/monte-carlo-analysis/julia.jl)

---

## Why RunMat is fast (accelerate + fusion)

RunMat fuses elementwise stages and keeps tensors resident on device between steps, while random number generation and updates execute in large, coalesced kernels—a strong fit for GPUs. For the big picture on fusion and residency, see the [Introduction to RunMat on the GPU](https://runmat.com/docs/accelerate/fusion-intro) document.

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).
