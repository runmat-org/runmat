# RunMat Benchmark: Batched NLMS (Normalized LMS) at Scale

Normalized LMS (NLMS) is a classic adaptive filter used in echo cancellation and online system identification. In batched settings—thousands of parallel filters updated every step—it becomes a throughput problem. This benchmark compares RunMat against NumPy, PyTorch, and Julia for a large batched NLMS workload.

We update weights `W ∈ ℝ^{p×C}` for `C` parallel columns over `T` steps. At each step:

```
d = sum(x .* x, 1)
y = sum(x .* W, 1)
e = d - y
nx = sum(x.^2, 1) + eps0
W = W + μ * x .* (e ./ nx)
```

and we report an `MSE` against the most recent step.

---

## Results

![Relative speed (higher is better), normalized to NumPy = 1×](../../results/batched_nlms_bar.png)

---

## Core implementation in RunMat (MATLAB‑syntax)

```matlab
rng(0);
p = 128; C = 2048; T = 200;
mu = single(0.5); eps0 = single(1e-3);

W = zeros(p, C, 'single');

for t = 1:T
  x = rand(p, C, 'single');
  d = sum(x .* x, 1);
  y = sum(x .* W, 1);
  e = d - y;
  nx = sum(x .^ 2, 1) + eps0;
  W = W + mu * x .* (e ./ nx);
end

mse = mean((d - sum(x .* W, 1)).^2, 'all');
fprintf('RESULT_ok MSE=%.6e\n', double(mse));
```

Full sources:
- RunMat / Octave: [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-nlms/runmat.m)
- Python (NumPy): [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-nlms/python_numpy.py)
- Python (PyTorch): [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-nlms/python_torch.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/batched-nlms/julia.jl)

---

## Why RunMat is fast (accelerate + fusion)

Workloads like this are a perfect fit for RunMat’s fusion and device residency: per‑column reductions plus long elementwise chains executed entirely on the GPU, with weights staying resident across steps.

See how RunMat works in practice in the [Introduction to RunMat on the GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md).

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).