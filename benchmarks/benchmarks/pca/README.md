# RunMat Benchmark: PCA via Covariance + Block Power Method

Principal Component Analysis (PCA) is ubiquitous in analytics pipelines. When your data matrix is tall and wide, computing a covariance and extracting top‑k eigenvectors becomes throughput‑bound. This benchmark compares RunMat against NumPy, PyTorch, and Julia on a realistic PCA workload using a covariance matrix and a block power iteration.

We center columns and form `G = (Aᵀ A)/(n−1)`, then run block power for `k` vectors with per‑column normalization. We report the top‑1 explained variance fraction and the sum of the top‑k Rayleigh quotients.

---

## Results

![Relative speed (higher is better), normalized to NumPy = 1×](../../results/pca_bar.png)

---

## Core implementation in RunMat (MATLAB-syntax)

```matlab
rng(0);
n = 200_000; d = 1024; k = 8; iters = 15;
A = rand(n, d, 'single');
mu = mean(A, 1, 'omitnan');
A = A - mu;
G = (A' * A) / single(n - 1);

Q = rand(d, k, 'single');
for j = 1:k
  nj = norm(Q(:, j));
  if nj > 0
    Q(:, j) = Q(:, j) ./ nj;
  end
end

for t = 1:iters
  Q = G * Q;
  for j = 1:k
    nj = norm(Q(:, j));
    if nj > 0
      Q(:, j) = Q(:, j) ./ nj;
    end
  end
end

Lambda = diag(Q' * G * Q);
explained = double(Lambda) / sum(double(diag(G)));
fprintf('RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n', explained(1), sum(double(Lambda)));
```

Full sources:
- RunMat / Octave: [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/pca/runmat.m)
- Python (NumPy): [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/pca/python_numpy.py)
- Python (PyTorch): [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/pca/python_torch.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/pca/julia.jl)

---

## Why RunMat is fast (accelerate + fusion)

RunMat fuses elementwise stages and keeps tensors resident on device; covariance formation and repeated matvecs benefit from coalesced access and cached pipelines. For an overview of fusion and device residency, see the [Introduction to RunMat on the GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md) document.

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).
