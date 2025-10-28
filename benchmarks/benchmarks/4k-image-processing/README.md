# RunMat Benchmark: 4K Image Preprocessing, GPU‑accelerated

If you ship geospatial or vision workloads, you’ve likely written this stage countless times: standardize each 4K tile, apply a small radiometric correction, gamma‑correct, and run a quick QC metric. On CPUs this is fine—until the batch grows and your wall‑clock explodes. 

In this article, we benchmark the performance of RunMat against NumPy, PyTorch, and Julia.

The math is deliberately simple and realistic: compute a per‑image mean and standard deviation, normalize, apply a modest gain/bias and a gamma curve, then validate with a mean‑squared error.

---

## Results

![Relative speed (higher is better), normalized to NumPy = 1×](../../results/4k_image_processing_bar.png)

---

## Core implementation in RunMat (MATLAB‑syntax)

We'll use a simple pipeline: compute a per‑image mean and standard deviation, normalize, apply a modest gain/bias and a gamma curve, then validate with a mean‑squared error.

```matlab
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
- RunMat / Octave: [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/4k-image-processing/runmat.m)
- Python (NumPy): [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/4k-image-processing/python_numpy.py)
- Python (PyTorch): [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/4k-image-processing/python_torch.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/4k-image-processing/julia.jl)

Note: MATLAB’s license agreement restricts usage of their runtime for benchmarking, so we do not include MATLAB runs. If you have numbers, consider sharing them on GitHub Discussions.

---

## Why RunMat is fast (accelerate + fusion)

<insert blurb about accelerate and fusion here, direct to the individual doc for more detail>

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).