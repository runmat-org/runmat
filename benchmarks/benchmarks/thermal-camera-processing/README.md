# RunMat Benchmark: Thermal Camera Processing (Dark/Flat/Radiometric + Temp)

Thermal imaging pipelines typically apply dark/flat corrections and a radiometric calibration before converting to temperature. At realistic resolutions and batch sizes, the workload becomes bandwidth‑bound and benefits from fusing elementwise stages and keeping frames resident on device. This benchmark compares RunMat against NumPy, PyTorch, and Julia on a synthesized thermal preprocessing stage.

We correct raw frames with `lin = (raw − dark) .* ffc`, apply radiometric gain/offset, clamp negatives, and convert to temperature via a simple `log1p` mapping.

---

## Results

![Relative speed (higher is better), normalized to NumPy = 1×](../../results/thermal_camera_processing_bar.png)

---

## Core implementation in RunMat (MATLAB‑syntax)

```matlab
rng(0);
B = 16; H = 1024; W = 1024;

raw = rand(B, H, W, 'single');

dark   = 0.02 + 0.01 * rand(H, W, 'single');
ffc    = 0.98 + 0.04 * rand(H, W, 'single');
gain   = 1.50 + 0.50 * rand(H, W, 'single');
offset = -0.05 + 0.10 * rand(H, W, 'single');

lin = (raw - dark) .* ffc;
radiance = lin .* gain + offset;
radiance = max(radiance, 0);

tempK = 273.15 + 80 * log1p(radiance);

mean_temp = mean(tempK, 'all');
fprintf('RESULT_ok MEAN_TEMP=%.6f\n', double(mean_temp));
```

Full sources:
- RunMat / Octave: [`runmat.m`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/thermal-camera-processing/runmat.m)
- Python (NumPy): [`python_numpy.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/thermal-camera-processing/python_numpy.py)
- Python (PyTorch): [`python_torch.py`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/thermal-camera-processing/python_torch.py)
- Julia: [`julia.jl`](https://github.com/runmat-org/runmat/blob/main/benchmarks/benchmarks/thermal-camera-processing/julia.jl)

---

## Why RunMat is fast (accelerate + fusion)

Dark/flat/radiometric/temperature stages form a long elementwise chain with identical shapes, making them ideal for fusion and device residency in RunMat. For a conceptual overview, see the project [Introduction to RunMat on the GPU](https://github.com/runmat-org/runmat/blob/main/docs/INTRODUCTION_TO_RUNMAT_GPU.md).

---

## Reproduce the benchmarks

See the benchmarks directory in the RunMat repo on GitHub for the full source code and instructions to reproduce the benchmarks: [runmat-org/runmat/benchmarks](https://github.com/runmat-org/runmat/tree/main/benchmarks).
