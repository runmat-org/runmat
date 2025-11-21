# RunMat - The Fastest Runtime for Math

🚀 Open Source • MIT Licensed • Free Forever

RunMat fuses back-to-back ops into fewer GPU steps and keeps arrays on device. MATLAB syntax. No kernel code, no rewrites.

- [Download](/download)
- [Get Started](/docs/getting-started)
- [Benchmarks](#benchmarks)

## GPU-Accelerated Math

Designed to compete with the top open source stack, Python + PyTorch (CUDA), in various GPU-optimized workloads.

## Write in MATLAB. Run on CPU or GPU.

Same MATLAB-syntax you already know. Bring your existing MATLAB code and run as is. RunMat will translate your math to GPU code and run it automatically. No kernel code, no rewrites.

- [Language coverage](/docs/language-coverage)
- [Builtin Function Reference](/docs/matlab-function-reference)

## GPU Fusion Architecture, Why it's Fast

RunMat fuses back-to-back math into fewer GPU steps and keeps arrays on the device between steps ("residency"). That means less memory traffic and fewer launches. so your scripts finish sooner.

## Why Use RunMat?

### ✅ Full MATLAB Language Semantics

Run the whole language, not a subset: proper indexing (`end`/colon/masks), multiple returns, classdef OOP, events/handles—plus built-ins across arrays, linalg, FFT/signal, stats, strings, and I/O. See [coverage](/docs/language-coverage).

### ⚡ Faster by Design (Fusion + Residency)

Fuse back-to-back ops into fewer GPU launches. Keep arrays on device between steps. Less memory traffic, fewer kernel launches, faster execution.

### 📦 Slim Core + Packages + IDE LSP

Modular architecture with a lean core runtime. Extensible through packages. Full IDE support with Language Server Protocol (LSP) for autocomplete, syntax highlighting, and error checking.

### 🧱 Portable + Lightweight

Single binary deployment. No complex dependencies. Works across Windows, macOS, and Linux. Small footprint, big performance.

## Real Workloads, Reproducible Results

From per-pixel 4K image pipelines to PCA, NLMS, and Monte Carlo, see how RunMat stacks up against Python + PyTorch, NumPy, and Julia. Run them locally and compare on your machine.

Benchmarked on an Apple M2 Max, 32GB

Reproduce with [benchmarks in the repo](#benchmarks)

- [4K image pipeline](/benchmarks/4k-image-pipeline)
- [PCA](/benchmarks/pca)
- [NLMS](/benchmarks/nlms)
- [Monte Carlo](/benchmarks/monte-carlo)
- [Full benchmark index](/benchmarks)

## Free and Open Source, Forever

Copy and paste the command below to get started with RunMat.

```bash
# Install command (see OSInstallCommand component for actual command)
```

- [More Install Options](/download)
- [Get Started](/docs/getting-started)


