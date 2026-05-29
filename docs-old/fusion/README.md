# RunMat Fusion Guide

RunMat’s acceleration layer recognises multiple flavours of fusible graphs and hands them to the GPU provider as single kernels. These notes are meant to help you understand what will fuse, how the GPU work is shaped, and where to look for deeper detail.

## Documents in This Folder

| Topic | Description |
| --- | --- |
| [Elementwise Chains](elementwise.md) | How RunMat collapses arithmetic / transcendental expressions into one shader with broadcasting. |
| [Reductions](reduction.md) | Behaviour of fused `sum`, `mean`, and similar column/row reductions, including omit-NaN and scaling rules. |
| [Matmul Epilogues](matmul_epilogue.md) | When matmul outputs stay on device for scale/bias/clamp/pow epilogues, including diagonal extraction. |
| [Centered Gram / Covariance](centered_gram.md) | Mean subtraction + covariance/Gram construction for any tall matrix. |
| [Power-Step Normalisation](power_step_normalize.md) | Fusion of matmul + vector normalisation stages in iterative solvers. |
| [Explained Variance](explained_variance.md) | Keeping `diag(Q' * G * Q)`-style diagnostics resident on GPU. |
| [Image Normalisation](image_normalize.md) | Batch × H × W whitening / gain / bias fusion for image-like tensors. |

## How to Use These Docs

1. **Looking for coverage:** Start with the link that matches your math. Each page lists the exact instruction patterns the fusion planner looks for and the operations that stay on device.
2. **Investigating surprises:** If a workload is not fusing, cross-check the prerequisites section for that category (e.g. single-consumer chains for elementwise groups or constant epsilon for power steps).
3. **Telemetry correlation:** Provider telemetry reports `fusion_kind` labels. Match those labels to the filenames above to understand what the GPU executed.

## Why These Fusion Groups Exist

- **Elementwise & reductions:** These appear in every numeric workload and are the easiest way to collapse dozens of scalar ops into one dispatch. Keeping them fused prevents bandwidth blowups from repeated reads/writes of the same tensor.
- **Matmul epilogues:** Dense linear algebra is already expensive; fusing scale/bias/activation epilogues avoids launching a second kernel that touches the full matrix again and is the key to RunMat’s “matmul + activation” parity goals.
- **Covariance / Gram / power-step / explained-variance chains:** Iterative factorizations spend most of their time in repeated “multiply, renormalize, measure” loops. By treating each stage as a fusion kind we keep eigensolvers, Krylov methods, and other orthogonal-basis builders resident on the GPU.
- **Image normalisation:** Many imaging and sensor pipelines start with per-frame whitening + gain/bias adjustments. Folding statistics and affine transforms into one kernel eliminates 3–4 launches per frame.

These groups were prioritised because they (a) show up across multiple domains (linear algebra, signal processing, imaging), (b) keep otherwise chatty host/device traffic off the PCIe bus, and (c) re-use existing provider kernels so we could ship them quickly.

## Next Fusion & Kernel Targets

We still have obvious wins ahead. The current shortlist mirrors the open technical gaps we track internally:

1. **Broaden GPU kernel coverage.** Add missing `AccelProvider` hooks so FFT/conv, RNG, sorting, scans, specialised signal-processing kernels, and the remaining MATLAB builtins can stay on device.
2. **Smarter chain planning.** Feed auto-offload cost models with telemetry so fusion can safely grab marginal chains (e.g. small elementwise spans or reductions with dynamic shapes) without regressing latency.
3. **Full end-to-end GPU paths.** Close the remaining host fallbacks in the Accelerate roadmap so “GPU mode” really means zero CPU detours for supported types.
4. **JIT parity for CPU fallbacks.** Finish the CraneLift tier for the constructs that still spill to the interpreter; even when a kernel is GPU-ready its callers need the same coverage to avoid sync points.
5. **Additional domain groups.** Candidates include FFT→pointwise→IFFT pipelines, convolution + activation blocks, sliding-window statistics, and RNG-heavy Monte Carlo payoffs. Each would convert entire measurement loops into single residency-preserving kernels.

If you have a new fusion flavour you'd like to see implemented, please open an issue or submit a pull request.