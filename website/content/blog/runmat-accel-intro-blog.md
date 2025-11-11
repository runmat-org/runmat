---
title: "Introducing RunMat Accelerate: The Fastest Runtime for Math"
description: "RunMat Accelerate is a modern, open-source runtime that executes MATLAB code quickly. A slim core, tiered execution, and a package system make it fast, predictable, and easy to extend."
date: "2025-11-05"
author: "Nabeel Allana"
readTime: "7 min read"
slug: "runmat-accel-intro-blog"
tags: ["MATLAB", "Rust", "JIT", "Octave", "scientific computing", "open source"]
keywords: "MATLAB runtime, GNU Octave comparison, Rust scientific computing, JIT compiler, numerical computing, open source"
excerpt: "RunMat Accelerate is a modern, open-source runtime that executes MATLAB code quickly. A slim core, tiered execution, and a package system make it fast, predictable, and easy to extend."
image: "/images/blog/runmat-hero.png"
imageAlt: "RunMat logo with performance charts"
ogType: "article"
ogTitle: "Introducing RunMat Accelerate: The Fastest Runtime for Math"
ogDescription: "RunMat is a modern, open-source runtime that executes MATLAB code quickly. A slim core, tiered execution, and a package system make it fast, predictable, and easy to extend."
twitterCard: "summary_large_image"
twitterTitle: "Introducing RunMat Accelerate: The Fastest Runtime for Math"
twitterDescription: "RunMat is a modern, open-source runtime that executes MATLAB code quickly. A slim core, tiered execution, and a package system make it fast, predictable, and easy to extend."
canonical: "https://runmat.org/blog/runmat-accel-intro-blog"
---

## The Fastest Runtime for Math

### Why a faster way to do math

Your mathematical code is elegant. It's also 50x slower than it should be. You know GPUs could fix this, but the path there is absurd: rewrite everything in CUDA, manage device memory explicitly, accept vendor lock-in, or pay thousands per seat for accelerators that still require code changes. The gap between "math as we think it" and "math as GPUs want it" has become so normalized that we've forgotten to question why  (A \- mean(A)) / std(A) should require anything more than writing exactly that. 

RunMat eliminates this gap entirely. Write your mathematical operations in clean, readable syntax. RunMat automatically fuses them into optimized GPU kernels that match or exceed hand-tuned CUDA performance. It runs on whatever GPU you have—NVIDIA, AMD, Apple Silicon, Intel—through native APIs (Metal/DirectX/Vulkan). No device management. No vendor lock-in. No rewrites. If you're tired of choosing between mathematical clarity and computational speed, this is for you.

#### What the best path looks like

Let's see RunMat in action with a common problem: processing high-resolution images in batches. Whether you're analyzing satellite imagery, medical scans, or camera sensor data, the pipeline is always the same—normalize the data, apply corrections, and enhance the signal. The math is straightforward. 

This pipeline mirrors common image preprocessing (remote sensing, medical, photography): per-image z-score normalization, radiometric gain/bias correction, gamma correction, and a simple quality check using MSE. We use 16 single-precision 4K tiles to avoid I/O effects and to stress the pattern GPUs handle well: long elementwise chains with light reductions. In RunMat, the MATLAB-style code remains as written, while Accelerate fuses elementwise steps and keeps arrays on the device, reducing kernel launches and transfers. 


``` Matlab
rng(0); B \= 16; H \= 2160; W \= 3840;  
gain \= single(1.0123); bias \= single(\-0.02);  
gamma \= single(1.8); eps0 \= single(1e-6);  
imgs \= rand(B, H, W, 'single');  
mu \= mean(imgs, \[2 3\]);  
sigma \= sqrt(mean((imgs \- mu).^2, \[2 3\]) \+ eps0);  
out \= ((imgs \- mu) ./ sigma) \* gain \+ bias;  
out \= out .^ gamma;  
mse \= mean((out \- imgs).^2, 'all');  
fprintf('Done. MSE=%.6e\\n', mse);
```

```mermaid
%%{init: {'theme':'dark','flowchart': {'curve':'linear'}}}%%
flowchart TD
  A["Your .m code"] --> B["Typed Graph (IR)"]
  B --> C["Planner (per step)"]
  B --> D["Profiler → Device Map"]
  D -. "size thresholds + transfer costs (refined at runtime)" .-> C

  subgraph CPU["CPU"]
    CJ["Ignition ➜ JIT (small arrays)<br/>Profiles hot loops; V8-style"]
    CB["CPU BLAS<br/>(Big CPU math or FP64)"]
  end

  subgraph GPU["GPU Fusion Engine (via WGPU)"]
    GF["Fuse back-to-back math into a bigger step<br/>(avoid re-scans)"]
    GR["Keep data resident on GPU until CPU needs it"]
  end

  classDef cpu fill:#e6e6e6,stroke:#999,color:#000
  classDef gpu fill:#e7f5e6,stroke:#7ab97a,color:#073b0b
  classDef mgr fill:#ffb000,stroke:#b37400,color:#2a1a00

  RM["Residency manager"]:::mgr
  C -. "partition / choose target / co-locate" .-> RM

  C -- "FP64 or big CPU-optimal" --> CB:::cpu
  C -- "small arrays" --> CJ:::cpu
  C -- "big or fuse-friendly" --> GF:::gpu

  RM <--> GF
  RM -. "avoid ping-pong" .- CJ
  RM -. "avoid ping-pong" .- CB

  GF --> GR:::gpu
  R(("Results"))
  GF --> R
  GR --> R
  CB --> R
  CJ --> R
	
```


### What RunMat does automatically:

1. Detects a GPU and selects the backend automatically (Metal / DirectX 12 / Vulkan). Falls back to CPU when none is available.  
2. Plans each operation to the best engine (CPU JIT, BLAS, or GPU) based on array sizes and op type.  
3. Fuses compatible steps and keeps arrays resident on the device to reduce kernel launches and transfers.  
4. One code path: no device flags, no vendor branches, no separate builds.

![][image2]

### The Complexity You're Avoiding

To understand what RunMat eliminates from your workflow, let's look at what GPU acceleration currently requires. PyTorch is the de facto standard for GPU computing in machine learning and scientific computing—it's mature, well-optimized, and widely used. If you want GPU acceleration today, PyTorch is often your best option. 

Here's our image preprocessing pipeline implemented in both PyTorch and RunMat. Notice what's required beyond the mathematical operations themselves:

![][image3]  
Vs 

![][image4]

The PyTorch version requires 24 lines to RunMat's 11, but line count isn't the real story. 

Look at what those extra lines are doing: 



* **Explicit device management:** PyTorch requires you to check for CUDA availability, explicitly move tensors to GPU. Every tensor operation must consider where the data lives. RunMat handles this automatically—if a GPU is available and it will be faster to use for your given task, it uses it. If not, it runs on CPU. Your code doesn't change.   
* **Framework-specific types:** PyTorch separates NumPy arrays and `torch.Tensor`, and you must track each tensor’s device. RunMat exposes one MATLAB-style array backed by a common `Tensor` abstraction (CPU or `GpuTensor`) with the same interface.  
* **Conditional execution paths:** PyTorch requires provider-specific branches (CUDA/ROCm/MPS) and device-aware tensor placement. RunMat has no provider branches: one script, with the runtime choosing Metal/DX12/Vulkan and keeping semantics consistent.

This isn't a criticism of PyTorch—it's an excellent framework that makes GPU programming accessible to millions of developers. But it still requires you to think about hardware details when you want to think about mathematics. RunMat removes that cognitive overhead entirely.

#### Cross-ecosystem comparison at a glance

A side-by-side of RunMat versus MATLAB+PCT, PyTorch, and Julia+CUDA.jl on six dimensions that matter for numerical computing: code surface, placement (CPU/GPU), fusion, residency, transfers, and learning curve. Scope: dense arrays and scripting—not full ML training stacks.

| Dimension                                                | RunMat Accelerate                                              | MATLAB \+ Parallel Computing Toolbox (gpuArray) | PyTorch (GPU)                                                | Julia \+ CUDA.jl                                                  |
|----------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------|
| **How you write code**                                   | Plain MATLAB-syntax (no gpuArray)                              | MATLAB with `gpuArray`, `gather`, device types  | Python tensors on `cuda`                                     | Julia `CuArray` types                                             |
| **CPU/GPU selection**                                    | **Automatic per-op** (live thresholds; fallback to CPU)        | Manual (you choose gpuArray)                    | Manual (you choose the device)                               | Manual (you choose the device)                                    |
| **Epilogue fusion** (e.g., matmul → divide → max → norm) | **Yes, cross-statement** (fused kernels)                       | No (separate kernels)                           | Limited by default; custom fusion via `torch.compile`/Triton | Broadcast fusion helps elementwise; GEMM epilogues often separate |
| **Data residency**                                       | **Automatic** (keeps tensors on device; gather only if needed) | Manual (you manage `gpuArray`/`gather`)         | Manual/partial (framework helps, but you manage boundaries)  | Manual/partial                                                    |
| **Host↔Device transfers**                                | **Minimized automatically**; device map amortizes copies       | Developer decides when to copy                  | Developer decides; easy to accidentally sync                 | Developer decides                                                 |
| **Learning curve**                                       | **Low** (keep existing scripts)                                | Medium (GPU types, residency patterns)          | Medium (tensor/device/dtype discipline; optional Triton)     | Medium (GPU array ecosystem)                                      |

#### What you can run today with RunMat

RunMat covers the core numerical stack so you can keep MATLAB-style code and use the GPU when it helps. Elementwise chains, reductions, filters, and matmul epilogues are fused on GPU; large linear-algebra ops call optimized BLAS/LAPACK on CPU. 

Full list and examples live in the library. \[ADD LINK\]

#### Why MATLAB syntax, not a new language

MATLAB is commonly taught in engineering programs, so many engineers already think in MATLAB-style array math. Keeping that surface lowers switching cost and preserves existing work.

* **Reuse existing code.** Teams can run current `.m` files with minimal edits rather than translating them into a new syntax.

* **Correct semantics for arrays.** MATLAB’s rules (column-major arrays, 1-based indexing, implicit expansion/broadcast) match how people write numerical work. We keep those semantics while changing the runtime.

* **Lower cognitive load.** Users focus on equations and shapes, not on learning new keywords or control structures.

* **Training and documentation.** Thousands of textbooks, lab notes, and snippets use MATLAB notation; keeping it preserves that knowledge base.

* **Separation of concerns.** The language stays stable; performance comes from the runtime (JIT, Accelerate, Fusion, WGPU). No API churn for users.

#### Try for yourself

* **Download RunMat:** \[Download\]

* **See benchmarks and examples:** \[Benchmarks\]

#### FAQ

**What is RunMat?**  
An open-source MATLAB-compatible runtime focused on fast, portable numerical computing. It keeps MATLAB-style syntax and accelerates code on GPU when available.

**Which operating systems and GPUs are supported?**  
macOS (Apple Silicon and supported AMD/NVIDIA cards), Windows, and Linux via the native GPU APIs above. If no compatible GPU is present, RunMat runs on CPU.

**Do I need MATLAB installed or a MATLAB license?**  
No. RunMat is a standalone runtime. It adheres to MATLAB semantics core language grammar, semantics and core builtins It is not associated or affiliated with MATHWORKS Matlab in any way, shape or form \[LINK Learn more grammar / semantics doc\] 

**How compatible is it with MATLAB?**  
Core array operations, elementwise math, reductions, common linear-algebra, FFT/signal/image, and statistics functions are covered. Check the function index for exact status. 

**How do I run my existing `.m` files?**  
Install RunMat and run your script with the CLI or the Jupyter kernel.
**Can I mix CPU and GPU in one script?**  
Yes. The planner chooses CPU JIT/BLAS or GPU per step. Fusion keeps GPU regions device-resident when beneficial.

**Does it work offline? Any telemetry?**  
 It runs locally and does not require internet access for execution. See the repository for any optional diagnostics and how to disable them. \[Privacy/Telemetry\]

**How do I report an issue or contribute?**  
 Open an issue or PR in the repository. Include OS, GPU/CPU info, a minimal script, and steps to reproduce. \[GitHub\



