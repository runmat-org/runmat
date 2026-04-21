---
title: "How to Use GPU in MATLAB Without CUDA: Apple Silicon, NVIDIA, AMD & Intel"
description: "A practical guide to GPU acceleration in MATLAB on any hardware. How it works on NVIDIA with the Parallel Computing Toolbox, why it doesn't work on Apple Silicon, AMD, or Intel, and how to get GPU acceleration without CUDA."
date: "2026-01-28"
dateModified: "2026-04-14"
authors:
  - name: "Fin Watterson"
    url: "https://www.linkedin.com/in/finbarrwatterson/"
readTime: "10 min read"
slug: "how-to-use-gpu-in-matlab"
tags: ["matlab", "gpu", "nvidia", "gpuArray", "scientific-computing", "parallel-computing-toolbox", "apple-silicon", "metal", "cross-platform"]
collections: ["guides"]
keywords: "matlab gpu without cuda, matlab gpu without parallel computing toolbox, free matlab gpu, matlab gpu any gpu, matlab gpu apple silicon, matlab gpu mac, matlab gpu m1, matlab gpu m2, matlab gpu m3, matlab gpu m4, matlab gpu amd, matlab gpu intel, matlab gpu alternative, matlab gpu acceleration, gpuArray mac, matlab gpu non-nvidia, how to use gpu in matlab, matlab metal gpu, matlab directx, matlab vulkan, gpuArray tutorial, gpuarray indexing, which matlab functions support gpuarray, can matlab use gpu, speed up matlab with gpu"
excerpt: "MATLAB's Parallel Computing Toolbox requires NVIDIA CUDA, which locks out Apple Silicon, AMD, and Intel entirely. This guide covers what works on each vendor, the gpuArray setup for NVIDIA users who have the toolbox, and how to get GPU acceleration without CUDA on any hardware."
ogType: "article"
ogTitle: "How to Use GPU in MATLAB Without CUDA: Apple Silicon, NVIDIA, AMD & Intel"
ogDescription: "A practical guide to GPU acceleration in MATLAB on any hardware. How it works on NVIDIA with the Parallel Computing Toolbox, why it doesn't work on Apple Silicon, AMD, or Intel, and how to get GPU acceleration without CUDA."
twitterCard: "summary_large_image"
twitterTitle: "How to Use GPU in MATLAB Without CUDA: Apple Silicon, NVIDIA, AMD & Intel"
twitterDescription: "What actually works for GPU acceleration in MATLAB on NVIDIA, Apple Silicon, AMD, and Intel, plus how to skip CUDA entirely."
image: "https://web.runmatstatic.com/MATLAB-NVIDIA.png"
imageAlt: "How to use GPU in MATLAB without CUDA: Apple Silicon, NVIDIA, AMD, and Intel"
canonical: "https://runmat.com/blog/how-to-use-gpu-in-matlab"
jsonLd:
  "@context": "https://schema.org"
  "@graph":
    - "@type": "BreadcrumbList"
      itemListElement:
        - "@type": "ListItem"
          position: 1
          name: "RunMat"
          item: "https://runmat.com"
        - "@type": "ListItem"
          position: 2
          name: "Blog"
          item: "https://runmat.com/blog"
        - "@type": "ListItem"
          position: 3
          name: "How to Use GPU in MATLAB Without CUDA"
          item: "https://runmat.com/blog/how-to-use-gpu-in-matlab"

    - "@type": "TechArticle"
      "@id": "https://runmat.com/blog/how-to-use-gpu-in-matlab#article"
      headline: "How to Use GPU in MATLAB Without CUDA: Apple Silicon, NVIDIA, AMD & Intel"
      alternativeHeadline: "What works on each GPU vendor, the gpuArray setup for NVIDIA, and how to get GPU acceleration without CUDA"
      description: "A practical guide to GPU acceleration in MATLAB on any hardware. Covers the Parallel Computing Toolbox workflow for NVIDIA users, why MATLAB doesn't work on Apple Silicon, AMD, or Intel GPUs, and how to get GPU acceleration without CUDA."
      datePublished: "2026-01-28T00:00:00Z"
      dateModified: "2026-04-14T00:00:00Z"
      image: "https://web.runmatstatic.com/MATLAB-NVIDIA.png"
      author:
        "@type": "Person"
        name: "Fin Watterson"
        url: "https://www.linkedin.com/in/finbarrwatterson/"
      publisher:
        "@type": "Organization"
        name: "RunMat by Dystr"
        logo:
          "@type": "ImageObject"
          url: "/runmat-logo.svg"
      about:
        - "@type": "SoftwareApplication"
          name: "MATLAB"
          applicationCategory: "ScientificApplication"
          operatingSystem: "Windows, Linux, macOS"
        - "@type": "SoftwareApplication"
          "@id": "https://runmat.com/#software"
          name: "RunMat"
          applicationCategory: "ScientificApplication"
          operatingSystem: "Browser, Windows, Linux, macOS"
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
      speakable:
        "@type": "SpeakableSpecification"
        cssSelector: ["h1"]

    - "@type": "FAQPage"
      "@id": "https://runmat.com/blog/how-to-use-gpu-in-matlab#faq"
      mainEntity:
        - "@type": "Question"
          name: "Can I use MATLAB GPU on a Mac?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. MATLAB's gpuArray requires CUDA, which only runs on NVIDIA GPUs. Apple Silicon Macs (M1, M2, M3, M4) have no NVIDIA GPU and no CUDA support. RunMat uses Apple's Metal API to provide GPU acceleration on every Mac with Apple Silicon — same MATLAB-style code, no toolbox required."
        - "@type": "Question"
          name: "Does MATLAB support M1/M2/M3/M4 GPU acceleration?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. MATLAB's Parallel Computing Toolbox is built on NVIDIA CUDA and does not support Apple's Metal API. M-series Macs have powerful GPUs, but MATLAB cannot use them. RunMat targets Metal directly, so M1/M2/M3/M4 GPUs are fully supported for array math and fusion."
        - "@type": "Question"
          name: "Can MATLAB use an AMD GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. MATLAB's GPU support requires NVIDIA CUDA. AMD GPUs (Radeon, Instinct) use ROCm or Vulkan, neither of which MATLAB's Parallel Computing Toolbox supports. RunMat uses Vulkan on Linux and DirectX 12 on Windows, so AMD GPUs work without any driver-level workarounds."
        - "@type": "Question"
          name: "Can MATLAB use an Intel GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. Intel GPUs (integrated or Arc) are not supported by MATLAB's Parallel Computing Toolbox, which requires NVIDIA CUDA. RunMat supports Intel GPUs through Vulkan on Linux and DirectX 12 on Windows."
        - "@type": "Question"
          name: "Do I need CUDA even if I have an NVIDIA GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "For MATLAB, yes — the Parallel Computing Toolbox requires CUDA. With RunMat, no. RunMat accesses NVIDIA GPUs through Vulkan instead of CUDA, so you get GPU acceleration without installing CUDA drivers or buying the toolbox."
        - "@type": "Question"
          name: "Is there a free way to get GPU acceleration with MATLAB code?"
          acceptedAnswer:
            "@type": "Answer"
            text: "MATLAB requires the Parallel Computing Toolbox (a paid add-on) for GPU acceleration. RunMat is free and includes GPU acceleration by default — no extra license, no toolbox. You write the same array math and the runtime handles CPU-vs-GPU routing automatically."
        - "@type": "Question"
          name: "Can I use GPU without the Parallel Computing Toolbox?"
          acceptedAnswer:
            "@type": "Answer"
            text: "In MATLAB, no. The toolbox is required for gpuArray and GPU-enabled functions, and it's a paid add-on. RunMat includes GPU acceleration by default: you write the same array math and the runtime decides CPU vs GPU and fuses operations without an extra license."
        - "@type": "Question"
          name: "What GPU do I need for MATLAB?"
          acceptedAnswer:
            "@type": "Answer"
            text: "MATLAB's Parallel Computing Toolbox requires an NVIDIA GPU with CUDA compute capability 3.5 or higher. No AMD, Intel, or Apple Silicon GPUs are supported. If you want GPU acceleration on non-NVIDIA hardware or without the toolbox, RunMat works on any modern GPU via Metal (Mac), DirectX 12 (Windows), and Vulkan (Linux)."
        - "@type": "Question"
          name: "Do I need to install CUDA or vendor drivers to use RunMat with an AMD or Intel GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. RunMat reaches AMD and Intel GPUs through Vulkan on Linux and DirectX 12 on Windows, which are already present on most systems. There is no CUDA, no ROCm, and no oneAPI toolchain to install."
        - "@type": "Question"
          name: "How do I try GPU acceleration on my Mac without installing anything?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Open runmat.com/sandbox in a WebGPU-capable browser (Safari 18+, Chrome 113+, Edge 113+, Firefox 139+) and paste your .m code. The sandbox dispatches to Metal on macOS automatically. For the native CLI on an M-series Mac, install RunMat and run 'runmat run your_script.m'."
        - "@type": "Question"
          name: "Why is my GPU slower than my CPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Most often: the arrays are too small, you're doing many tiny steps, or you're transferring/synchronizing frequently (e.g., gather or printing in a loop). Fix it by batching into larger arrays and calling gather only once at the end."
        - "@type": "Question"
          name: "Should I use single or double precision on GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Use what your numerics require. Single (FP32) is faster, uses half the memory, and is the right default for most workloads. Double precision is available in MATLAB on NVIDIA and in RunMat on NVIDIA, AMD, and Intel; performance then depends on the GPU's FP64 capability. Apple's Metal does not support FP64 at all, so on Mac FP32 is the only GPU option."
        - "@type": "Question"
          name: "What's the simplest rule for GPU performance?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Make the work big and contiguous, and avoid transfers. Everything else is a refinement."

    - "@type": "HowTo"
      "@id": "https://runmat.com/blog/how-to-use-gpu-in-matlab#howto-without-cuda"
      name: "How to get MATLAB GPU acceleration without CUDA"
      description: "Step-by-step guide to GPU-accelerating MATLAB-syntax code on any GPU (Apple Silicon, AMD, Intel, or NVIDIA) without CUDA and without the Parallel Computing Toolbox."
      step:
        - "@type": "HowToStep"
          name: "Check your GPU vendor"
          text: "Open System Information (Mac), Device Manager (Windows), or run lspci (Linux) to identify your GPU. RunMat works on any vendor; MATLAB's Parallel Computing Toolbox requires NVIDIA + CUDA."
        - "@type": "HowToStep"
          name: "Understand the CUDA limitation"
          text: "MATLAB's gpuArray requires CUDA, which only runs on NVIDIA GPUs. Even NVIDIA users need the paid Parallel Computing Toolbox. RunMat skips CUDA entirely."
        - "@type": "HowToStep"
          name: "Install RunMat"
          text: "Download RunMat from runmat.com or use the browser sandbox at runmat.com/sandbox. RunMat uses Metal (macOS), DirectX 12 (Windows), and Vulkan (Linux) to access any GPU."
        - "@type": "HowToStep"
          name: "Run your existing script"
          text: "RunMat accepts .m files with standard MATLAB syntax. Run your script as-is — the runtime auto-detects the GPU and routes eligible array operations to it without gpuArray or gather."
        - "@type": "HowToStep"
          name: "Verify the GPU path"
          text: "Run a large array computation (millions of elements) and compare wall-clock time to a known CPU baseline. If the GPU path is active, large vectorized operations will complete significantly faster."

    - "@type": "HowTo"
      "@id": "https://runmat.com/blog/how-to-use-gpu-in-matlab#howto-gpuarray"
      name: "How to accelerate MATLAB code with gpuArray on NVIDIA"
      description: "The short-form gpuArray workflow for MATLAB users who have an NVIDIA GPU and the Parallel Computing Toolbox."
      step:
        - "@type": "HowToStep"
          name: "Check prerequisites"
          text: "Verify you have an NVIDIA GPU with CUDA compute capability 3.5+, a compatible CUDA driver, and a Parallel Computing Toolbox license. Confirm MATLAB sees the GPU with gpuDeviceCount and gpuDevice."
        - "@type": "HowToStep"
          name: "Upload once with gpuArray"
          text: "Wrap your input with gpuArray() - e.g. x = gpuArray.rand(N, 1, 'single') - or convert an existing array with gpuArray(x)."
        - "@type": "HowToStep"
          name: "Compute on GPU"
          text: "Run vectorized operations (sin, elementwise multiply, mean, etc.); they dispatch to the GPU automatically while x is a gpuArray."
        - "@type": "HowToStep"
          name: "Gather once at the end"
          text: "Call gather() once when you need the result on the CPU. For the full GPU-enabled function list and driver compatibility matrix, see MathWorks' Parallel Computing Toolbox documentation."
---

MATLAB has exactly one path to GPU acceleration: the Parallel Computing Toolbox, a paid add-on. It comes with two requirements that often get conflated. The GPU has to be NVIDIA (the hardware), and the system has to have CUDA installed (the software), NVIDIA's proprietary compute runtime that the toolbox calls into. They aren't the same thing, but the toolbox needs both. If your machine has an Apple Silicon, AMD, or Intel GPU, neither requirement is met and `gpuArray` returns "No supported GPU device was found." This guide covers what actually works on each vendor, the `gpuArray` setup for NVIDIA users, and how to get GPU acceleration without CUDA.

## TL;DR

- **Apple Silicon Mac (M1/M2/M3/M4):** MATLAB's `gpuArray` returns "No supported GPU device was found" because CUDA doesn't run on Apple GPUs. [RunMat](https://runmat.com) uses Metal and runs the same MATLAB-syntax code on any M-series Mac.
- **AMD (Radeon, Instinct) or Intel (Arc, integrated):** Same story. The Parallel Computing Toolbox only targets NVIDIA CUDA. RunMat uses Vulkan (Linux) and DirectX 12 (Windows) so these GPUs work without CUDA.
- **NVIDIA without the Parallel Computing Toolbox:** You don't need the paid add-on to use your NVIDIA GPU. RunMat reaches NVIDIA through Vulkan, no CUDA toolkit required.
- **NVIDIA with the toolbox:** The standard `gpuArray` pattern works and is well-documented by MathWorks. The short summary is further down.
- FP32 is the safe first choice for any GPU path. Apple's Metal doesn't support FP64 at all.
---

## **What GPU do you have?**

MATLAB's GPU support and the path you'll take depend almost entirely on what's inside your machine. Start here:

| GPU vendor | MATLAB (Parallel Computing Toolbox) | Without CUDA (RunMat) |
|------------|-------------------------------------|------------------------|
| **NVIDIA** (GeForce, RTX, Tesla) | Yes, with toolbox license ($) | Yes, via Vulkan |
| **Apple Silicon** (M1/M2/M3/M4) | No | Yes, via Metal |
| **AMD** (Radeon, Instinct) | No | Yes, via Vulkan / DirectX 12 |
| **Intel** (Arc, integrated) | No | Yes, via Vulkan / DirectX 12 |

The Parallel Computing Toolbox needs both NVIDIA hardware and CUDA. Without an NVIDIA card *and* a valid CUDA driver/toolkit, `gpuArray` and every GPU-enabled function are unavailable, regardless of how capable your GPU is.

Jump to the section that matches your hardware:

- [Apple Silicon (M1, M2, M3, M4)](#matlab-gpu-on-apple-silicon-m1-m2-m3-m4)
- [AMD and Intel](#matlab-gpu-on-amd-and-intel)
- [NVIDIA + the Parallel Computing Toolbox](#if-you-have-nvidia-the-parallel-computing-toolbox)
- [GPU acceleration without CUDA: how RunMat works](#gpu-acceleration-without-cuda-how-runmat-works)

---

## **MATLAB GPU on Apple Silicon (M1, M2, M3, M4)**

Every Mac sold since late 2020 has an Apple Silicon chip with a capable GPU that MATLAB can't touch. Apple's M-series GPUs deliver serious single-precision throughput:

| Chip | FP32 throughput (approx) |
|------|--------------------------|
| M1 | ~2.6 TFLOPS |
| M2 | ~3.6 TFLOPS |
| M3 | ~4.1 TFLOPS |
| M3 Max (40-core GPU) | ~14 TFLOPS |
| M4 | ~4.6 TFLOPS |
| M4 Max (40-core GPU) | ~18 TFLOPS |

Numbers are public estimates; exact figures vary with GPU core count and sustained clocks. Even the base chips sit in the same league as a mid-range discrete GPU, idle in MATLAB's eyes.

### Why MATLAB GPU doesn't work on Mac

MATLAB's GPU path is built on CUDA. NVIDIA stopped shipping Mac-compatible GPUs and drivers in 2016, and Apple moved the entire Mac line to its own ARM-based chips in 2020. There is no CUDA runtime for Apple Silicon and no indication either company plans to change that.

If you open MATLAB on an M-series Mac and try the GPU path, you'll get:

```matlab
>> gpuDeviceCount
ans = 0

>> gpuArray(rand(1000))
Error using gpuArray
No supported GPU device was found on this computer.
```

MathWorks confirms this in their [GPU support by release](https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html) documentation: only NVIDIA GPUs with CUDA compute capability 3.5+ are supported.

### What Mac users actually do

The workarounds are unappealing. MATLAB Online doesn't include GPU support in the standard tier, so you'd need a cloud instance with an NVIDIA GPU attached, which means paying for both the MATLAB license and the cloud compute. Remote desktop into a Windows or Linux box with an NVIDIA card works but adds latency and hardware cost. Most Mac-based MATLAB users just run on CPU and accept slower runtimes on large array workloads.

### GPU acceleration on Mac with RunMat

[RunMat](https://runmat.com) uses Apple's [Metal](https://developer.apple.com/metal/) API directly, so M1/M2/M3/M4 GPUs are addressable with the same MATLAB-syntax code you'd write for CPU. The computation that fails with `gpuArray` on Mac runs with GPU acceleration in RunMat without any changes:

```matlab:runnable
rng(0);
x = rand(10_000_000, 1, 'single');
y = sin(x) .* x + 0.5;
m = mean(y, 'all');
fprintf("m = %.6f\n", double(m));
```

What's missing is `gpuArray`, `gather`, CUDA, and any vendor check. RunMat's runtime inspects the computation shape and decides per-operation whether to run on CPU or GPU. Large, contiguous elementwise chains get *fused* into a single GPU kernel. Small or irregular work stays on CPU.

### Where M-series GPUs shine

The workloads MATLAB users most often wish they could GPU-accelerate on Mac are exactly the ones Metal handles well: image and signal processing (big FFTs, 2D convolutions), large elementwise math pipelines (physical simulations, Monte Carlo), and dense linear algebra on single-precision data. The unified memory architecture on Apple Silicon also means the CPU-GPU "transfer" is effectively a memory fence, not a DMA copy, which softens one of the traditional GPU penalties covered further down.

### One thing to know about precision

Metal doesn't implement FP64. If your code explicitly depends on double precision, parts of the pipeline will run on CPU in FP64 and only the FP32-safe portions will reach the GPU. For most scientific and engineering workloads FP32 is fine; for legacy numerics that require FP64 (certain accumulators, ill-conditioned solves), check whether reformulating to mixed precision is acceptable before expecting a GPU speedup.

### Trying it

You can run the code block above in RunMat without installing anything at [runmat.com/sandbox](https://runmat.com/sandbox) (WebGPU). For native Metal on your own Mac, [install RunMat](/download) and run your `.m` file with `runmat run your_script.m`.

---

## **MATLAB GPU on AMD and Intel**

AMD and Intel GPUs are in the same position as Apple Silicon. MATLAB's Parallel Computing Toolbox is NVIDIA-only, and neither AMD's ROCm stack nor Intel's oneAPI stack plugs in as a substitute. "ROCm for MATLAB" and "oneAPI for MATLAB" are frequent searches with no official answer: the toolbox simply doesn't have a backend for either.

On a system with only an AMD or Intel GPU, `gpuDeviceCount` returns 0 and `gpuArray` errors with "No supported GPU device was found," identical to the Apple Silicon case.

### What each vendor brings

- **AMD Radeon RX** (7900 XTX, 9070, etc.): strong FP32 throughput, great value per TFLOP, common on Linux workstations. Unused by MATLAB.
- **AMD Instinct** (MI250, MI300): datacenter-grade FP32 and FP64. MATLAB sees none of it.
- **Intel Arc** (A770, A750, B580): newer discrete GPUs aimed at compute alongside gaming. Same story.
- **Intel integrated** (Iris Xe, UHD): not performance monsters, but capable of meaningful speedups on elementwise pipelines where array size is big enough. MATLAB ignores them completely.

### GPU acceleration without CUDA

RunMat reaches these GPUs through the graphics APIs they already support: Vulkan on Linux (AMD and Intel) and DirectX 12 on Windows (AMD and Intel). No CUDA, no ROCm install, no oneAPI toolchain. The same MATLAB-syntax code runs unchanged:

```matlab:runnable
rng(0);
x = rand(10_000_000, 1, 'single');
y = sin(x) .* x + 0.5;
m = mean(y, 'all');
fprintf("m = %.6f\n", double(m));
```

If you have multiple GPUs (e.g. an integrated Intel plus a discrete AMD on Linux), set `RUNMAT_ACCEL_WGPU_POWER=high` to prefer the discrete card or `low` to favour integrated. The default is `auto`, which lets wgpu pick based on the system's own hint.

The [install link](/download) covers Windows and Linux builds. If you just want to confirm things work before installing, the [browser sandbox](https://runmat.com/sandbox) uses WebGPU, which dispatches to your AMD or Intel GPU through the same underlying drivers.

---

## **If you have NVIDIA + the Parallel Computing Toolbox**

The official `gpuArray` path works and is well-documented. You need an NVIDIA GPU with CUDA compute capability 3.5 or higher (most cards from 2012 onward, full list on [NVIDIA's CUDA GPUs page](https://developer.nvidia.com/cuda-gpus)), a compatible CUDA driver, and a Parallel Computing Toolbox license. Confirm MATLAB sees the GPU with `gpuDeviceCount` and `gpuDevice`. If either misbehaves, the usual culprits are a driver/MATLAB version mismatch, a missing toolbox license, or integrated graphics showing up instead of the discrete card.

The short form of the `gpuArray` workflow is upload-once, compute, gather-once:

```matlab:runnable
rng(0);
x = gpuArray.rand(10_000_000, 1, 'single');
y = sin(x) .* x + 0.5;
m = mean(y, 'all');
fprintf("m = %.6f\n", gather(m));
```

For the full GPU-enabled function list, driver version matrix, and per-function examples, MathWorks maintains the authoritative references: [Run Built-In Functions on a GPU](https://www.mathworks.com/help/parallel-computing/run-built-in-functions-on-a-gpu.html) and [GPU Support by Release](https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html).

---

## **Two GPU traps that apply to any vendor**

Whether you're using `gpuArray` on NVIDIA, Metal on Mac, or Vulkan/DX12 on AMD or Intel, the same two patterns cause most "my GPU is slower than my CPU" surprises.

The first is running on arrays that are too small. GPU kernel launches have fixed overhead. A million elements is usually enough to see a speedup on elementwise math; 10K usually isn't. Code that runs a loop over thousands of small arrays almost always loses to the CPU. The fix is to batch: replace the loop with one larger array and run the computation once.

```matlab:runnable
% Overhead-heavy shape
acc = single(0);
for i = 1:2000
    x = rand(4096, 1, 'single');
    acc = acc + sum(sin(x) .* x + 0.5, 'all');
end
```

```matlab:runnable
% Better shape for any GPU
X = rand(4096, 2000, 'single');
acc = sum(sin(X) .* X + 0.5, 'all');
```

The second is hidden transfers and syncs. Every `gather`, every `fprintf` or `disp` on a GPU value, every plot inside a hot loop forces a round-trip to the CPU. Keep inspection outside your timed region, and `gather` exactly once when you actually need the result on the host.

Precision choice, kernel fusion, and memory layout are refinements on those two ideas. For the MATLAB-specific version see MathWorks' [Measure and Improve GPU Performance](https://www.mathworks.com/help/parallel-computing/measure-and-improve-gpu-performance.html).

---

## **GPU acceleration without CUDA: how RunMat works**

MATLAB's GPU path needs three things you may not have: CUDA, an NVIDIA GPU, and the Parallel Computing Toolbox. RunMat drops all three. It runs MATLAB-syntax `.m` files on [Apple Silicon](#matlab-gpu-on-apple-silicon-m1-m2-m3-m4), [AMD and Intel](#matlab-gpu-on-amd-and-intel), and NVIDIA through a single runtime built on [wgpu](https://wgpu.rs/) (the WebGPU standard), which dispatches to Metal on macOS, DirectX 12 on Windows, Vulkan on Linux, and WebGPU in the browser. No CUDA toolkit, no ROCm, no vendor-specific annotations in your code.

The runtime inspects each computation (array sizes, operation types, data dependencies) and decides per-operation whether to run on CPU or GPU. Large, contiguous elementwise chains get *fused* into a single GPU kernel so one launch handles what would otherwise be many. Small or irregular work stays on CPU. Your `.m` file doesn't change.

### Correctness guarantees

Speed only matters if the numbers are right. Every GPU-accelerated builtin in RunMat is parity-tested against its CPU reference on every merge: the GPU kernel must reproduce the CPU result within a documented tolerance (`1e-9` for f64, `1e-3` for f32) or CI fails the build. The parity tests are `cargo test` files in the public repository, so you can reproduce them without a MATLAB license. The full coverage table and per-builtin test links live on the [Correctness & Trust page](/docs/correctness#coverage-table).

### GPU-resident plotting

The "avoid transfers" principle extends to visualization. RunMat's plotting renders directly from GPU memory with zero copy between the computation and the chart, which matters most on pipelines that compute millions of points per frame. See the [MATLAB plotting guide](/blog/matlab-plotting-guide) for runnable examples.

### Where to run it

| Environment | GPU path | Best for |
|-------------|----------|----------|
| Browser ([runmat.com/sandbox](https://runmat.com/sandbox)) | WebGPU | Trying code with no install |
| CLI (`runmat run script.m`) | Metal / DX12 / Vulkan | Scripts, benchmarks, CI, max performance |
| Desktop app (coming soon) | Metal / DX12 / Vulkan | Full IDE + full GPU headroom |

For benchmarks against MATLAB, PyTorch, and NumPy, see [Introducing RunMat Accelerate](/blog/runmat-accelerate-fastest-runtime-for-your-math).

---

## **FAQ: common GPU + MATLAB questions**

**Can I use MATLAB GPU on a Mac?**

No. MATLAB's gpuArray requires NVIDIA CUDA, which does not exist on Apple Silicon. If you try `gpuArray` on an M-series Mac, you'll get "No supported GPU device was found." RunMat uses Metal on macOS, so M1/M2/M3/M4 Macs get GPU acceleration with the same code. See [MATLAB GPU on Apple Silicon](#matlab-gpu-on-apple-silicon-m1-m2-m3-m4).

**Does MATLAB support M1/M2/M3/M4 GPU acceleration?**

No. The Parallel Computing Toolbox is built on NVIDIA CUDA and does not support Apple's Metal API. M-series GPUs are powerful but MATLAB cannot use them. RunMat targets Metal directly. See [MATLAB GPU on Apple Silicon](#matlab-gpu-on-apple-silicon-m1-m2-m3-m4).

**Can MATLAB use an AMD GPU?**

No. MATLAB requires NVIDIA CUDA. AMD GPUs (Radeon, Instinct) are not supported by the Parallel Computing Toolbox. RunMat uses Vulkan (Linux) and DirectX 12 (Windows) for AMD. See [MATLAB GPU on AMD and Intel](#matlab-gpu-on-amd-and-intel).

**Can MATLAB use an Intel GPU?**

No. Intel GPUs (integrated or Arc) are not supported by the Parallel Computing Toolbox. RunMat supports Intel GPUs through Vulkan and DirectX 12. See [MATLAB GPU on AMD and Intel](#matlab-gpu-on-amd-and-intel).

**Do I need CUDA even if I have an NVIDIA GPU?**

For MATLAB, yes. The Parallel Computing Toolbox requires CUDA. With RunMat, no: NVIDIA GPUs are accessed through Vulkan, so you get GPU acceleration without installing CUDA drivers or buying the toolbox.

**Can I use GPU without the Parallel Computing Toolbox?**

In MATLAB, no. The toolbox is required and is a paid add-on. RunMat includes GPU acceleration by default, free, even on NVIDIA hardware. See [GPU acceleration without CUDA](#gpu-acceleration-without-cuda-how-runmat-works) and [free MATLAB alternatives](/blog/free-matlab-alternatives).

**Is there a free way to get GPU acceleration with MATLAB code?**

MATLAB's GPU path requires the Parallel Computing Toolbox ($), CUDA, and an NVIDIA GPU. RunMat is free and accelerates on any vendor without any of those. See [GPU acceleration without CUDA](#gpu-acceleration-without-cuda-how-runmat-works).

**What GPU do I need for MATLAB?**

NVIDIA only, with CUDA compute capability 3.5+. See [NVIDIA + the Parallel Computing Toolbox](#if-you-have-nvidia-the-parallel-computing-toolbox) and [NVIDIA's CUDA GPUs page](https://developer.nvidia.com/cuda-gpus). If you want GPU acceleration on non-NVIDIA hardware or without the toolbox, [RunMat](https://runmat.com) works on any modern GPU.

**Do I need to install CUDA or vendor drivers to use RunMat with an AMD or Intel GPU?**

No. RunMat reaches AMD and Intel GPUs through Vulkan (Linux) and DirectX 12 (Windows), which are already present on most systems. There's no CUDA, no ROCm, and no oneAPI toolchain to install.

**How do I try GPU acceleration on my Mac without installing anything?**

Open [runmat.com/sandbox](https://runmat.com/sandbox) in a WebGPU-capable browser (Safari 18+, Chrome 113+, Edge 113+, Firefox 139+) and paste your `.m` code. The sandbox dispatches to Metal on macOS automatically. For the native CLI on an M-series Mac, [install RunMat](/download) and run `runmat run your_script.m`.

**Why is my GPU slower than my CPU?**

Usually the arrays are too small, you're doing many tiny steps, or you're transferring often (e.g. `gather` or printing in a loop). Batch into larger arrays and call `gather` only once at the end. See [Two GPU traps that apply to any vendor](#two-gpu-traps-that-apply-to-any-vendor).

**Should I use single or double precision on GPU?**

Use what your numerics need. Single (FP32) is faster, uses half the memory, and is the right default for most workloads. Double precision is still available in MATLAB on NVIDIA (performance then depends on the GPU's FP64 capability) and in RunMat on NVIDIA/AMD/Intel. Apple's Metal does not support FP64 at all, so on Mac FP32 is the only GPU option.

**What's the simplest rule for GPU performance?**

Make the work big and contiguous, and avoid transfers. Everything else is a refinement.

---

## **Related reading**

- [Best Free MATLAB Alternatives in 2026](/blog/free-matlab-alternatives) — RunMat, Octave, Julia, and Python compared across 15 dimensions.
- [MATLAB Plotting Guide](/blog/matlab-plotting-guide) — plotting with GPU-resident rendering and runnable examples.
- [Introducing RunMat Accelerate](/blog/runmat-accelerate-fastest-runtime-for-your-math) — benchmarks comparing RunMat's fusion engine against MATLAB, PyTorch, and NumPy.
