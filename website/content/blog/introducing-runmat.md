---
title: "Introducing RunMat: A Fast, Free, Modern MATLAB Code Runtime"
description: "A fast, open-source runtime for MATLAB code. Slim core written in Rust, V8-inspired execution, generational GC, and a package-first standard library."
date: "2025-08-07"
authors:
  - name: "Nabeel Allana"
    url: "https://x.com/nabeelallana"
readTime: "7 min read"
slug: "introducing-runmat"
tags: ["MATLAB", "Rust", "JIT", "Octave", "scientific computing", "open source"]
keywords: "MATLAB runtime, GNU Octave comparison, Rust scientific computing, JIT compiler, numerical computing, open source"
excerpt: "RunMat is a modern, open-source runtime that executes MATLAB code quickly. A slim core, tiered execution, and a package system make it fast, predictable, and easy to extend."
image: "/plot-example.jpg"
imageAlt: "RunMat plot example"
ogType: "article"
ogTitle: "Introducing RunMat: Fast, Free, Modern MATLAB Code Runtime"
ogDescription: "Run MATLAB code with performance and Rust safety, completely free."
twitterCard: "summary_large_image"
twitterTitle: "RunMat: A modern, fast MATLAB code runtime built in Rust"
twitterDescription: "Slim core, tiered execution, generational GC. Open-source, designed for performance and extensibility."
canonical: "https://runmat.org/blog/introducing-runmat"
jsonLd:
  "@context": "https://schema.org"
  "@graph":
    - "@type": "BreadcrumbList"
      itemListElement:
        - "@type": "ListItem"
          position: 1
          name: "RunMat"
          item: "https://runmat.org"
        - "@type": "ListItem"
          position: 2
          name: "Blog"
          item: "https://runmat.org/blog"
        - "@type": "ListItem"
          position: 3
          name: "Introducing RunMat"
          item: "https://runmat.org/blog/introducing-runmat"

    - "@type": "BlogPosting"
      "@id": "https://runmat.org/blog/introducing-runmat#article"
      headline: "Introducing RunMat: A Fast, Free, Modern MATLAB Code Runtime"
      alternativeHeadline: "RunMat vs GNU Octave: Performance Benchmarks"
      description: "RunMat is a modern, open-source runtime that executes MATLAB code quickly using a Rust-based engine, tiered JIT execution, and automatic GPU acceleration."
      image: "https://runmat.org/plot-example.jpg"
      datePublished: "2025-08-07T00:00:00Z"
      dateModified: "2025-08-07T00:00:00Z"
      author:
        "@type": "Person"
        name: "Nabeel Allana"
        url: "https://x.com/nabeelallana"
      publisher:
        "@type": "Organization"
        name: "Dystr Inc."
        logo:
          "@type": "ImageObject"
          url: "/runmat-logo.svg"
          width: 600
          height: 60
      about:
        - "@type": "SoftwareApplication"
          name: "RunMat"
          sameAs: "https://runmat.org"
          applicationCategory: "ScientificApplication"
          operatingSystem: ["Windows", "macOS", "Linux"]
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
        - "@type": "SoftwareApplication"
          name: "MATLAB"
          sameAs: "https://en.wikipedia.org/wiki/MATLAB"
          applicationCategory: "ScientificApplication"
          operatingSystem: ["Windows", "macOS", "Linux"]
        - "@type": "SoftwareApplication"
          name: "GNU Octave"
          sameAs: "https://en.wikipedia.org/wiki/GNU_Octave"
          applicationCategory: "ScientificApplication"
          operatingSystem: ["Windows", "macOS", "Linux"]
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
        - "@type": "ComputerLanguage"
          name: "Rust"
          sameAs: "https://en.wikipedia.org/wiki/Rust_(programming_language)"

    - "@type": "SoftwareApplication"
      name: "RunMat"
      operatingSystem: ["Linux", "macOS", "Windows"]
      applicationCategory: "Scientific Computing"
      license: "https://opensource.org/licenses/MIT"
      featureList: "JIT Compilation, GPU Acceleration, MATLAB Compatibility"
      offers:
        "@type": "Offer"
        price: "0"
        priceCurrency: "USD"

    - "@type": "FAQPage"
      mainEntity:
        - "@type": "Question"
          name: "What is RunMat?"
          acceptedAnswer:
            "@type": "Answer"
            text: "RunMat is a modern, open-source runtime built in Rust that executes MATLAB code. It features a slim core, tiered execution (interpreter + JIT), and a package-first standard library."
        - "@type": "Question"
          name: "Is RunMat faster than GNU Octave?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. Benchmarks on an Apple M2 Max show RunMat is <b>150x–180x faster</b> than GNU Octave for startup time, matrix operations, and control flow loops."
        - "@type": "Question"
          name: "Does RunMat require a MATLAB license?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. RunMat is a standalone, free, and open-source project. It is not affiliated with MathWorks and does not use any proprietary MATLAB code."
---

## TL;DR

- **RunMat** is a modern, open-source runtime that executes MATLAB code fast.
- We implement the **full language grammar** and the **core semantics** (arrays, indexing, control flow, functions, cells/structs, OOP).
- The core stays **small and fast**; everything else grows via a **package system** (native Rust or source MATLAB).
- Core built-ins are **canonical** (e.g., `sin`, `cos`, `sum`, and `printf`-style formatting like `fprintf`/`sprintf`) and match the expected, documented behavior. When semantics are domain-specific or ambiguous, they live in packages.
- Built in **Rust**, with **tiered execution** (interpreter first, JIT for hot code) and a **generational GC** tuned for numerics.
- Benchmarks show **150x–180x speedups** vs GNU Octave on representative workloads; see Performance below.
- New to MATLAB? Start with our primer: [What is MATLAB? The Language, The Runtime, and RunMat](/blog/what-is-matlab).

---

## Why another runtime?

If you've written MATLAB code, you know the trade-offs:

- MATLAB is powerful but proprietary and heavy to start; deployment is license-bound.
- GNU Octave is free and compatible with lots of code, but startup and hot-path performance can be limiting.
- Moving to a new language means rewriting and - perhaps most importantly - **retraining**.

RunMat aims for a fourth path: keep the MATLAB language you know, but put it on a modern engine with a smaller core, clean semantics, and open extensibility.

---

## Language compatibility at a glance

A quick view of core language semantics. Full details: [here](/docs/language-coverage).

| Feature Category | RunMat | Octave |
| :-- | :--: | :--: |
| Grammar & parser (full MATLAB surface) | ✅ | ✅ |
| Arrays & indexing (`end`, colon, logical masks, N-D slicing) | ✅ | ✅ |
| Multiple returns, `varargin`/`varargout`, `nargin`/`nargout` | ✅ | ✅ |
| OOP `classdef` (props/methods), operator overloading | ✅ | ❌ |
| Events/handles (`addlistener`, `notify`, `isvalid`, `delete`) | ✅ | ❌ |
| Imports precedence & static access (`Class.*`) | ✅ | ❌ |
| Metaclass operator `?Class` | ✅ | ❌ |
| String arrays (double-quoted) | ✅ | ❌ |
| Standardized `MException` identifiers | ✅ | ❌ |

If something you rely on is not in the core, packages are the intended extension point.

---

## What RunMat is (and is not)

What it is:

- A new runtime that accepts MATLAB syntax and executes the core semantics quickly.
- A slim, production-oriented engine written in Rust with a stable Value/Type/ABI.
- A system that grows through packages: built-ins implemented in Rust or MATLAB.
- A **predictable core** of canonical built-ins (math, array ops, formatting/IO) with stable behavior; broader or niche functionality ships as packages.

What it is not:

- Not a re-packaging of MATLAB. We don't ship MATLAB code, assets, or toolboxes.
- Every historical builtin. We prioritize a small, consistent core and let packages provide breadth.
- Not affiliated with MathWorks; not a drop-in replacement for every workflow.

Legal clarity: RunMat is an independent project that implements a compatible language runtime. “MATLAB” is a MathWorks trademark; we use it nominatively to describe the language whose grammar and semantics our compiler/interpreter accepts. We are not endorsed by or associated with MathWorks.

---

## Performance

On an Apple M2 Max (32GB), our micro-benchmarks (matrix ops, math functions, control-flow loops) show large speedups over GNU Octave on the same machine:

### Summary results

| Benchmark | GNU Octave avg (s) | RunMat interp avg (s) | RunMat JIT avg (s) | Speedup vs Octave |
| :-- | --: | --: | --: | --: |
| Startup Time | 0.9147 | 0.0050 | 0.0053 | <span className="text-green-600 font-semibold">172x–183x faster</span> |
| Matrix Operations | 0.8220 | 0.0050 | 0.0050 | <span className="text-green-600 font-semibold">164x faster</span> |
| Mathematical Functions | 0.8677 | 0.0057 | 0.0053 | <span className="text-green-600 font-semibold">153x–163x faster</span> |
| Control Flow | 0.8757 | 0.0057 | 0.0057 | <span className="text-green-600 font-semibold">155x faster</span> |

- We don't compare to MATLAB here due to licensing constraints (we decline to install Matlab and agree with their license terms). Our focus is the design: a slim core and a modern engine.
- Benchmarks are in the repo under `/benchmarks` with a script to reproduce. Numbers vary by hardware, BLAS, and build settings; please measure on your workload. To reproduce locally:

```bash
cd benchmarks
./run_benchmarks.sh
cat results/benchmark_YYYYMMDD_HHMMSS.yaml
```

For a broader landscape view, see our comparison of RunMat vs Octave, Julia, and Python in the
[MATLAB alternatives guide](/blog/free-matlab-alternatives).

---

## How it works

- **Ignition interpreter**: immediate execution, great for REPL and scripts.
- **Turbine JIT**: hot functions get compiled to optimized machine code (Cranelift backend).
- **Slim builtins**: a curated set in core; everything else via packages. Docs are generated from runtime metadata.
- **Great developer experience**: built in Jupyter kernel, flow-sensitive inference for great autocomplete and type hints, and more.
- **Portable**: single binary, no dependencies, runs on Linux/macOS/Windows and embedded devices.
- **GPU-optimized**: built in, configurable, swappable GPU planner with automatic fusion and data residency. Run your code on GPUs without any modifications across CPU, Metal (macOS), DirectX 12 (Windows), and Vulkan (Linux) via the wgpu backend. Additional backends (CUDA, ROCm, OpenCL) are planned.

For a deeper dive, see [How It Works](/docs/how-it-works) and the [Architecture & Internals](/docs/architecture) section.

---

## Packages: extending the runtime

Two ways to add capabilities:

- **Native (Rust) packages**: implement built-ins with `#[runtime_builtin]`, get strong typing and speed, and ship as a dynamic library.
- **Source (MATLAB) packages**: ship `.m` files; RunMat interprets or compiles them.

Documentation is generated from runtime metadata, so everything you add shows up in the reference automatically. See the [Package Manager](/docs/package-manager) doc for the full design and examples.

---

## What you can run today

- Core language: arrays, slicing (`end`, colon, logical masks), functions and multiple returns, cells/structs, OOP (`classdef` with properties/methods), `try/catch`, `global`, `persistent`, function handles, command-form.
- Extensive builtin coverage in the runtime (canonical math like `sin/cos/tan`, reductions like `sum/min/max`, basic string/formatting via `fprintf/sprintf`, array creation like `zeros/ones/eye`, linear algebra, FFT/signal processing, statistics, and I/O), with additional functions available through packages.

If your code relies on many niche built-ins, the recommended path is to move those pieces into packages. The docs call out differences and migration notes where they exist.

A note about about plotting: RunMat's plot package is a work in progress. It's near-complete, but not yet ready for production use. We're working on it, and will have a complete plotting system in the next few releases.

---

## MATLAB, Octave, RunMat — a quick contrast

- **MATLAB**: proprietary, massive standard library, heavy startup, license-gated deployment. Feature-rich; closed source.
- **GNU Octave**: free, community-driven project with partial compatibility with MATLAB scripts. It carries a broad builtin surface and a classic interpreter architecture, but startup times and hot-loop performance can be poor. Octave is a full application; its design optimizes for breadth and compatibility.
- **RunMat**: open-source, modern engine with full language grammar and core semantics. We deliberately keep the core small and fast (canonical built-ins only) and move breadth into packages. This lets us optimize the engine aggressively for performance and predictability while still enabling a large library surface via the package system.

---

## Try it & get involved

- Read the docs: Getting Started, Design Philosophy, and How It Works.
- Browse the builtin reference; it's generated from the runtime.
- Star the repo and open issues: https://github.com/runmat-org/runmat
- Interested in contributing? Packages are the best place to start (Rust or MATLAB source).

---

*RunMat is not affiliated with MathWorks, Inc. “MATLAB” is a registered trademark of MathWorks, Inc. We reference it nominatively to describe the language whose grammar and semantics our independent runtime accepts.*

*RunMat is a free, open source community project developed by [Dystr](https://dystr.com).* 