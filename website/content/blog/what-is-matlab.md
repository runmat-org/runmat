---
title: "What is MATLAB? The Language, The Runtime, and RunMat"
description: "A plain-English explainer of MATLAB's language model and how RunMat delivers a modern, high-performance open-source runtime for it."
date: "2025-01-01"
dateModified: "2025-02-01"
image: "https://web.runmatstatic.com/blog-images/matrix-math.png"
imageAlt: "What is MATLAB"
authors:
  - name: "Julie Ruiz"
    url: "https://www.linkedin.com/in/julie-ruiz-64b24328/"
readTime: "10 min read"
slug: "what-is-matlab"
tags: ["MATLAB", "RunMat", "language", "runtime"]
visibility: unlisted
canonical: "https://runmat.com/blog/what-is-matlab"
jsonLd:
  "@context": "https://schema.org"
  "@graph":
    - "@type": "BreadcrumbList"
      itemListElement:
        - "@type": "ListItem"
          position: 1
          name: "Docs"
          item: "https://runmat.com/docs"
        - "@type": "ListItem"
          position: 2
          name: "Concepts"
          item: "https://runmat.com/docs/concepts"
        - "@type": "ListItem"
          position: 3
          name: "What is MATLAB?"
          item: "https://runmat.com/blog/what-is-matlab"
    - "@type": "TechArticle"
      "@id": "https://runmat.com/blog/what-is-matlab#article"
      headline: "What is MATLAB? The Language, The Runtime, and RunMat"
      alternativeHeadline: "MATLAB vs RunMat: A Technical Overview"
      description: "An engineer's guide to the MATLAB language, its mental model (arrays, matrices, indexing), and how RunMat provides a modern, high-performance open-source runtime for it."
      proficiencyLevel: "Beginner"
      datePublished: "2025-01-01T00:00:00Z"
      dateModified: "2025-02-01T00:00:00Z"
      author:
        "@type": "Organization"
        name: "Dystr Inc."
        url: "https://dystr.com"
        alternateName: "RunMat Team"
      publisher:
        "@type": "Organization"
        name: "Dystr Inc."
        logo:
          "@type": "ImageObject"
          url: "/runmat-logo.svg"
      about:
        - "@type": "SoftwareApplication"
          name: "MATLAB"
          sameAs: "https://en.wikipedia.org/wiki/MATLAB"
          applicationCategory: "Numerical Computing"
          operatingSystem: ["Windows", "macOS", "Linux"]
        - "@type": "SoftwareApplication"
          name: "RunMat"
          url: "https://runmat.com"
          applicationCategory: "Runtime Environment"
          operatingSystem: ["Windows", "macOS", "Linux"]
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
        - "@type": "SoftwareApplication"
          name: "GNU Octave"
          sameAs: "https://en.wikipedia.org/wiki/GNU_Octave"
          applicationCategory: "ScientificApplication"
          operatingSystem: ["Windows", "macOS", "Linux"]
          offers:
            "@type": "Offer"
            price: "0"
            priceCurrency: "USD"
      mainEntity: { "@id": "#faq" }
    - "@type": "FAQPage"
      "@id": "#faq"
      mainEntity:
        - "@type": "Question"
          name: "What is MATLAB?"
          acceptedAnswer:
            "@type": "Answer"
            text: "MATLAB is a programming language and computing environment designed for working with matrices, vectors, and numeric data. It focuses on array-first programming where most operations are expressed as high-level array math."
        - "@type": "Question"
          name: "How is RunMat different from MATLAB and Octave?"
          acceptedAnswer:
            "@type": "Answer"
            text: "<b>MATLAB</b> is proprietary software with a massive toolbox ecosystem. <b>GNU Octave</b> is an open-source alternative aiming for compatibility but lacks modern JIT performance. <b>RunMat</b> is a high-performance, open-source runtime that executes MATLAB code using modern tiered compilation (JIT) and automatic GPU acceleration."
        - "@type": "Question"
          name: "Does RunMat support MATLAB indexing and matrix operations?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. RunMat implements core MATLAB semantics including 1-based indexing, matrix vs. element-wise operations (<code>*</code> vs <code>.*</code>), and <code>classdef</code> OOP."
        - "@type": "Question"
          name: "How does RunMat handle performance?"
          acceptedAnswer:
            "@type": "Answer"
            text: "RunMat uses <b>Fusion</b> to capture chains of array operations and automatically route them to the CPU or GPU based on cost. It also features a tiered execution model with a fast-start interpreter and a JIT compiler for hot loops."
        - "@type": "Question"
          name: "Is MATLAB free?"
          acceptedAnswer:
            "@type": "Answer"
            text: "No. MATLAB is proprietary software from MathWorks, requiring paid licenses that can cost thousands of dollars per year (especially with toolboxes). However, <b>RunMat</b> is a free, open-source runtime that executes MATLAB code without requiring a MathWorks license."
        - "@type": "Question"
          name: "Why is MATLAB slow?"
          acceptedAnswer:
            "@type": "Answer"
            text: "MATLAB can feel slow for several reasons: unvectorized loops, interpreter overhead, and lack of automatic GPU acceleration. RunMat addresses these with a tiered execution model (fast interpreter + JIT compiler) and <b>Fusion</b>, which automatically routes array operations to CPU or GPU based on cost."
        - "@type": "Question"
          name: "Can MATLAB run on GPU?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Traditional MATLAB requires the Parallel Computing Toolbox and explicit <code>gpuArray</code> calls to use the GPU. RunMat takes a different approach: its <b>Fusion</b> engine automatically detects when GPU execution would be faster and routes operations there transparently—no code changes required."
        - "@type": "Question"
          name: "Can I run MATLAB code without MATLAB?"
          acceptedAnswer:
            "@type": "Answer"
            text: "Yes. <b>RunMat</b> is an open-source runtime that executes MATLAB code without requiring a MathWorks license. It implements MATLAB's core language semantics (not just similar syntax) and focuses on high performance via JIT compilation and automatic GPU acceleration."
---



MATLAB is a **programming language and computing environment** designed for working with **matrices, vectors, and numeric data**. The name comes from *MATrix LABoratory*, and that focus shows up everywhere: most values are arrays, and most operations are expressed as high-level array math.

In practice, when people say “MATLAB”, they may mean a few different things:

- **The language**: the `.m` syntax and its semantics (arrays, indexing rules, functions, classes, errors).
- **The standard library**: thousands of built-in functions (math, statistics, signal processing, images, optimization, etc.).
- **Toolboxes**: optional domain packages distributed by MathWorks (and sometimes third parties).
- **The IDE and workflow**: editor, debugger, workspace browser, plotting tools, and interactive REPL-like experience.

This FAQ explains MATLAB at a language/concept level, then explains how **RunMat** relates to it.

---

## Why MATLAB is popular

MATLAB has stayed widely used in engineering and research because it makes a few things unusually easy:

- **Array-first programming**: express computations at the level of matrices/vectors instead of manual loops.
- **Fast iteration**: interactive exploration, quick plots, and a tight “edit → run → inspect” loop.
- **Numerics and linear algebra**: first-class support for matrix operations, decompositions, solvers, and signal/image workflows.
- **Readable math**: code often resembles the equations you’d write on a whiteboard.

Even when teams eventually move production systems to other languages, MATLAB often remains the fastest way to prototype and validate the math.

---

## The MATLAB mental model (language basics)

### Arrays are the default

In MATLAB, scalars are 1×1 arrays, vectors are 1-D arrays, and matrices are 2-D arrays. Many operations are defined in terms of array algebra.

### Matrix vs element-wise operations

MATLAB distinguishes **matrix operations** from **element-wise operations**:

- `A * B` is matrix multiplication.
- `A .* B` is element-wise multiplication.
- `A ^ 2` is matrix power (when defined).
- `A .^ 2` is element-wise power.

This is one of the most important “MATLAB-isms” to internalize.

### 1-based indexing and slicing

MATLAB indexes from **1**, not 0:

- `A(1,1)` is the first element.
- `A(:, 3)` means “all rows, column 3”.
- `A(2:10)` means elements 2 through 10.
- `end` can appear in indexing and means “the last index in this dimension”.

### Scripts and functions live in `.m` files

- A **script** is a sequence of statements executed in order.
- A **function** introduces its own scope and can return multiple outputs.

### Control flow exists, but vectorization is idiomatic

Loops and `if` statements exist and are heavily used, but MATLAB style often prefers operations that work on whole arrays at once.

---

## MATLAB vs Octave vs “MATLAB-like”

- **MATLAB (MathWorks)** is proprietary software with a large integrated library/toolbox ecosystem.
- **GNU Octave** is an open-source environment that aims for MATLAB compatibility, but it differs in many corners (especially around some newer language features and library/toolbox behavior).
- Many systems claim “MATLAB-like syntax”, but may differ significantly in semantics (especially indexing rules, type behavior, and edge cases).

If you care about correctness of existing MATLAB code, **language semantics** matter as much as surface syntax.

---

## Where RunMat fits

RunMat is a **high-performance, open-source runtime for MATLAB code**.

It’s helpful to separate what MATLAB “is” into parts:

- **Language**: grammar + semantics
- **Library/toolboxes**: breadth of functions
- **IDE**: editor/debugger/workflow

RunMat’s strategy (by design) is:

- **Implement MATLAB grammar and core semantics** in the runtime.
- Ship a **smaller core standard library**, with a clear path to grow coverage.
- Treat “toolbox breadth” as **packages** (rather than baking everything into the core).
- Focus on **performance and portability**, especially for array-heavy numeric workloads.

As an open source project, RunMat’s framing is discussed explicitly in [Design Philosophy](/docs/design-philosophy).

---

## MATLAB compatibility in RunMat

RunMat’s goal is to run MATLAB code with **principled semantics parity**, not just “similar syntax”.

What’s already strongly covered:

- Core language constructs: control flow, functions, closures, errors/exceptions.
- MATLAB-style indexing and slicing: including `end` arithmetic and logical masks.
- Many “hard” language features: `classdef` OOP, operator overloading, imports/name resolution.

For a detailed feature-by-feature status, see [Language Coverage](/docs/language-coverage).

What to expect in practice:

- Many `.m` scripts run with few or no changes.
- If a specific builtin/toolbox function isn’t present yet, that’s a **library coverage** gap (not necessarily a language gap).
- RunMat prefers **documented behavior** and stable error identifiers over copying historical quirks.

---

## Performance: why RunMat exists

RunMat is built around the idea that you shouldn’t have to rewrite MATLAB-style math to get modern performance.

### Automatic CPU + GPU choice (Fusion)

RunMat can capture chains of array operations, **fuse** them into larger kernels, and then **route execution to CPU or GPU** based on cost heuristics (size/shape/transfer cost). This is how RunMat targets “GPU speed without GPU programming.”

Key properties about RunMat Fusion:

- **No device flags** for typical code paths: the runtime chooses CPU vs GPU automatically.
- **Cross-platform GPU backend** via `wgpu`, targeting **Metal (macOS), DirectX 12 (Windows), Vulkan (Linux)**.
- **Residency awareness**: keep arrays on device when it’s faster, and avoid unnecessary transfers.

To read more about RunMat Fusion, see [Introduction to RunMat GPU/Fusion](/docs/accelerate/fusion-intro).

### Tiered CPU execution (fast startup + fast hot loops)

RunMat also invests heavily in the CPU story:

- A fast-start interpreter tier (Ignition).
- A JIT compiler tier (Turbine/Cranelift) for hot paths.
- A generational garbage collector (GC) tuned for numeric workloads.

---

## Quick start: try RunMat on MATLAB-style code

The fastest way to try RunMat is in your browser—no install required. <a href="/sandbox" data-ph-capture-attribute-destination="sandbox" data-ph-capture-attribute-source="blog-what-is-matlab" data-ph-capture-attribute-cta="try-sandbox">Open the RunMat sandbox</a> and start running MATLAB-style code immediately. WebGPU acceleration is available in supported browsers (Chrome, Edge).

For local use, RunMat is CLI-first. Common entry points:

```bash
# Start an interactive REPL
runmat

# Run a .m file
runmat my_script.m

# Benchmark a script
runmat benchmark my_script.m --iterations 5 --jit

# Inspect acceleration provider / GPU status
runmat accel-info
```

For the full CLI reference, see [CLI](/docs/cli).

---

## Common questions about MATLAB

### Is MATLAB free?

No. MATLAB is proprietary software from MathWorks, and licenses can be expensive—often thousands of dollars per year, especially when you add toolboxes. Academic licenses are cheaper but still not free, and they come with usage restrictions.

**RunMat** is a free, open-source alternative. It executes MATLAB code without requiring a MathWorks license, making it accessible to hobbyists, students, and teams who want to run `.m` files without the licensing overhead.

### Why is MATLAB slow?

MATLAB's performance reputation is mixed. Well-vectorized code can be fast, but several things commonly cause slowdowns:

- **Unvectorized loops**: MATLAB's interpreter has overhead on each iteration. Writing explicit `for` loops over large arrays is often slower than expressing the same logic as array operations.
- **Interpreter overhead**: Even vectorized code goes through an interpreter layer before hitting optimized BLAS/LAPACK routines.
- **No automatic GPU**: Traditional MATLAB requires explicit `gpuArray` calls (and the Parallel Computing Toolbox) to use GPU acceleration.

RunMat addresses these with a **tiered execution model**—a fast-start interpreter (Ignition) plus a JIT compiler (Turbine) for hot paths—and **Fusion**, which automatically routes array operations to CPU or GPU based on cost heuristics.

### Can MATLAB run on GPU?

Yes, but not automatically. Traditional MATLAB requires the **Parallel Computing Toolbox** (an additional paid license) and explicit code changes: you must wrap arrays in `gpuArray()` and manage transfers yourself.

RunMat takes a different approach. Its **Fusion** engine captures chains of array operations, decides whether CPU or GPU execution would be faster, and routes accordingly—no code changes, no device flags, no extra toolbox. This works across platforms via `wgpu` (Metal on macOS, DirectX 12 on Windows, Vulkan on Linux).

For details, see [Introduction to RunMat GPU/Fusion](/docs/accelerate/fusion-intro).

### Can I run MATLAB code without MATLAB?

Yes. That's exactly what **RunMat** is built for: a high-performance, open-source runtime that executes MATLAB code without requiring a MathWorks license.

RunMat focuses on **semantic compatibility**—not just "similar syntax" but correct behavior for MATLAB's indexing, array semantics, `classdef` OOP, and more. Many `.m` scripts run with few or no changes.

If a specific toolbox function isn't available yet, that's a library coverage gap (which grows over time), not a fundamental language incompatibility.

---

## What RunMat is (and is not)

- **RunMat is**: a modern runtime that executes MATLAB code, with a strong focus on performance (CPU + GPU) and a clean extension story.
- **RunMat is not**: “MATLAB-in-full” bundled into one proprietary distribution; toolbox breadth is intended to grow via packages over time.

---

If you’d like to learn more about RunMat, please visit Runmat.org, or read the following articles:

- **Language compatibility**: [Language Coverage](/docs/language-coverage)
- **Why the project is structured this way**: [Design Philosophy](/docs/design-philosophy)
- **How GPU acceleration works**: [Introduction to RunMat GPU/Fusion](/docs/accelerate/fusion-intro)
- **How to run/benchmark/configure RunMat**: [CLI](/docs/cli)
- **Toolboxes-as-packages direction**: [Package Manager (design)](/docs/package-manager)

---

## Trademark notice

MATLAB® is a registered trademark of The MathWorks, Inc. RunMat is not affiliated with, endorsed by, or sponsored by The MathWorks, Inc.
