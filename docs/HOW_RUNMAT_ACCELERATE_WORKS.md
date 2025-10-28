# Deep Dive: How RunMat Accelerate Works

This document explains how RunMat turns MATLAB syntax into efficient GPU work without asking you to write kernels or worry about drivers. It is a technical deep dive meant for engineers who want to understand the moving parts and the trade‑offs.

## What RunMat Accelerate does (at a glance)

- Picks an acceleration backend (GPU via wgpu, or a CPU fallback) and exposes a uniform interface to the runtime.
- Decides whether each builtin should run on CPU or GPU using size/shape heuristics and live context from fusion.
- When profitable, generates and executes GPU kernels, keeping data resident on device as long as possible.

Think of RunMat Accelerate as a traffic controller between high‑level math and the GPU: it schedules, compiles, dispatches, and measures.

## The Provider abstraction

RunMat works with a "provider", which is an active acceleration backend able to offload work to a GPU. Today that is usually the `wgpu` provider (portable over Metal on macOS, DirectX 12 on Windows, and Vulkan on Linux/ARM), but the architecture is designed to support custom backends if needed. The provider:

- Manages the device and queue; allocates and frees GPU buffers; uploads and downloads tensors.
- Compiles WGSL (the GPU language we generate) into native compute pipelines, caches them, and reuses them across runs.
- Implements hooks for common categories (elementwise math, reductions, matmul, random, and fused kernels).
- Reports timings so the planner can learn when GPU is a win.

If no GPU is available or features are missing, a simple in-process provider steps in, keeping scripts working on the CPU.

## From script to execution: the control flow

1. The runtime lowers your MATLAB syntax code into an acceleration graph: nodes are ops (like `+`, `sin`, `mean`), edges are data.
2. The fusion planner scans the graph to find long elementwise chains and standalone reductions that can be consolidated.
3. The auto-offload logic weighs sizes, shapes, and whether results will stay on device; it picks CPU or GPU for each region.
4. For GPU regions, the fusion layer emits WGSL, the provider compiles (or reuses) pipelines, and dispatches kernels.
5. Results remain resident on device until a host-only sink (e.g., `fprintf`, file I/O) requires a gather back to the CPU.

The end result is fewer kernel launches, less memory traffic, and decisions that adapt to what the script is actually doing.

## Choosing CPU vs GPU (auto-offload)

The auto-offload engine combines simple rules of thumb with live context:

- Element count matters. Very small arrays don’t amortize upload/launch overheads—those stay on the CPU.
- Operation mix matters. Compute-heavy reductions and matmuls tip toward GPU; tiny scalar work does not.
- Residency avoids re-uploads. If an operand is already on the GPU and downstream is GPU-friendly, it often stays there.
- Calibration is optional. On your machine, quick micro-benchmarks can refine thresholds for elementwise, reduction, and matmul.

These decisions are conservative by default and configurable via environment or runtime options.

## Fusion: what we combine and why

Fusion looks for two profitable shapes:

- Elementwise chains that read the same array(s) and apply a sequence of pointwise ops; these become one kernel.
- Single reductions (e.g., sum/mean) where the preceding or following context can reuse the GPU output.

The planner enforces shape/broadcast compatibility, bails out on unknown shapes, and preserves MATLAB semantics (NaN handling, `'like'`, etc.). Successful plans provide enough metadata (element counts, reduce sizes) for the offload engine and the provider.

## Kernel generation and caching

- WGSL generation. For elementwise groups we emit a loop that reads inputs, applies the fused formula, and writes the result. For reductions we emit tiled kernels that use workgroup memory and, for large reduce lengths, optionally switch to a two‑pass strategy.
- Tuning knobs. Providers advertise defaults like reduction workgroup size and the threshold where two‑pass reductions win; env overrides let you tune without code changes.
- Pipeline caching. The provider hashes shader source plus layout/workgroup signature to reuse compute pipelines. Warm‑up can compile common kernels on startup so first‑use is smooth. Timings (upload/compute/download) and cache hit/miss counters feed observability.
- Portability. The same WGSL compiles via wgpu to Metal/DX12/Vulkan—no script changes required.

## Memory model and residency

Accelerate tries to move data once, then keep it on device:

- Uploads happen at region boundaries; scalars are materialized as tiny buffers if needed.
- Intermediate results of fused kernels remain resident (marked in a residency table) so the next GPU region can consume them directly.
- Host sinks (printing, exporting) trigger gathers and clear residency for those handles; correctness beats residency when required.

This model minimizes PCIe (or integrated GPU) copies and preserves the biggest wins.

## Provider lifecycle (initialization and fallback)

On first use, Accelerate selects and registers a provider. With wgpu available, it enumerates adapters, logs device info, and may run a short warm‑up to populate caches. If a GPU is unavailable or disabled, the in‑process provider is installed so scripts still run—slowly, but correctly—with the same semantics.

## Observability and tuning

- Device and cache info. A provider info path can expose device name, backend (Metal/DX12/Vulkan), cache hits/misses, and default tuning values.
- Metrics. Timers around upload, compute, and download inform thresholds and help catch regressions.
- Overrides. Environment variables (e.g., reduction workgroup size, two‑pass threshold) allow quick experimentation.

These levers make behavior inspectable and adjustable without changing the script.

## Putting it together (end-to-end)

1. The runtime captures intent (graph).
2. Fusion identifies GPU-friendly regions.
3. Auto-offload decides CPU vs GPU using sizes and context.
4. WGSL is generated, compiled (or cache‑hit), and dispatched by the provider.
5. Results stay resident until a host sink forces a gather; residency is updated.

This layered design balances MATLAB compatibility, performance portability, and incremental backend support. Builtins describe what they can do on GPU; Accelerate decides when and how to do it efficiently.

## Deeper dive

For a deeper dive into the implementation, see the [RunMat Accelerate source code & documentation](https://github.com/runmat-org/runmat/tree/main/crates/runmat-accelerate), and the [RunMat Runtime Builtin Functions source code & documentation](https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins).