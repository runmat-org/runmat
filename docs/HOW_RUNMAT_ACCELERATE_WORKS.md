# Deep Dive: RunMat Accelerate Architecture

This document explains how RunMat routes high-level MATLAB-compatible builtins onto GPU hardware through the **RunMat Accelerate** subsystem. The focus is on the provider abstraction, auto-offload planner, and how new-style runtime builtins cooperate with the acceleration layer.

## 1. Architectural Overview

RunMat Accelerate defines a backend-agnostic façade that lives in `crates/runmat-accelerate/src/lib.rs`. The crate exposes three main responsibilities:

- Select and initialise an acceleration provider (wgpu, in-process fallback, etc.).
- Maintain heuristics that decide whether a builtin should execute on the CPU or on the selected device.
- Coordinate with the fusion planner to compile larger expression graphs into GPU kernels.

At runtime the `Accelerator` façade (`crates/runmat-accelerate/src/lib.rs`) sits between builtin dispatchers and the provider interface. It receives RunMat `Value` arguments, asks the planner where to execute, and either forwards to the legacy CPU implementation (`runmat_runtime`) or marshals tensors into GPU buffers before calling provider hooks.

## 2. Provider Abstraction (`runmat-accelerate-api`)

Acceleration providers implement the `AccelProvider` trait defined in `crates/runmat-accelerate-api/src/lib.rs`. The trait is extensive: aside from core memory operations (`upload`, `download`, `free`), it contains optional hooks for elementwise math, reductions, linear algebra, signal processing, random number generation, sorting, set operations, cumulative scans, and fused kernels (`fused_elementwise`, `fused_reduction`).

Important pieces:

- **Registration** – `unsafe fn register_provider` installs a `'static` provider instance into a global (`GLOBAL_PROVIDER`). The helper `provider()` returns an `Option<&'static dyn AccelProvider>` for callers.
- **Structured metadata** – providers can override `device_info_struct()` to surface vendor / backend / memory information for user-facing builtins such as `gpuDevice`.
- **Fused kernel entry points** – the fusion engine relies on `fused_elementwise` and `fused_reduction` to submit generated WGSL code and obtain new `GpuTensorHandle` results without falling back to the CPU.
- **Convenience functions** – the API offers helpers such as `try_elem_add` and random tensor initialisers that internally check whether a provider is registered.

Because the trait covers a superset of MATLAB semantics, providers can opt-in incrementally. Missing features return `Err` so Callers can fall back to host code.

## 3. Provider Lifecycle & Initialisation

`AccelerateInitOptions` in `crates/runmat-accelerate/src/lib.rs` packages all runtime configuration (provider preference, power preference, fallback behaviour, auto-offload options). `initialize_acceleration_provider_with` performs the following steps:

1. Applies auto-offload configuration (`configure_auto_offload` ➜ global `Lazy<RwLock<AutoOffloadOptions>>`).
2. Short-circuits if a provider is already registered.
3. Registers the preferred backend:
   - With the `wgpu` feature enabled, it creates a `WgpuProvider` using `backend::wgpu::provider::register_wgpu_provider`, logging adapter details and triggering a warm-up to amortise shader compilation.
   - Otherwise it chooses the in-process provider (`simple_provider::register_inprocess_provider`) as a compatibility fallback.

During initialisation the crate also installs residency hooks via `runmat_accelerate_api::register_residency_clear`, enabling the runtime dispatcher to clear GPU residency metadata when tensors are gathered back to the host (`crates/runmat-accelerate/src/fusion_residency.rs`).

## 4. Planner & Execution Path

The `Planner` (`crates/runmat-accelerate/src/lib.rs`) encapsulates the decision of whether to offload a builtin. The current implementation is intentionally simple: `choose_elem_add` checks tensor sizes and shape compatibility before returning `ExecutionTarget::Gpu` or `ExecutionTarget::Cpu`. The `Accelerator::elementwise_add` façade demonstrates the standard flow:

1. Gather GPU handles into host tensors if required (e.g., when one operand is still a `GpuTensorHandle` but the provider lacks buffer bookkeeping).
2. For CPU execution, call back into `runmat_runtime`.
3. For GPU execution, upload host tensors with `provider.upload`, invoke the elementwise hook, download the result, and wrap it into a `Value::Tensor`.

Higher-level builtins follow the same pattern after passing through the new dispatcher. The planner will evolve to consult profiling data produced by `NativeAutoOffload`.

## 5. Native Auto-Offload Engine

`crates/runmat-accelerate/src/native_auto.rs` manages heuristics that automatically promote builtin arguments to GPU residency. Key details:

- **Global singleton** – the first call invokes `initialize`, checks configuration flags (environment variables or `AutoOffloadOptions`), and captures the active provider reference.
- **Thresholds** – `ThresholdConfig` stores minimum element counts / FLOPs before GPU execution becomes worthwhile. Defaults are conservative and can be overridden at runtime.
- **Calibration** – optional `auto_calibrate` benchmarks (`compare_elemwise`, `compare_reduction`, `compare_matmul`) run both CPU and provider kernels to refine thresholds.
- **Dynamic decisions** – methods like `promote_binary` and `promote_reduction` reuse cached GPU handles when possible, consult the fusion engine (`active_fusion`) to factor in upcoming element counts, and fall back to gathering values for sink builtins.
- **`'like'` handling** – reduction helpers honour MATLAB semantics such as `'like'` or `'native'` prototypes by re-uploading host results when requested.

These heuristics let front-end code remain MATLAB-friendly while opportunistically keeping tensors resident on the GPU.

## 6. Runtime Builtins & GPU Specs

New-style builtins inside `crates/runmat-runtime/src/builtins` record their GPU capabilities using inventory macros (`register_builtin_gpu_spec!`) and describe fusion templates via `BuiltinFusionSpec` (`crates/runmat-runtime/src/builtins/common/spec.rs`). For example:

- `math/elementwise/exp.rs` declares an elementwise GPU spec that points to the provider hook `unary_exp` and a fusion template that emits WGSL for single-input exponentials.
- `math/reduction/sum.rs` registers a reduction spec referencing `reduce_sum_dim`, marks omit-nan requirements, and exposes metadata used by the fusion planner (e.g., preferred workgroup size).

At execution time each builtin:

1. Validates / broadcasts arguments using helpers from `builtins/common`.
2. Consults provider hooks through `runmat_accelerate_api::provider()`.
3. Falls back to the CPU implementation when the provider reports `Err`.
4. Delegates residency decisions to `NativeAutoOffload` for `'like'` or auto-offload scenarios.

The dispatcher (`crates/runmat-runtime/src/dispatcher.rs`) automatically gathers GPU tensors when a builtin rejects device residency, ensuring MATLAB compatibility without extra user code.

## 7. WGPU Provider Implementation

The default GPU backend, `WgpuProvider` (`crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`), owns:

- The `wgpu::Device` / `Queue`.
- A handle table (`Mutex<HashMap<u64, BufferEntry>>`) used to translate `GpuTensorHandle::buffer_id` back into `wgpu::Buffer` objects.
- A `WgpuPipelines` bundle (`crates/runmat-accelerate/src/backend/wgpu/pipelines.rs`) that lazily creates compute pipelines for elementwise math, reductions, scans, permutations, convolution, random number generation, etc. Pipeline layouts share a consistent binding scheme so generated WGSL from the fusion engine can be compiled and cached (`fused_pipeline_cache`).
- Fused kernel helpers that hash shader sources to reuse compute pipelines across invocations. The warm-up pass executed during provider registration primes caches by dispatching representative kernels and reporting hit/miss counts for observability.
- Metrics (`metrics::WgpuMetrics`) that track transfer / dispatch timings. These counters feed the auto-offload cost model via the profiling subsystem.

Because the provider implements most `AccelProvider` hooks, builtins can offload a broad spectrum of operations while retaining a CPU fallback path.

## 8. In-Process Provider Fallback

When GPU acceleration is disabled or unavailable, `simple_provider::register_inprocess_provider` registers `InProcessProvider` from `crates/runmat-accelerate/src/simple_provider.rs`. This shim stores tensors in a `HashMap` keyed by `buffer_id` and implements hooks by delegating to host algorithms from `runmat_runtime`. It exists to preserve builtins that expect GPU handles without forcing every configuration to ship with a real GPU backend.

## 9. Residency Tracking & Gather Semantics

Some GPU tensors stay resident across fused kernels. The fusion subsystem marks handles via `fusion_residency::mark` and clears them when a gather occurs. The runtime dispatcher (`gather_if_needed`) invokes the `runmat_accelerate_api::clear_residency` hook before reconstructing a dense tensor, which lets higher-level logic know the buffer is no longer live on the device.

## 10. Putting It Together

A typical accelerated builtin call flows as follows:

1. User code invokes a builtin that was generated by `runmatfunc` (e.g., `sum`).
2. The builtin checks for provider support and invokes `NativeAutoOffload` to promote inputs when profitable.
3. The fusion planner inspects the surrounding `AccelGraph` (if the builtin participates in a fusion opportunity) and, when successful, emits WGSL via `FusionGroupPlan::generate_wgsl` or `generate_reduction_wgsl`.
4. `fusion_exec` submits the kernel to the provider using `fused_elementwise`/`fused_reduction`. Otherwise the builtin calls a direct provider hook (e.g., `reduce_sum_dim`).
5. Results stay on the GPU unless the caller explicitly requests a gather or the provider lacks an implementation, in which case the dispatcher materialises a CPU tensor.

This layered design lets RunMat balance MATLAB compatibility, performance portability, and incremental backend support while keeping the new builtin surface area focused on declarative specs rather than device plumbing.
