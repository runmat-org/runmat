## RunMat Accelerate

### Purpose
`runmat-accelerate` provides the high-level acceleration layer that integrates GPU backends with the language runtime. It implements provider(s) for `runmat-accelerate-api` so that `gpuArray`, `gather`, and accelerated math and linear algebra can execute on devices transparently where appropriate.

### Architecture
- Depends on `runmat-accelerate-api` to register an `AccelProvider` implementation at startup.
- Backends (e.g., `wgpu`, `cuda`, `rocm`, `metal`, `vulkan`, `opencl`) are feature-gated. Only one provider is registered globally, but a future multi-device planner can fan out.
- `Planner` decides when to run ops on CPU vs GPU (size thresholds, op types, fusion opportunities). `Accelerator` exposes ergonomic entry points used by the runtime or higher layers.

### Fusion and optimization
- The planner and fusion engine automatically fuse common elementwise chains and reductions to reduce temporaries and host↔device transfers. Fusion detects compatible operations and generates optimized GPU kernels.
- For providers that expose fused kernels, the planner routes operations to those paths, improving performance by minimizing kernel launches and memory traffic.
- Autograd support (planned): Tensor/Matrix operations will participate in reverse-mode autograd by default. The runtime will record a compact tape of primitive ops; gradients will be computed by chaining primitive derivatives.

### What it provides today
- A fully functional `Accelerator` with automatic CPU/GPU routing: the planner chooses CPU path (delegating to `runmat-runtime`) or GPU path (via provider methods) based on tensor sizes, operation types, and fusion opportunities.
- Integration points for `gpuArray`/`gather`: when a provider is registered, runtime builtins route through the provider API defined in `runmat-accelerate-api`.
- Fusion engine that detects and executes fused elementwise chains and reductions on GPU.
- Native auto-offload that transparently promotes tensors to GPU when beneficial.

### How it fits with the runtime
- The MATLAB-facing builtins (`gpuArray`, `gather`) live in `runmat-runtime` for consistency with all other builtins. They call into `runmat-accelerate-api::provider()`, which is implemented and registered by this crate.
- This separation avoids dependency cycles and keeps the language surface centralized while enabling pluggable backends.

### Backends
- `wgpu` (feature: `wgpu`) is the primary cross-vendor backend, providing support for Metal (macOS), DirectX 12 (Windows), and Vulkan (Linux) through a single portable implementation.
- Additional backends (CUDA/ROCm/OpenCL) are planned (features already stubbed).
- Backend responsibilities:
  - Allocate/free buffers, handle host↔device transfers
  - Provide kernels for core ops (elementwise, transpose, matmul/GEMM, reductions, signal processing)
  - Report device information (for planner decisions)
  - Compile and cache compute pipelines for optimal performance

### Current state
- Fully functional wgpu backend with cross-platform GPU support (Metal/DX12/Vulkan).
- CPU fallback path fully functional; GPU path implemented and production-ready.
- Fusion engine operational, automatically detecting and fusing elementwise chains and reductions.
- Native auto-offload heuristics working, transparently promoting operations to GPU when beneficial.

### Roadmap
- ✅ In-process provider with buffer registry (implemented).
- ✅ wgpu backend (implemented): upload/download, elementwise ops, transpose, matmul, reductions, signal processing.
- ✅ Operator fusion (implemented): elementwise chains and reductions are automatically fused.
- ✅ Planner cost model and auto-offload (implemented): automatic CPU/GPU routing based on tensor sizes and operation types.
- Planned: Streams/queues, memory pools, pinned/unified buffers, and multi-device support.
- Planned: Additional backends (CUDA/ROCm/OpenCL) for specialized use cases.

### Example usage
The provider is registered at process startup (REPL/CLI/app). Once registered, MATLAB-like code can use:
```matlab
G = gpuArray(A);      % move tensor to device
H = G + 2;            % elementwise add (planner may choose GPU path)
R = gather(H);        % bring results back to host
```

### Native acceleration

RunMat Accelerate powers native acceleration for RunMat. It lets users avoid needing to use the gpuArray/gather builtins and instead use the native acceleration API:

```matlab
% Example: Large matrix multiplication and elementwise operations
A = randn(10000, 10000);   % Large matrix
B = randn(10000, 10000);

% Normally, in MATLAB you'd need to explicitly use gpuArray:
%   G = gpuArray(A);
%   H = G .* B;           % Elementwise multiply on GPU
%   S = sum(H, 2);
%   R = gather(S);

% With RunMat Accelerate, the planner can transparently move data to the GPU
% and back as needed, so you can just write:
H = A .* B;           % Planner may choose GPU for large ops
S = sum(H, 2);        % Fused and executed on device if beneficial

% Results are automatically brought back to host as needed
disp(S(1:10));        % Print first 10 results

% The planner/JIT will optimize transfers and fuse operations for best performance.

```

### Device info (gpuDevice)
- `gpuDevice()` returns a structured value with details about the active provider/device when available. Fields include `device_id`, `name`, `vendor`, optional `memory_bytes`, and optional `backend`.

```matlab
info = gpuDevice();
% Example output (in-process provider):
%   struct with fields:
%       device_id: 0
%            name: 'InProcess'
%          vendor: 'RunMat'
%      backend: 'inprocess'
```

### Reduction tuning and defaults
- Defaults used by the WGPU provider (subject to device variation):
  - two-pass threshold: 1024 elements per slice
  - reduction workgroup size: 256
- Environment overrides:
  - `RUNMAT_TWO_PASS_THRESHOLD=<usize>`
  - `RUNMAT_REDUCTION_WG=<u32>`
- Selection logic:
  - Single-pass when `reduce_len <= threshold`, otherwise two-pass (partials + second-stage reduce).
  - Fusion and builtin paths honor provider defaults automatically; call sites may pass `wg=0` to use provider defaults.
- Sweep to tune for your device with the `wgpu_profile` binary:
  - Quick sweep: `cargo run -p runmat-accelerate --bin wgpu_profile --features wgpu -- --only-reduce-sweep --reduce-sweep --quick`
  - Force single-pass: `RUNMAT_TWO_PASS_THRESHOLD=1000000 cargo run -p runmat-accelerate --bin wgpu_profile --features wgpu -- --only-reduce-sweep --reduce-sweep`
  - Force two-pass: `RUNMAT_TWO_PASS_THRESHOLD=1 cargo run -p runmat-accelerate --bin wgpu_profile --features wgpu -- --only-reduce-sweep --reduce-sweep`

### Pipeline cache and warmup
- The provider caches compute pipelines by shader hash and layout to reduce first-dispatch latency.
- A brief warmup compiles common kernels during provider initialization.
- Cache metrics (hits/misses) are accessible via `runmat_accelerate::provider_cache_stats()` (when built with `wgpu`) and are printed by `wgpu_profile` after a sweep.
- On-disk persistence: cache metadata (`.json`) and shader WGSL (`.wgsl`) are persisted under the OS cache directory or `RUNMAT_PIPELINE_CACHE_DIR` (fallback `target/tmp`).
- Warmup precompiles pipelines from disk when available; `runmat accel-info` prints the last warmup duration in milliseconds.
