## runmat-accelerate

### Purpose
`runmat-accelerate` provides the high-level acceleration layer that integrates GPU backends with the language runtime. It implements provider(s) for `runmat-accelerate-api` so that `gpuArray`, `gather`, and (later) accelerated math and linear algebra can execute on devices transparently where appropriate.

### Architecture
- Depends on `runmat-accelerate-api` to register an `AccelProvider` implementation at startup.
- Backends (e.g., `wgpu`, `cuda`, `rocm`, `metal`, `vulkan`, `opencl`) are feature-gated. Only one provider is registered globally, but a future multi-device planner can fan out.
- `Planner` decides when to run ops on CPU vs GPU (size thresholds, op types, fusion opportunities). `Accelerator` exposes ergonomic entry points used by the runtime or higher layers.

### What it provides today
- A scaffolding `Accelerator` with elementwise add routing: choose CPU path (delegating to `runmat-runtime`) or GPU path (via provider methods). The GPU path currently uses upload/compute/download placeholders and is ready to be backed by a real backend.
- Integration points for `gpuArray`/`gather`: when a provider is registered, runtime builtins route through the provider API defined in `runmat-accelerate-api`.

### How it fits with the runtime
- The MATLAB-facing builtins (`gpuArray`, `gather`) live in `runmat-runtime` for consistency with all other builtins. They call into `runmat-accelerate-api::provider()`, which is implemented and registered by this crate.
- This separation avoids dependency cycles and keeps the language surface centralized while enabling pluggable backends.

### Backends
- `wgpu` (feature: `wgpu`) is the first cross-vendor target. CUDA/ROCm/Metal/Vulkan/OpenCL are planned (features already stubbed).
- Backend responsibilities:
  - Allocate/free buffers, handle host↔device transfers
  - Provide kernels for core ops (elementwise, transpose, matmul/GEMM)
  - Report device information (for planner decisions)

### Current state
- Compiles and wires through to the runtime via the API layer.
- CPU fallback path fully functional; GPU path ready for provider implementation.

### Roadmap
- Implement an in-process provider with a buffer registry (proof-of-concept) to make `gpuArray`/`gather` round-trip actual data without copying through a real device yet.
- Implement first real backend (likely `wgpu`): upload/download, elementwise add/mul/div/pow, transpose, matmul, with simple planner thresholds.
- Add streams/queues, memory pools, pinned/unified buffers, and multi-device support.
- Planner cost model and operator fusion (elementwise chains and simple BLAS fusions).

### Example usage
The provider is registered at process startup (REPL/CLI/app). Once registered, MATLAB-like code can use:
```matlab
G = gpuArray(A);      % move tensor to device
H = G + 2;            % elementwise add (planner may choose GPU path)
R = gather(H);        % bring results back to host
```


