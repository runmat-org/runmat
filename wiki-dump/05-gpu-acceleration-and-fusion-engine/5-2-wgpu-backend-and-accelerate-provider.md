---
title: "wgpu Backend & Accelerate Provider"
repo: "runmat-org/runmat"
branch: "dev"
source_url: "https://app.devin.ai/org/runmat-org/wiki/runmat-org/runmat?branch=dev#5.2"
wiki_hash: "#5.2"
section: "5.2"
category: "GPU Acceleration & Fusion Engine"
category_hash: "#5"
page_order: 20
last_updated: "May 28, 2026, 9:18:58 PM"
diagram_count: 0
source_files:
  - label: "crates/runmat-accelerate-api/src/lib.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate-api/src/lib.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/dispatch/mod.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/dispatch/mod.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/params.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/params.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/pipelines.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/pipelines.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/backend_types.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_types.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/core.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/core.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/helpers.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/helpers.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/init.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/init.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/constructors.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/constructors.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/elementwise.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/elementwise.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/image.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/image.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/indexing.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/indexing.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/random.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/random.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/signal.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/signal.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/solve.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/solve.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/tensor.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/tensor.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/provider/ops/window.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/window.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/shaders/creation.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/shaders/creation.rs"
  - label: "crates/runmat-accelerate/src/backend/wgpu/shaders/mod.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/shaders/mod.rs"
  - label: "crates/runmat-accelerate/src/simple_provider.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/simple_provider.rs"
  - label: "crates/runmat-runtime/src/builtins/builtins-json/exprnd.json"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/exprnd.json"
  - label: "crates/runmat-runtime/src/builtins/builtins-json/normrnd.json"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/normrnd.json"
  - label: "crates/runmat-runtime/src/builtins/builtins-json/unifrnd.json"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/unifrnd.json"
  - label: "crates/runmat-runtime/src/builtins/common/random.rs"
    url: "https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/common/random.rs"
  - label: "docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md"
    url: "https://github.com/runmat-org/runmat/blob/82685330/docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md?plain=1"
  - label: "docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md"
    url: "https://github.com/runmat-org/runmat/blob/82685330/docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md?plain=1"
headings:
  - level: 1
    text: "wgpu Backend & Accelerate Provider"
    id: "5.2-wgpu-backend-accelerate-provider"
  - level: 2
    text: "Architecture & Module Organization"
    id: "5.2-architecture-module-organization"
  - level: 3
    text: "System Entity Map"
    id: "5.2-system-entity-map"
  - level: 2
    text: "Pipeline Creation & Management"
    id: "5.2-pipeline-creation-management"
  - level: 2
    text: "Buffer Management & Residency"
    id: "5.2-buffer-management-residency"
  - level: 2
    text: "Operation Categories"
    id: "5.2-operation-categories"
  - level: 2
    text: "WGSL Shader Dispatch"
    id: "5.2-wgsl-shader-dispatch"
  - level: 2
    text: "SimpleProvider Abstraction"
    id: "5.2-simpleprovider-abstraction"
---
# wgpu Backend & Accelerate Provider

<details>
<summary>Relevant source files</summary>

- [crates/runmat-accelerate-api/src/lib.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate-api/src/lib.rs)
- [crates/runmat-accelerate/src/backend/wgpu/dispatch/mod.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/dispatch/mod.rs)
- [crates/runmat-accelerate/src/backend/wgpu/params.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/params.rs)
- [crates/runmat-accelerate/src/backend/wgpu/pipelines.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/pipelines.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/backend_types.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_types.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/core.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/core.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/helpers.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/helpers.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/init.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/init.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/constructors.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/constructors.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/elementwise.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/elementwise.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/image.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/image.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/indexing.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/indexing.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/random.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/random.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/signal.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/signal.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/solve.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/solve.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/tensor.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/tensor.rs)
- [crates/runmat-accelerate/src/backend/wgpu/provider/ops/window.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/window.rs)
- [crates/runmat-accelerate/src/backend/wgpu/shaders/creation.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/shaders/creation.rs)
- [crates/runmat-accelerate/src/backend/wgpu/shaders/mod.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/shaders/mod.rs)
- [crates/runmat-accelerate/src/simple_provider.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/simple_provider.rs)
- [crates/runmat-runtime/src/builtins/builtins-json/exprnd.json](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/exprnd.json)
- [crates/runmat-runtime/src/builtins/builtins-json/normrnd.json](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/normrnd.json)
- [crates/runmat-runtime/src/builtins/builtins-json/unifrnd.json](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/builtins-json/unifrnd.json)
- [crates/runmat-runtime/src/builtins/common/random.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-runtime/src/builtins/common/random.rs)
- [docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md](https://github.com/runmat-org/runmat/blob/82685330/docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md?plain=1)
- [docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md](https://github.com/runmat-org/runmat/blob/82685330/docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md?plain=1)

</details>

The `wgpu` backend serves as the primary hardware acceleration layer for RunMat, providing a high-performance implementation of the `AccelProvider` trait [crates/runmat-accelerate-api/src/lib.rs #103-104](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate-api/src/lib.rs#L103-L104) It leverages the WebGPU API (via the `wgpu` crate) to execute MATLAB-compatible operations across diverse hardware including Vulkan, Metal, DirectX, and WebGPU in the browser.

## Architecture & Module Organization

The backend is organized into a modular tree structure that separates lifecycle management, buffer residency, and operation-specific logic.

### System Entity Map

The following diagram bridges the high-level backend concepts to their specific implementation files and structs.

WGPU Backend Entity Mapping

Sources: [crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs #58-95](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs#L58-L95) [docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md #81-112](https://github.com/runmat-org/runmat/blob/82685330/docs-tmp/ACCELERATE_PROVIDER_REFACTOR.md?plain=1#L81-L112)

## Pipeline Creation & Management

Pipeline management is centralized in the `WgpuPipelines` struct [crates/runmat-accelerate/src/backend/wgpu/pipelines.rs](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/pipelines.rs) During initialization, the provider compiles WGSL shaders and creates `wgpu::ComputePipeline` objects for every supported operation.

1. Initialization: `WgpuProvider::new` invokes the initialization logic in `init.rs` [crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs #74-75](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs#L74-L75)
2. Shader Compilation: Shaders are modularized in the `shaders/` directory, with `logical.rs` and `elementwise.rs` providing common WGSL snippets [crates/runmat-accelerate/src/backend/wgpu/shaders/mod.rs #8-20](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/shaders/mod.rs#L8-L20)
3. Binding Limits: The backend validates hardware limits (e.g., `max_storage_buffers_per_shader_stage`) before attempting to create bind groups [crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs #7-32](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs#L7-L32)

Sources: [crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs #107-109](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs#L107-L109) [crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs #7-32](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs#L7-L32)

## Buffer Management & Residency

The `WgpuProvider` maintains a `BufferResidency` system to track GPU-resident tensors [crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs #108](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs#L108-L108)

- Storage Buffers: Created via `create_storage_buffer_checked`, which enforces per-buffer size limits reported by the adapter [crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs #39-49](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs#L39-L49)
- Uniform Buffers: Short-lived buffers for kernel parameters (e.g., `ScalarParamsF64`, `Conv1dParams`) managed by the `UniformBufferKey` system [crates/runmat-accelerate/src/backend/wgpu/params.rs #28-61](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/params.rs#L28-L61)
- Zero-Copy Export: Supports exporting internal `wgpu::Buffer` references for external consumers like the plotting system [crates/runmat-accelerate-api/src/lib.rs #110-112](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate-api/src/lib.rs#L110-L112)

Sources: [crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs #39-49](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend_shared.rs#L39-L49) [crates/runmat-accelerate/src/backend/wgpu/params.rs #1-61](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/params.rs#L1-L61)

## Operation Categories

The backend implements the MATLAB standard library through specialized `_exec` methods.

| Category | Key Implementation File | Description |
| --- | --- | --- |
| Elementwise | ops/elementwise.rs | Unary, binary (with broadcasting), and logical operations docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md#20-33 |
| Reduction | ops/reduction/mod.rs | Sum, mean, std, var, and moments. Supports global and dimension-wise reduction docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md#38-40 |
| Linalg | ops/linalg.rs | Matrix multiplication (mtimes), transpose, and decompositions crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs#78-79 |
| Signal | ops/signal.rs | 1D convolution, IIR filtering, and cumulative operations (cumsum, cumprod) crates/runmat-accelerate/src/backend/wgpu/provider/ops/signal.rs#95-97 |
| Image | ops/image.rs | imfilter with specialized kernel handling and padding modes crates/runmat-accelerate/src/backend/wgpu/provider/ops/image.rs#20-25 |
| Random | ops/random.rs | Philox-based RNG for rand, randn, exprnd, and normrnd crates/runmat-accelerate/src/backend/wgpu/provider/ops/random.rs#125-130 |
| Polynomial | ops/polynomial.rs | polyval, polyder, and polyfit (via host fallback if needed) crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs#18-53 |

Sources: [docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md #20-97](https://github.com/runmat-org/runmat/blob/82685330/docs-tmp/ACCELERATE_PROVIDER_REFACTOR_PROGRESS.md?plain=1#L20-L97) [crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs #58-95](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/backend.rs#L58-L95)

## WGSL Shader Dispatch

RunMat uses a standardized dispatch pattern to ensure optimal GPU utilization across different architectures.

Data Flow: Host to GPU Dispatch

wgpu::Queuewgpu::Devicedispatch/elementwise.rsops/elementwise.rsWgpuProviderwgpu::Queuewgpu::Devicedispatch/elementwise.rsops/elementwise.rsWgpuProviderbinary_op_exec(A | B | op)get_entry(A), get_entry(B)uniform_buffer(ScalarParams)create_bind_group()run(device | queue | pipeline | bind_group | workgroups)submit(command_encoder)

Sources: [crates/runmat-accelerate/src/backend/wgpu/provider/ops/elementwise.rs #26-28](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/elementwise.rs#L26-L28) [crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs #174-180](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/backend/wgpu/provider/ops/polynomial.rs#L174-L180)

## SimpleProvider Abstraction

For environments where WebGPU is unavailable or for operations not yet ported to GPU kernels, the `SimpleProvider` acts as a host-side reference implementation [crates/runmat-accelerate/src/simple_provider.rs #6-17](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/simple_provider.rs#L6-L17)

- Host Fallbacks: Many complex linear algebra operations (e.g., `inv`, `rank`, `linsolve`) default to `SimpleProvider` which wraps optimized CPU BLAS/LAPACK routines [crates/runmat-accelerate/src/simple_provider.rs #33-40](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/simple_provider.rs#L33-L40)
- Shared Logic: Both `WgpuProvider` and `SimpleProvider` share logic for shape broadcasting and index computation via `runmat-runtime` [crates/runmat-accelerate/src/simple_provider.rs #20-23](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/simple_provider.rs#L20-L23)

Sources: [crates/runmat-accelerate/src/simple_provider.rs #1-46](https://github.com/runmat-org/runmat/blob/82685330/crates/runmat-accelerate/src/simple_provider.rs#L1-L46)
