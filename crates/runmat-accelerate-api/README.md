## runmat-accelerate-api

### Purpose
`runmat-accelerate-api` defines the minimal, dependency-light interface between the language runtime and GPU acceleration providers. It exists so that:

- The runtime (`runmat-runtime`) can expose user-facing builtins like `gpuArray` and `gather` without taking a hard dependency on any particular backend implementation.
- Acceleration crates (e.g., `runmat-accelerate`) can implement providers and register themselves at process startup, enabling device-resident execution when available and falling back to CPU when not.

This crate intentionally avoids depending on other workspace crates to prevent dependency cycles.

### What it contains
- **GpuTensorHandle**: A small, opaque handle describing a device-resident tensor (shape, device id, and a backend-managed buffer id). The actual device memory is fully managed by the backend.
- **AccelProvider**: The provider trait backends implement to upload/download/free GPU buffers and to report device info.
- **Global provider registry**: `register_provider` and `provider()` allow a single global provider to be installed by the host (e.g., `runmat-accelerate` or an application) and used by the runtime.
- **Host tensor types**: `HostTensorView` and `HostTensorOwned` provide simple host-side tensor representations (data + shape) without introducing a dependency on `runmat-builtins`.

### How it fits with the rest of the system
- `runmat-runtime` implements the MATLAB-facing builtins `gpuArray(x)` and `gather(x)`. These builtins call `runmat-accelerate-api::provider()` and, if a provider is registered, forward to it. Otherwise, they return CPU fallbacks (keeping semantics predictable even when no GPU is present).
- `runmat-accelerate` implements one or more concrete providers and calls `register_provider(...)` during initialization (e.g., from the REPL/CLI startup or host app), so the runtime automatically benefits from GPU acceleration where appropriate.

### Safety and lifetime
`register_provider` stores a `'static` reference. The caller must ensure the provider outlives the program (common in singletons created at startup). The handle (`GpuTensorHandle`) contains only POD metadata (no GC pointers) and is safe to move/copy.

### API reference (concise)
- `pub struct GpuTensorHandle { pub shape: Vec<usize>, pub device_id: u32, pub buffer_id: u64 }`
- `pub trait AccelProvider { upload(&HostTensorView) -> Result<GpuTensorHandle>; download(&GpuTensorHandle) -> Result<HostTensorOwned>; free(&GpuTensorHandle) -> Result<()>; device_info() -> String }`
- `pub struct HostTensorView<'a> { pub data: &'a [f64], pub shape: &'a [usize] }`
- `pub struct HostTensorOwned { pub data: Vec<f64>, pub shape: Vec<usize> }`
- `pub unsafe fn register_provider(&'static dyn AccelProvider)`
- `pub fn provider() -> Option<&'static dyn AccelProvider>`

### Example (provider skeleton)
```rust
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorOwned, HostTensorView, register_provider};

struct MyProvider;

impl AccelProvider for MyProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        // allocate device buffer, copy host.data, return handle
        Ok(GpuTensorHandle { shape: host.shape.to_vec(), device_id: 0, buffer_id: 1 })
    }
    fn download(&self, h: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
        // copy device -> host and return
        Ok(HostTensorOwned { data: vec![0.0; h.shape.iter().product()], shape: h.shape.clone() })
    }
    fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> { Ok(()) }
    fn device_info(&self) -> String { "MyDevice 0".to_string() }
}

// at startup
unsafe { register_provider(&MyProvider); }
```

### Current state
- Stable trait and handle definitions.
- Used by `runmat-runtime` builtins (`gpuArray`, `gather`).
- Ready for concrete provider implementations in `runmat-accelerate`.

### Roadmap
- Add optional fields (dtype, strides) without breaking existing providers (via feature flags or additive fields).
- Extend provider API for streams/queues, memory pools, unified/pinned memory, and device events.
- Multi-device provider model and selection heuristics.


