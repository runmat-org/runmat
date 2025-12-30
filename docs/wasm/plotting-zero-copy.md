# RunMat Web/WebGPU Plotting Architecture

## Goals
- Share a single `wgpu::Device`/`Queue` between RunMat Accelerate (compute/fusion) and runmat-plot (render) so tensors never bounce through host RAM.
- Stream figures into both native (`gui`) and browser (`plot-web`) frontends with identical semantics.
- Keep rendering scalable for “video-game scale” plots (multi-billion/trillion point clouds, dense lidar/radar).
- Provide a portable contract JS/Tauri shells can consume without forking the renderer.

## Current State
- Shared `wgpu::Device/Queue` plumbing is live: Accelerate exports the context and `runmat-plot` mirrors it so native + WASM now drive the same hardware.
- `scatter3`, `scatter`, `plot`, `surf`, `mesh`, `surfc`, `meshc`, `bar`, `hist`, and `stairs` detect single-precision `gpuArray` inputs, export the provider buffers, and feed them into the compute shaders housed in `crates/runmat-plot/src/gpu/shaders` so we emit renderer-ready vertex streams without touching host RAM.
- Double-precision gpuArrays now flow through the plotting stack without allocating mirror buffers: each compute kernel has dedicated `f32`/`f64` variants baked into `crates/runmat-plot/src/gpu/shaders`, so adapters with `SHADER_F64` read provider buffers directly while adapters without it continue to gather on the host.
- The histogram builtin now understands MATLAB-style `'BinEdges'` and `'Normalization'` name-value pairs; uniform edges remain zero-copy on the GPU path, while non-uniform edges automatically fall back to the CPU implementation.
- `plot` now handles multiple X/Y series per call (as long as every series appears before the shared style block) and parses MATLAB marker options (`'Marker'`, `'MarkerSize'`, `'MarkerEdgeColor'`, `'MarkerFaceColor'`). Marker usage intentionally forces the CPU renderer until the GPU marker shaders gain feature parity.
- GPU packers still assume uniform color + size and rely on MATLAB defaults; richer per-point colors/markers remain on the roadmap alongside the renderer performance passes.

## Proposed Architecture
### 1. Expose the Accelerate WGPU context
- Extend `runmat_accelerate_api` with a `WgpuContextHandle` (Arc<`wgpu::Device`>, Arc<`wgpu::Queue`>, adapter info, limits).
- Teach `runmat-accelerate`’s WGPU provider to vend this handle via `AccelerateProvider::export_context(AccelContextKind::Plotting)`.
- WASM: the same context will already be backed by the browser WebGPU device; we simply pass the handle through.

### 2. runmat-plot “external device” API
- Add `PlotRenderer::from_context(context: RunMatPlotContext, surface: SurfaceTarget)` so the renderer uses the shared device/queue.
- Remove device creation logic from `WebRenderer::new` / GUI window bootstrap; instead take a `RunMatPlotContext`.
- Keep a fallback path that still owns its own device for CLI builds without Accelerate (for tests).

### 3. Zero-copy tensor transfer
- Extend `runmat_builtins::Tensor` with an optional `GpuTensorHandle` accessor (already present internally) and teach plotting helpers to detect GPU residency.
- Introduce `PlotBufferRef` that wraps either a host slice (for CPU tensors) or a `GpuTensorHandle`.
- In `runmat-plot`, accept `PlotBufferRef` when building `RenderData`. For GPU refs, create `wgpu::BufferSlice` views using `wgpu::Buffer::as_entire_buffer_binding` without copying.
- For large clouds/surfaces, emit indirect draw commands / compute-stage packing kernels (shared WGPU device allows this).

### 4. Shared frame scheduling
- Add a lightweight “plot pass” to the Accelerate runtime scheduler: compute kernels finish, then the renderer records a `wgpu::CommandEncoder` using the same queue.
- In WASM, expose `requestAnimationFrame` driven updates that borrow the shared queue via async mutex to avoid overlapping submissions.

### 5. JS/Tauri bindings
- `registerPlotCanvas` now provides an initialized `WebRendererContext` (canvas surface config only). The actual device comes from the Rust side.
- Tauri/native shells ask the runtime for `PlotSurfaceHandle` (swap chain info) and feed window events back across IPC; the rendering still happens in Rust so throughput stays native.

## Testing Strategy
- Add `runmat-plot` stress tests that render 10–100M point clouds using synthetic `GpuTensorHandle` mocks to ensure buffer slicing works.
- Runtime unit tests: instrument plotting builtins to ensure GPU tensors stay resident (no `gather` calls) when a provider is active.
- WASM integration test: spin up `wasm-bindgen-test` harness, register a fake canvas, ensure `scatter3` writes to the shared queue.

## Rollout Steps
1. Finish GPU packers for the remaining high-volume builtins (scatter/plot/line family, `surf`, `mesh`, histograms). Each should follow the scatter2/3 pattern: detect resident tensors, run a compute kernel to pack vertices, and measure bounds via the shared context.
2. Extend marker-size/colour parsing so both CPU and GPU paths honor MATLAB’s full argument surface (per-point sizes/colors, name-value pairs, styles). GPU kernels will need structured inputs or staging buffers for those attributes.
3. Delete the legacy `src/plotting.rs` shim once every builtin has a proper module under `builtins::plotting`.
4. Keep docs (`docs/wasm/plan.md`, `TEMP_WORK_CONTEXT.md`, `docs/ARCHITECTURE.md`) updated as each builtin lands zero-copy support to preserve a clear WASM roadmap.

This keeps the plotting stack aligned with Fusion/Accelerate goals and ensures browser + native targets share one high-performance path.
