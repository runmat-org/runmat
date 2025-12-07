# Plotting / WASM Work Tracker — Context Snapshot

This scratch file preserves the current mental model for the RunMat plotting/WASM effort so we can reload it later without rereading the entire repo history. It summarizes what we just accomplished (new plotting modules + figure/hold/subplot state), the rationale for upcoming work, and the exact TODO tree for the remaining tasks.

---

## Current State (as of this snapshot)
- All legacy `src/plotting.rs` shims are gone; each builtin (`plot`, `scatter`, `bar`, `hist`, `surf`, `mesh`, `scatter3`, `stairs`, `contour`, `contourf`) now lives under `crates/runmat-runtime/src/builtins/plotting/` with figure-aware logic. They build runmat-plot primitives and call `render_active_plot`, which uses the new `state.rs` registry to honor MATLAB’s `figure`, `hold`, and `subplot` semantics.
- `runmat-plot` exposes `Scatter3Plot` (renamed from PointCloud), per-axes plot assignment (`add_*_on_axes`), and setter APIs (`set_axis_labels`, `set_subplot_grid`, etc.) so the runtime can manage subplots and handles.
- Shared Accelerate/plotting WGPU contexts exist and `scatter3`, `scatter`, `plot`, `surf`, `mesh`, `bar`, and now `hist` stay GPU-resident when fed `single` gpuArray inputs. The remaining plotting builtins (mesh variants that emit contours, etc.) still gather tensors to host slices.
- `plot` now handles MATLAB-style strings/name-value pairs (`'LineWidth'`, `'Color'`, `'LineStyle'`, `'Marker*'`) and multiple X/Y series so long as every series appears before the shared style block. Inline style tokens between series (e.g., `plot(x1,y1,'r',x2,y2,'--')`) are parsed per-series, and `'LineStyleOrder'` accepts string scalars, string arrays, or cell arrays of strings to seed the default line-style cycle. The figure registry stores the current `LineStyleOrder` per axes and cycles it across successive plot calls (reset on `hold off`), mirroring MATLAB defaults. Marker usage logs a warning and forces CPU rendering until the GPU path grows marker support. Other plotting builtins still accept only their minimal argument forms.

This means our next work streams are:
1. Zero-copy renderer integration (shared WGPU context, GPU buffer refs, renderer perf work).
2. Full MATLAB argument coverage for every plotting builtin.
3. Exposing figure state to wasm/TS bindings so the UI (Tauri/web) can show multiple figures/tabs.

---

## 1. Zero-Copy Renderer Integration
Goal: ensure plotting uses the same WGPU device/queue as RunMat Accelerate so data stays on the GPU, and update runmat-plot to consume GPU buffers directly.

- [x] **1.1 Accelerate context export**
  - `runmat_accelerate_api` now defines `AccelContextKind` and `AccelContextHandle::Wgpu` (with `Arc<wgpu::Device/Queue/Adapter>` + limits/features + adapter info).
  - `runmat-accelerate`’s WGPU provider stores the adapter/device/queue in `Arc`s and implements `export_context(AccelContextKind::Plotting)` so downstream callers can clone the shared context.
  - `runmat-runtime` gates the plotting module behind `plot-core` when default features are disabled, keeping `runmat-accelerate` builds (which compile runtime `default-features = false`) working.

- [x] **1.2 runmat-plot external context support**
  - `runmat_plot::web::WebRenderer` and the native `PlotWindow` path now both accept a `SharedWgpuContext` that carries the originating `wgpu::Instance`. When the runtime installs Accelerate’s exported handle, browser canvases and desktop windows reuse the exact adapter/device/queue trio instead of requesting a second adapter.
  - Exported contexts now include the originating `wgpu::Instance`, so surfaces created by runmat-plot are guaranteed to belong to the same adapter. Image export continues to fall back to a private adapter only when no shared context exists.

- [ ] **1.3 Runtime + wasm plumbing**
  - During `RunMatSession` initialization (native), request the plotting context from Accelerate and store it in the registry before any builtin renders.
  - In `runmat-wasm`, hook `initRunMat` + `registerPlotCanvas` to reuse the shared device/queue instead of creating a separate WebGPU instance.
  - Provide error messages if a plotting context is requested without an active GPU provider (fallback to CPU path).
  - _Current state_: `builtins::plotting::context` now installs the exported handle and mirrors it into `runmat_plot::context`, so every consumer (native GUI, exporters, wasm) can access the shared `wgpu::Device`. Both `RunMatSession::from_snapshot` and the wasm init path eagerly call `ensure_context_from_provider`, which in turn seeds the runmat-plot registry. Web canvases read from that registry first and only fall back to querying the provider if absolutely necessary. ✅ Native GUI windows and the image exporter now reuse the shared device/queue whenever one is available; they only request a private adapter when running CPU-only. Remaining work under 1.4 can assume both desktop + wasm renderers sit on the provider device.

- [ ] **1.4 GPU buffer references**
  - Introduce a `PlotBufferRef` (enum: HostSlice, GpuHandle) accessible from tensors/fusion outputs.
  - Update plot builders (`LinePlot`, `ScatterPlot`, `Scatter3Plot`, `SurfacePlot`, etc.) to accept these refs and create vertex/index buffers without host copies.
  - Extend runmat-plot renderer pipelines to bind existing GPU buffers (via `wgpu::Buffer::slice`), handling offsets/stride safely.
- _Scaffolding in place_: `runmat_accelerate_api` now exposes `export_wgpu_buffer(handle)` and the WGPU provider returns an `Arc<wgpu::Buffer>` + shape metadata. `Scatter3`, `scatter`, `plot`, `surf`, `mesh`, `surfc`, `meshc`, `bar`, `hist`, `stairs`, `contour`, and `contourf` all consume those handles: compute shaders in `runmat-plot::gpu::{scatter3,scatter2,line,surface,contour,contour_fill,bar,histogram,stairs}` pack the tensors into renderer `Vertex` layouts, bounding boxes come from the runtime `min`/`max` reductions, and the builtins only fall back to host copies when precision ≠ f32 or the shared device is missing. The shaders share the same runtime-provided workgroup size so they stay in sync with Accelerate’s tuning. Scatter/scatter3 now stream per-point sizes/colors via shared attribute buffers, so the next zero-copy targets are richer 2-D line styles (dashed/thick with hardware instancing) plus GPU-side marker shape variants.
- _New_: `mesh` reuses the surface packer and now keeps GPU triangles for wireframes, `hist` leverages the compute shaders in `runmat-plot/src/gpu/shaders/histogram.rs` to bin provider-resident samples before handing them to the shared bar packer, `stairs` streams gpuArray inputs directly through a dedicated compute packer (`runmat-plot/src/gpu/stairs.rs`), and `surfc`/`meshc` now ride the marching-squares contour kernel (`runmat-plot/src/gpu/contour.rs`). Every plotting compute shader has `f32`/`f64` variants embedded directly in `runmat-plot/src/gpu/shaders`, so adapters with `SHADER_F64` consume double-precision `gpuArray` buffers in place (no staging copies) and adapters without it continue to fall back to host gathers. `docs/wasm/plotting-zero-copy.md` and this tracker capture which builtins remain CPU-only.

- [ ] **1.5 Renderer performance passes**
  - Scatter/Scatter3: add instanced draws, optional compute-based frustum culling, level-of-detail toggles.
  - Surface/Mesh: share acceleration staging buffers; consider compute kernels for shading prep to avoid CPU remeshing.
  - Add stress tests/benchmarks (10M+ points, large surfaces) verifying zero-copy path meets perf goals on both native + wasm.

---

## 2. Builtin Argument Parity & Parser Work
Goal: every plotting builtin should accept the full MATLAB argument surface (style strings, name-value pairs, per-series options) while supporting the zero-copy buffer path.

### 2.1 Shared parsing utilities
- [ ] Color specs: char shortcuts (`'r'`), RGB triples, short strings (e.g., `"magenta"`), per-point color arrays, `'flat'`.
- [ ] Marker specs & style strings (e.g., `'--or'`), line widths, marker sizes, `'filled'` flag.
- [ ] Name-value parser covering both char arrays and string scalars, respecting MATLAB argument order/rules.
- [ ] Validate scalar vs. vector inputs, broadcast sizes, and error messages matching MATLAB IDs.

_Status_: the shared parser infrastructure lives in `plot::style` and already supports global colours, line widths, and marker options, but only `plot` consumes it so far. We still need to hoist it into a reusable module and add the richer color/marker forms listed above.
_Audit 2024-05-05_: `LineStyleParseOptions` currently forbids both leading and interleaved numeric arguments, which blocks scatter/stairs from mixing data + style strings. Parser also lacks per-point color vectors (`'flat'`), `'filled'` marker semantics, and RGB triplets coming from `gpuArray` tensors. We need to relax those guards when used by point-based builtins and expand parsing helpers accordingly.

### 2.2 Builtin-specific tasks
- [ ] **`plot`**
  - [x] Multiple series `plot(x1,y1,x2,y2,...)` (all series must appear before the shared style block; interleaved per-series styles still TODO).
  - [x] Style string combinations, `'LineWidth'`, `'Marker*'`, `'Color'` (markers parsed but currently force CPU fallback).
  - [x] `'LineStyleOrder'` parsing from string scalars/arrays/cell arrays with per-call cycling for series that keep the default style (axes-level persistence still TODO).
  - GPU path: ensure each series can reuse existing device buffers without gather.

- [ ] **`scatter` / `scatter3`**
  - Arguments `[X,Y,S,C]`, `'filled'`, per-point sizes/colors, colormap scaling.
  - Accept size/ color matrices (match MATLAB rules).
  - Renderer support for per-point attributes when using GPU buffers.
- _Status 2024-05-05_: `scatter` and `scatter3` both parse positional + name/value args via the shared parser (size/color vectors, `'filled'`, `'Marker*'`) and now pass per-point metadata straight into the GPU packers. Scatter2/Scatter3 shaders consume optional size + RGBA buffers, and the runtime falls back to host gathers only when the provider can’t export a buffer or the dtype isn’t `single`. CPU paths hydrate host vectors lazily so gpuArray callers still work when plotting is forced off-GPU. Remaining work: broaden parser coverage (`'flat'`, mixed series ordering) and wire marker-shape metadata so dashed/marker-heavy combinations avoid CPU fallbacks entirely.

- [ ] **`bar`**
  - Grouped/stacked inputs (matrix data), `'stacked'` option, category labels, width controls, color per series.
  - Handle negative values + baseline options.

- [ ] **`hist` / `histogram`**
  - Bin edges, normalization (`'pdf'`, `'probability'`, etc.), weights, output counts.
  - Possibly add modern `histogram` builtin if MATLAB expects it.
  - ✅ GPU histogram path using compute shaders is in place; bin centers + normalization literals/name-value pairs and `'BinEdges'` now parse on both CPU and GPU (GPU path still limited to uniform bins/count mode). Follow-up work: histogram-style weighting + exposing the richer `histogram` builtin API.

- [ ] **`surf` / `mesh` (and eventually `surfc`, `meshc`)**
  - Support matrix X/Y inputs, color matrices (`C`), shading options (`flat`, `interp`), lighting toggles, alpha data.
  - Accept name-value args for colormap, lighting, `'EdgeColor'`, `'FaceAlpha'`.

- [ ] **Docs/tests**
  - For each builtin, expand `DOC_MD` with new argument coverage, add unit tests for parser edge cases, and integration tests mirroring MATLAB scripts.
  - _Status 2024-05-05_: `stairs` accepts shared parser output (line width, colors) and reflects styles into both CPU + zero-copy GPU packers. Missing pieces: comprehensive doc/tests and marker semantics (MATLAB exposes markers on `stairs`, which we still ignore).

---

## 3. Multi-Figure UX & Bindings
Goal: expose figure handles/axes info to wasm/TS so the Tauri/web UI can display multiple figures (tabs or simultaneous canvases) without guessing.

- [ ] **3.1 Runtime surfacing**
  - Decide on the API surface for reporting current figure handle (return value from plotting builtins? event hook? dedicated `gcf` builtin?).
  - Add helpers for `gcf`, `gca`, `close`, `clf` if needed.
  - Ensure regressions (e.g., `figure(2); plot(...)`) update the registry + return consistent status messages.

- [ ] **3.2 runmat-wasm / TS bindings**
  - Expose JS/TS wrappers for `figure(handle?)`, `hold(mode?)`, `subplot(m,n,p)`, `close(handle?)`.
  - Emit events or callback hooks such as `onFigureUpdated({ handle, axesIndex, plotType })`.
  - Provide helpers for managing canvases/tabs: e.g., `registerFigureCanvas(handle, canvasElement)` or guidance for sharing one canvas with tabs.

- [ ] **3.3 UI contract + docs**
  - Document recommended UX patterns (tabs vs. multiple canvases) for the closed-source Tauri app; keep runmat repo references generic.
  - Update `docs/wasm/plan.md` and `docs/filesystem-notes.md` with new plotting considerations (figure handles, GPU requirements).
  - Provide example code in `bindings/ts/README.md` showing how to work with multiple figures.

---

## Working Notes
- Keep this file updated whenever a checklist item is started/completed; it’s our single source of truth when context gets compressed.
- When an item spans multiple commits (e.g., argument parsing + renderer support), jot interim notes here so we remember what’s blocked.
- Delete this file once the WASM plotting initiative is fully wrapped. Until then, it should contain enough detail to rehydrate the full state of the project quickly.
