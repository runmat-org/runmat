# Plotting / WASM Work Tracker — Context Snapshot

This scratch file preserves the current mental model for the RunMat plotting/WASM effort so we can reload it later without rereading the entire repo history. It summarizes what we just accomplished (new plotting modules + figure/hold/subplot state), the rationale for upcoming work, and the exact TODO tree for the remaining tasks.

---

## Current State (as of this snapshot)
- All legacy `src/plotting.rs` shims are gone; each builtin (`plot`, `scatter`, `bar`, `hist`, `surf`, `mesh`, `scatter3`, `stairs`, `contour`, `contourf`) now lives under `crates/runmat-runtime/src/builtins/plotting/` with figure-aware logic. They build runmat-plot primitives and call `render_active_plot`, which uses the new `state.rs` registry to honor MATLAB’s `figure`, `hold`, and `subplot` semantics.
- `runmat-plot` exposes `Scatter3Plot` (renamed from PointCloud), per-axes plot assignment (`add_*_on_axes`), and setter APIs (`set_axis_labels`, `set_subplot_grid`, etc.) so the runtime can manage subplots and handles.
- Shared Accelerate/plotting WGPU contexts exist and `scatter3`, `scatter`, `plot`, `surf`, `mesh`, `bar`, `hist`, `contour`, and `contourf` stay GPU-resident when fed `single` gpuArray inputs. The remaining legacy paths only gather tensors when adapters lack `f64` or plotting is forced to CPU.
- `plot` now handles MATLAB-style strings/name-value pairs (`'LineWidth'`, `'Color'`, `'LineStyle'`, `'Marker*'`) and multiple X/Y series so long as every series appears before the shared style block. Inline style tokens between series (e.g., `plot(x1,y1,'r',x2,y2,'--')`) are parsed per-series, and `'LineStyleOrder'` accepts string scalars, string arrays, or cell arrays of strings to seed the default line-style cycle. The figure registry stores the current `LineStyleOrder` per axes and cycles it across successive plot calls (reset on `hold off`), mirroring MATLAB defaults. Marker specs now ride a zero-copy path: the runtime exports GPU marker metadata, runmat-plot packs point vertices alongside the line buffers, and the renderer draws both primitives so hybrid line+marker plots stay GPU-resident (no more CPU fallback warnings). The GPU packer also keeps dashed/dash-dot/dotted styles and multi-pixel widths on-device by expanding each segment into line pairs or extruded triangle strips while tracking vertex counts atomically, so only unsupported precisions fall back to the CPU path. Other plotting builtins still accept only their minimal argument forms.
- Parser parity work has started to spread beyond `plot`: `bar`, `hist`, `contour`, and `contourf` now accept MATLAB-style color/width/name-value pairs (`'FaceColor'`, `'EdgeColor'`, `'FaceAlpha'`, `'BarWidth'`, `'DisplayName'`, `'LevelList'`, `'LevelListMode'`, `'LevelStep'`, `'LineColor'`, etc.) while keeping gpuArray inputs zero-copy. Shared helpers (`parse_bar_style_args`, `parse_contour_style_args`) live next to the line-style parser so new builtins can adopt the same syntax surface without rolling bespoke logic. Contour docs/tests now cover the `'LevelList'`/`'LevelStep'`/`'LineColor'` parsing paths so regressions are visible.
- `hist` mirrors MATLAB’s multi-output semantics and advanced bin controls: `[N, X] = hist(...)` now works via a dedicated VM branch, the runtime stores counts/centers in `HistEvaluation`, and the parser accepts `NumBins`, `BinWidth`, `BinLimits`, and `'sqrt'`/`'sturges'`/`'integers'` `BinMethod` tokens on top of `'Weights'`/`'Normalization'`. GPU execution stays zero-copy whenever those options produce uniform bins (including integer bins); non-uniform descriptions fall back to the CPU path automatically, and the new unit tests cover the added permutations.
- LOD/perf tuning now covers scatter2/scatter3 and surfaces. Scatter kernels honor an env/runtime-configurable target (`RUNMAT_PLOT_SCATTER_TARGET` or `set_scatter_target_points`) and surfaces coalesce via `RUNMAT_PLOT_SURFACE_VERTEX_BUDGET`, both scaled by the plot’s extent so zoomed-in views retain more detail. Stress harnesses live in `runmat-plot` (skipping by default unless `RUNMAT_PLOT_FORCE_GPU_TESTS=1`) to sanity-check million-point clouds without forcing GPU CI. Figure lifecycle events (`FigureEvent::Created/Updated/Closed`) now flow through the wasm bridge so TypeScript consumers can subscribe via `onFigureEvent`, mount canvases per handle, and (new) drive figure/hold/subplot state from JS via the exported `figure()`, `setHoldMode()`, and `configureSubplot()` helpers.
- Figure lifecycle bindings are rounded out: the runtime exposes `clear_figure`, `close_figure`, and `current_axes_state`, wasm exports wrap them with structured error payloads (`InvalidHandle`, `InvalidSubplotGrid`, `InvalidSubplotIndex`), and the TypeScript wrapper now ships `clearFigure`, `closeFigure`, `currentAxesInfo`, and label-aware `FigureEvent` objects (`kind = created/updated/cleared/closed`). Hosts no longer have to parse MATLAB stdout to learn that a subplot index was invalid; they can inspect `error.code`/`error.rows`/`error.cols` straight from the promise rejection.
- Figure close events now forcibly detach any registered wasm canvases (`runmat-runtime/src/builtins/plotting/core/web.rs`, `crates/runmat-wasm/src/lib.rs`), so `close(handle)` immediately releases GPU resources and signals host shells without waiting for manual deregistration.
- Desktop GUI builds now install the same lifecycle observer: `runmat-runtime` notifies `runmat-plot` when a figure closes, the single-window manager keeps a per-handle atomic close flag, and `PlotWindow` checks that flag each frame so MATLAB `close(handle)` tears down the corresponding winit window immediately instead of leaving a stale canvas on screen.
- The native window manager and GUI thread manager now participate in the shared close-signal path (`gui/native_window.rs`, `gui/thread_manager.rs`). The lifecycle module automatically routes plotting through native windows first, then the GUI thread manager, then the single-window fallback (unless `RUNMAT_PLOT_DESKTOP_BACKEND` overrides the order). Every backend receives the shared close signal, so multi-window shells can keep multiple canvases alive concurrently while still tearing them down instantly when MATLAB code issues `close(handle)`.
- Parser parity resumed: `stairs` reuses the shared line-style parser end-to-end (marker metadata now feeds both CPU and GPU packers, so dashed/marker-heavy stairs no longer drop to CPU), and `bar`/`hist` accept MATLAB’s `'FaceColor','flat'` syntax by generating per-bar color cycles on the CPU path while automatically skipping the GPU packers when flat colors are requested.

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

- [x] **1.3 Runtime + wasm plumbing**
  - During `RunMatSession` initialization (native), request the plotting context from Accelerate and store it in the registry before any builtin renders.
  - In `runmat-wasm`, hook `initRunMat` + `registerPlotCanvas` to reuse the shared device/queue instead of creating a separate WebGPU instance.
  - Provide error messages if a plotting context is requested without an active GPU provider (fallback to CPU path).
- _Current state_: `builtins::plotting::context` now installs the exported handle and mirrors it into `runmat_plot::context`, so every consumer (native GUI, exporters, wasm) can access the shared `wgpu::Device`. Both `RunMatSession::from_snapshot` and the wasm init path eagerly call `ensure_context_from_provider`, which in turn seeds the runmat-plot registry. Web canvases read from that registry first and only fall back to querying the provider if absolutely necessary. ✅ Native GUI windows and the image exporter now reuse the shared device/queue whenever one is available; they only request a private adapter when running CPU-only. CLI startup and the wasm `gpuStatus` surface now bubble `PlotContextError` so zero-copy issues are visible instead of failing silently. Remaining work under 1.4 can assume both desktop + wasm renderers sit on the provider device.

- [ ] **1.4 GPU buffer references**
  - Introduce a `PlotBufferRef` (enum: HostSlice, GpuHandle) accessible from tensors/fusion outputs.
  - Update plot builders (`LinePlot`, `ScatterPlot`, `Scatter3Plot`, `SurfacePlot`, etc.) to accept these refs and create vertex/index buffers without host copies.
  - Extend runmat-plot renderer pipelines to bind existing GPU buffers (via `wgpu::Buffer::slice`), handling offsets/stride safely.
- _Scaffolding in place_: `runmat_accelerate_api` now exposes `export_wgpu_buffer(handle)` and the WGPU provider returns an `Arc<wgpu::Buffer>` + shape metadata. `Scatter3`, `scatter`, `plot`, `surf`, `mesh`, `surfc`, `meshc`, `bar`, `hist`, `stairs`, `contour`, and `contourf` all consume those handles: compute shaders in `runmat-plot::gpu::{scatter3,scatter2,line,surface,contour,contour_fill,bar,histogram,stairs}` pack the tensors into renderer `Vertex` layouts, bounding boxes come from the runtime `min`/`max` reductions, and the builtins only fall back to host copies when precision ≠ f32 or the shared device is missing. The shaders share the same runtime-provided workgroup size so they stay in sync with Accelerate’s tuning. Scatter/scatter3 stream per-point sizes/colors via shared attribute buffers, and the line packer now emits dashed/dash-dot/dotted segments plus multi-pixel widths (triangles) while keeping marker metadata zero-copy. Remaining zero-copy work lives under 1.5 (renderer perf, LOD, and stress tooling).
- _New_: `mesh` reuses the surface packer and now keeps GPU triangles for wireframes, `hist` handles weighted bins entirely on the device (per-bin float atomics + probability/pdf scaling) before handing the counts to the shared bar packer, `stairs` streams gpuArray inputs directly through a dedicated compute packer (`runmat-plot/src/gpu/stairs.rs`), and `surfc`/`meshc` now ride the marching-squares contour kernel (`runmat-plot/src/gpu/contour.rs`). Every plotting compute shader has `f32`/`f64` variants embedded directly in `runmat-plot/src/gpu/shaders`, so adapters with `SHADER_F64` consume double-precision `gpuArray` buffers in place (no staging copies) and adapters without it continue to fall back to host gathers. `docs/wasm/plotting-zero-copy.md` and this tracker capture which builtins remain CPU-only.

- [ ] **1.5 Renderer performance passes**
  - Scatter/Scatter3: add instanced draws, optional compute-based frustum culling, level-of-detail toggles.
  - Surface/Mesh: share acceleration staging buffers; consider compute kernels for shading prep to avoid CPU remeshing.
  - Add stress tests/benchmarks (10M+ points, large surfaces) verifying zero-copy path meets perf goals on both native + wasm. _(New: `line_plot_handles_large_trace` exercises a 50k-point CPU polyline; still need GPU perf harnesses plus scatter/surface stress suites.)_
  - ✅ Scatter now honors env-configurable LOD budgets (`RUNMAT_PLOT_SCATTER_TARGET`) and the surface packer computes stride/lod pairs based on `RUNMAT_PLOT_SURFACE_VERTEX_BUDGET` (default 400k verts). Added GPU stress tests inside `runmat-plot/src/gpu/scatter2.rs`, `scatter3.rs`, and `surface.rs` that allocate multi-million-point clouds on a headless `wgpu` device (skips automatically unless `RUNMAT_PLOT_FORCE_GPU_TESTS=1`), so CI/devs can sanity-check the zero-copy path without relying on CPU fallbacks. Runtime-facing setters (`set_scatter_target_points`, `set_surface_vertex_budget`) and TS docs mean hosts can tune budgets without exporting env vars. Remaining TODO: camera-aware heuristics for figure-aware zooming, instancing/culling, and perf metrics harness that records frame time deltas rather than just vertex counts.

---

## 2. Builtin Argument Parity & Parser Work
Goal: every plotting builtin should accept the full MATLAB argument surface (style strings, name-value pairs, per-series options) while supporting the zero-copy buffer path.

### 2.1 Shared parsing utilities
- [ ] Color specs: char shortcuts (`'r'`), RGB triples, short strings (e.g., `"magenta"`), per-point color arrays, `'flat'`.
- [ ] Marker specs & style strings (e.g., `'--or'`), line widths, marker sizes, `'filled'` flag.
- [ ] Name-value parser covering both char arrays and string scalars, respecting MATLAB argument order/rules.
- [ ] Validate scalar vs. vector inputs, broadcast sizes, and error messages matching MATLAB IDs.

_Status_: the shared parser infrastructure lives in `plot::style` and now understands MATLAB's `'flat'` marker colors, RGB(A) gpuArray triples, and interleaved series/style tokens (we relaxed `LineStyleParseOptions::plot`). `scatter`/`scatter3` reuse the same utilities; remaining work is to expose the upgraded parser to `stairs`, `bar`, etc., and to cover `'flat'` line colors + name/value shorthand that still fall back today.
_Audit 2024-05-05_: `LineStyleParseOptions` previously forbade both leading and interleaved numeric arguments (blocking scatter/stairs from mixing data + style strings) and treated `'flat'` as an error. Both issues are fixed for point-style options, but we still need to audit other builtins before checking this box.

### 2.2 Builtin-specific tasks
- [ ] **`plot`**
  - [x] Multiple series `plot(x1,y1,x2,y2,...)` (all series must appear before the shared style block; interleaved per-series styles still TODO).
  - [x] Style string combinations, `'LineWidth'`, `'Marker*'`, `'Color'` (markers parsed but currently force CPU fallback).
  - [x] `'LineStyleOrder'` parsing from string scalars/arrays/cell arrays with per-call cycling for series that keep the default style (axes-level persistence still TODO).
  - [~] GPU path: ensure each series can reuse existing device buffers without gather.
  - _Status 2025-XX-YY_: GPU path stays zero-copy for solid/dashed/dash-dot/dotted lines, switches between line/triangle pipelines based on `LineWidth`, and renders markers entirely on-device. Remaining `plot` work is parser parity (per-series inline tokens, MATLAB name/value aliases) and `LineStyleOrder` persistence across hold/figure cycles.

- [ ] **`scatter` / `scatter3`**
  - Arguments `[X,Y,S,C]`, `'filled'`, per-point sizes/colors, colormap scaling.
  - Accept size/ color matrices (match MATLAB rules).
  - Renderer support for per-point attributes when using GPU buffers.
- _Status 2024-05-05_: `scatter` and `scatter3` both parse positional + name/value args via the shared parser (size/color vectors, `'filled'`, `'Marker*'`) and now pass per-point metadata straight into the GPU packers. Scatter2/Scatter3 shaders consume optional size + RGBA buffers, `'flat'` marker face colours are honoured (edge `'flat'` still errors by design), and the runtime falls back to host gathers only when the provider can’t export a buffer or the dtype isn’t `single`. CPU paths hydrate host vectors lazily so gpuArray callers still work when plotting is forced off-GPU. New: both builtins respect `'DisplayName'`/`'Label'` so MATLAB series names propagate into legends/figure events on CPU and GPU paths, and the parser no longer mistakes those tokens for inline style strings. Remaining work: broaden parser coverage for `'flat'` line colours / stairs, and wire marker-shape metadata all the way through the renderer so line plots no longer warn/fallback when markers are requested.
- _Status 2025-02-__: Marker edge `'flat'` now mirrors MATLAB semantics: the parser records `'MarkerEdgeColor','flat'`, `resolve_scatter_style` validates that per-point color data exists, and both CPU + zero-copy GPU paths feed the per-vertex colors into the new shader mix so edge colors stay on the device. Tests exercise the parser/runtime wiring. Remaining work: broaden parser coverage for `'flat'` line colours / stairs, and wire marker-shape metadata all the way through the renderer so line plots no longer warn/fallback when markers are requested.

- [x] **`bar`**
  - ✅ Grouped/stacked matrix inputs (per-column series), `'stacked'` option parsing, MATLAB-style defaults, and category labels all work on CPU and zero-copy GPU paths. The GPU packer now understands series indices/row strides, including positive/negative stacking with per-category offsets.
  - ✅ Handle negative values + baseline options. Stacked mode keeps independent positive/negative bases so downward bars accumulate correctly; grouped mode continues to offset series within a category.
  - ✅ Shared parser hooks allow `'FaceColor'`, `'EdgeColor'`, `'FaceAlpha'`, `'BarWidth'`, and `'DisplayName'` pairs across CPU/GPU paths without dropping GPU residency. Styles merge into `BarChart::apply_face_style`/`apply_outline_style` so zero-copy bars honor MATLAB’s options without gathering.

- [ ] **`hist` / `histogram`**
  - Bin edges, normalization (`'pdf'`, `'probability'`, etc.), weights, output counts.
  - Possibly add modern `histogram` builtin if MATLAB expects it.
  - ✅ Weighted histograms now stay zero-copy: `'Weights'` tensors (host or gpuArray) flow into `HistogramGpuWeights`, the compute shader performs float atomics per bin, and the convert pass scales by `'count'`, `'probability'`, or `'pdf'` using the true weighted totals. CPU fallbacks reuse the same normalization logic and degenerate bins gather the weighted sum before emitting the single spike. Remaining work: expose MATLAB’s richer `histogram` builtin and broaden parser/test coverage.
- [ ] **`surf` / `mesh` / variants**
  - Support matrix X/Y inputs, color matrices (`C`), shading options (`flat`, `interp`), lighting toggles, alpha data.
  - Accept name-value args for colormap, lighting, `'EdgeColor'`, `'FaceAlpha'`.
- ✅ New `SurfaceStyleDefaults`/`SurfaceStyle` parser in `core::style` wires `'Colormap'`, `'Shading'`, `'FaceColor'`, `'FaceAlpha'`, `'EdgeColor'` (auto/none), `'Lighting'`, `'DisplayName'`, and `'Visible'` into the zero-copy path. `surf`, `mesh`, `surfc`, and `meshc` now take trailing name/value pairs, preserve their historical defaults (Parula vs. Turbo, shaded vs. wireframe), and propagate the resolved colormap down to contour overlays so mesh/surface projections stay in sync. Remaining work: true `'EdgeColor'` RGB overrides, `'LevelList'`/`'LevelStep'` name/value routing for contour combos, and exposing lighting/material tweaks beyond the basic on/off toggle.
- ✅ `contour`/`contourf` accept MATLAB-style `'LevelList'`, `'LevelStep'`, and `'LineColor'` options (defaulting to auto color tables). GPU + CPU paths share the new `ContourLineColor` enum so explicit colors stay zero-copy; `'LineColor','none'` simply suppresses the contour overlay when used via `contourf`. Need doc/test coverage that mirrors MATLAB semantics (see TODO list below).

- [ ] **Docs/tests**
  - For each builtin, expand `DOC_MD` with new argument coverage, add unit tests for parser edge cases, and integration tests mirroring MATLAB scripts.
  - _Status 2024-05-05_: `stairs` accepts shared parser output (line width, colors) and reflects styles into both CPU + zero-copy GPU packers. Missing pieces: comprehensive doc/tests and marker semantics (MATLAB exposes markers on `stairs`, which we still ignore).
  - _Status 2025-12-??_: `stairs` now honors `'DisplayName'` from name/value pairs so legends match MATLAB. Next up is marker support, line-style cycling, and doc/test coverage.

---

## 3. Multi-Figure UX & Bindings
Goal: expose figure handles/axes info to wasm/TS so the Tauri/web UI can display multiple figures (tabs or simultaneous canvases) without guessing.

  - [ ] **3.1 Runtime surfacing**
  - Decide on the API surface for reporting current figure handle (return value from plotting builtins? event hook? dedicated `gcf` builtin?).
  - Add helpers for `gcf`, `gca`, `close`, `clf` if needed. ✅ (implemented; future work is expanding renderer lifecycle hooks beyond wasm).
  - Ensure regressions (e.g., `figure(2); plot(...)`) update the registry + return consistent status messages.

- [ ] **3.2 runmat-wasm / TS bindings**
  - [x] Emit figure update callbacks via `onFigureEvent` and allow multiple canvases with `registerFigureCanvas(handle, canvas)` (bindings + wasm bridge). `runmat-wasm` now stores a global observer and proxies events to JS; TS bindings expose `onFigureEvent`, `registerFigureCanvas`, and `deregisterFigureCanvas`.
  - [x] Expose wrappers for `figure(handle?)`, `hold(mode?)`, `subplot(m,n,p)` so UI code can drive the registry explicitly instead of issuing MATLAB text commands. `runmat-wasm` exports `newFigureHandle`, `selectFigure`, `currentFigureHandle`, `setHoldMode`, and `configureSubplot`; `@runmat/wasm` mirrors these helpers (plus `holdOn/holdOff`) and documents them in the README.
  - [ ] Surface error status (missing handles, invalid subplot slots, `close`/`clf` responses) through the wasm bridge so hosts can provide UX hints. Future work also needs explicit wrappers for `close`, `clf`, `gcf`, and per-axes metadata.

- [ ] **3.3 UI contract + docs**
  - Document recommended UX patterns (tabs vs. multiple canvases) for the closed-source Tauri app; keep runmat repo references generic.
  - Update `docs/wasm/plan.md` and `docs/filesystem-notes.md` with new plotting considerations (figure handles, GPU requirements).
  - Provide example code in `bindings/ts/README.md` showing how to work with multiple figures.

---

## Current Blocking TODOs (short list)
- ✅ Promote full MATLAB builtin coverage for `close`, `clf`, `gcf`, and `gca` so scripts (not just the wasm API) can drive the new registry helpers. Follow-up completed: the native GUI window path now listens to figure close events, so desktop canvases release without polling.
- Parser parity follow-through: `stairs` now renders markers/line styles zero-copy, `bar`/`hist` accept `'FaceColor','flat'`, matrix `bar` inputs ride zero-copy GPU packers (grouped + stacked), scatter marker edges honour `'flat'`, histogram weighting/probability/pdf scaling stay entirely on the GPU, and `hist` exposes MATLAB’s `NumBins`/`BinWidth`/`BinLimits`/`BinMethod` knobs with proper multi-output handling. Remaining work: land the separate `histogram` builtin plus full doc/test coverage for the remaining parser surfaces (per-series style strings, stairs markers, histogram docstrings/tests).

## Working Notes
- Keep this file updated whenever a checklist item is started/completed; it’s our single source of truth when context gets compressed.
- When an item spans multiple commits (e.g., argument parsing + renderer support), jot interim notes here so we remember what’s blocked.
- Delete this file once the WASM plotting initiative is fully wrapped. Until then, it should contain enough detail to rehydrate the full state of the project quickly.
