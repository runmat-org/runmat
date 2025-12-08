# RunMat WebAssembly Strategy

This document tracks the technical decisions and workstreams required to bring the full RunMat experience to browsers via WebAssembly. It supersedes prior scratch notes and captures the latest guidance from the WASM planning discussion.

## Goals
- Run the MATLAB-compatible runtime inside modern browsers with WebGPU acceleration, fusion, GC, interpreter, and plotting wherever possible.
- Provide a high-level TypeScript API for embedding RunMat into web apps (REPLs, notebooks, teaching tools).
- Keep feature parity with desktop builds whenever the browser sandbox allows, and clearly document any gaps.

## Platform Constraints & Responses

| Constraint | Response |
| --- | --- |
| **Browser sandbox lacks raw filesystem and sockets.** | Abstract filesystem/config IO and document unsupported MATLAB IO/network functions. Where feasible, bridge through JS (Fetch, IndexedDB) and list fallbacks in this doc. |
| **WebGPU offers single-precision compute today.** | Enforce `ProviderPrecision::F32` in browser builds; the planner already prefers f64 kernels but we will route unsupported double workloads to the CPU path per `docs/GPU_BEHAVIOR_NOTES.md`. |
| **Snapshots are large and cannot be `include_bytes!` indefinitely.** | Treat snapshots as streamed assets (fetch from CDN or embed via JS). The WASM init API accepts snapshot bytes at runtime so hosts can decide how to load them. |
| **No native threads / Cranelift JIT / BLAS.** | Ship interpreter-only builds for wasm (Ignition + GC), rely on WebAssembly SIMD intrinsics for hot CPU kernels, and lean on WebGPU for heavy math instead of BLAS/LAPACK. |

## Architecture Changes

1. **Workspace features & targets**
   - Add feature flags (`jit`, `native-blas`, `native-gui`, etc.) at the workspace level to let `cargo build --target wasm32-unknown-unknown` disable host-only subsystems.
   - Ensure a wasm CI job runs `cargo check --target wasm32-unknown-unknown` with the correct feature set.

2. **Core runtime extraction**
   - Factor interpreter/GC/builtins/snapshot loading into a host-agnostic crate ("runmat-core").
   - Teach the snapshot loader to consume `&[u8]` and async byte streams so the wasm build can accept snapshot payloads from JS.

3. **WebGPU backend updates**
   - Make the wgpu provider async-friendly (no `pollster::block_on`) and support surfaces created from an HTML `<canvas>`.
   - Gate disk caches and other host filesystem access behind `cfg(not(target_arch = "wasm32"))`.
   - Report precision capabilities accurately so f64 graphs automatically fall back to CPU per GPU behavior notes.

4. **Plotting**
   - Implement the `runmat-plot/web` feature: render into a provided canvas, schedule frames via `requestAnimationFrame`, and drop thread manager dependencies.
   - Fall back to static SVG/PNG rendering when WebGPU is unavailable so docs/examples still work.
   - **Zero-copy renderer plan**: see `docs/wasm/plotting-zero-copy.md` for the shared-device design (Accelerate-exported WGPU context, GPU tensor handles, async frame scheduling) that keeps plotting and fusion on the same device/queue in both native and wasm builds.

5. **CLI relocation & embedding API**
   - ✅ Move the CLI crate into `crates/runmat-cli` to isolate the desktop entrypoint from the reusable runtime.
   - ✅ `crates/runmat-wasm` now exposes the `initRunMat` async API (snapshot streaming + async WebGPU bring-up) for browser hosts.
   - ✅ `bindings/ts` scaffolds the npm-ready wrapper that re-exports the wasm API as a typed ESM module (`@runmat/wasm`). Website integration still pending.

6. **Host services & telemetry**
   - Replace direct `std::fs`, `std::net`, and thread usage with host abstractions; implement wasm adapters that use browser storage / Fetch / structured concurrency.
   - Telemetry in browsers must use HTTPS fetch only; document opt-in expectations and unsupported UDP paths.
   - ✅ `webread` / `webwrite` now ride on a shared HTTP transport layer that uses `reqwest` on native builds and synchronous `XMLHttpRequest` inside browsers (with Basic Auth + header handling). Calls still obey browser sandbox rules (CORS, no raw sockets) but MATLAB APIs stay intact.
   - ✅ `GpuStatus` surfaced by `runmat-wasm` now includes adapter metadata (name/vendor/backend/memory/precision) pulled from the active provider so the REPL can show accurate capability badges.
   - ✅ Snapshot ingestion accepts JS `ReadableStream` handles (either via `snapshot.stream` or fetchers that return `Response`/streams) so hosts can forward CDN downloads without buffering entire archives in JS memory.

## Documentation & Product Notes
- Maintain `/docs/wasm/` as the canonical home for browser-related docs, including feature parity tables and IO/network limitations.
- Explicitly document which MATLAB IO/network builtins are unsupported, partially supported (via Fetch), or require server-side helpers.
- Explain the snapshot streaming requirement for third parties who host RunMat in static sites.

## Deliverables
1. Interpreter-only wasm build of the core runtime with snapshot ingestion.
2. WebGPU provider capable of fused elementwise/reduction kernels inside browsers.
3. Plotting backend that works with browser canvases.
4. TypeScript bindings + example integration inside the `website/` Next.js app (REPL, plotting demo).
5. Documentation of IO/network gaps and feature switches.

## Browser Embedding API Expectations
- Ship a `runmat-wasm` package that exposes `init(options)` returning a `RunMatSession` with methods like `execute(script: string, opts?: ExecuteOptions)` and `reset()`.
- `init` accepts `{ snapshot?: { bytes | url | stream | fetcher }, canvas?: HTMLCanvasElement, telemetryConsent?: boolean }` so hosts control snapshot streaming, plotting surface, and telemetry. Hosts that already fetched a snapshot can hand us a `ReadableStream` and avoid another buffer copy.
- Execution results resolve to `{ value?: JsonValue, stdout: string[], plots: PlotArtifact[], residency: 'cpu' | 'gpu' }` to drive REPL-style UIs.
- Register callbacks (`onPlot`, `onTelemetry`, `onStatusChange`) so the website can react to GPU residency, errors, and background downloads without polling.
- `RunMatSession.gpuStatus()` returns `{ requested, active, error?, adapter? }` where `adapter` includes `{ name, vendor, backend, deviceId, memoryBytes?, precision }`. The Monaco-driven UI can display this next to the canvas to explain why plotting or fusion fell back to CPU.
- Publish generated TypeScript definitions and a thin React hook (`useRunMatSession`) under `website/` for first-party tooling.

## Open Questions
- Which MATLAB IO functions must throw vs. provide JS hooks?
- How do we persist calibration/cache data when persistent storage is unavailable?
- What telemetry granularity is acceptable in a browser (per GDPR/consent requirements)?

## Build & Test Notes
- `cargo check -p runmat-wasm` exercises the host build; cross-compiling for browsers requires `rustup target add wasm32-unknown-unknown` followed by `cargo check -p runmat-wasm --target wasm32-unknown-unknown`.
- The npm wrapper lives in `bindings/ts`. Run `npm install` followed by `npm run build` (invokes `wasm-pack` + `tsc`) to produce publishable `pkg/` + `dist/` artifacts.

## Task Tracker
- [x] Add workspace features and wasm target check.
- [x] Extract host-agnostic runtime crate and wasm snapshot loader.
- [x] Update WebGPU provider for async wasm init and F32 enforcement.
- [x] Implement `runmat-plot/web` and canvas-based rendering.
- [x] Move CLI to `crates/runmat-cli` and stabilize embedding API.
- [ ] Ship `runmat-wasm` bindings plus website integration.
  - [x] Expose wasm-bindgen bindings (`crates/runmat-wasm`) with snapshot streaming + async GPU init.
  - [x] Scaffold `bindings/ts` npm package with build scripts/documented workflow.
  - [x] Surface WebGPU plotting registration from wasm + TypeScript (canvas attach helpers + docs).
  - [ ] Wire the website REPL + plotting demos to the new bindings.
- [ ] Document IO/network limitations and snapshot streaming guidance.

## Status Snapshot (2025-12-05)

### Completed Highlights
- **Runtime extraction & wasm entrypoint** – `runmat-core` now hosts the interpreter/GC (`crates/runmat-core/src/lib.rs`) and `runmat-wasm` exposes async init + snapshot streaming (`crates/runmat-wasm/src/lib.rs`) with optional JS `ReadableStream` sources.
- **Filesystem virtualization** – A dedicated `runmat-filesystem` crate provides pluggable providers (native, sandbox, wasm bridge, remote) and global install hooks, with runtime builtins now routed through it (`crates/runmat-runtime/src/builtins/io/**` plus `crates/runmat-runtime/src/filesystem_provider_tests.rs` for coverage).
- **Remote FS + JS parity** – High-throughput HTTP/S3 proxy via `RemoteFsProvider` (`crates/runmat-filesystem/src/remote/native.rs`) and matching TypeScript helpers (`bindings/ts/src/fs/remote.ts`) backed by a synchronous Node XHR shim (`bindings/ts/src/test/node-sync-xhr.ts`) and worker-isolated test server (`bindings/ts/src/fs/providers.spec.ts`).
- **HTTP builtins & transport refactor** – `webread`/`webwrite` flow through a shared transport layer so native uses `reqwest` while wasm leverages the browser bridge (`crates/runmat-runtime/src/builtins/io/http/transport.rs`).
- **Bindings packaging** – `bindings/ts` now publishes the `@runmat/wasm` ESM wrapper (`bindings/ts/src/index.ts`, `bindings/ts/README.md`) with IndexedDB, in-memory, and remote filesystem providers plus Vitest coverage.
- **Web plotting renderer** – `runmat-plot` ships a wasm-only `web` module (`crates/runmat-plot/src/web.rs`) that drives a `<canvas>` via WebGPU; `runmat-runtime/src/plotting.rs` uses it directly in wasm builds, and `runmat-wasm` exposes async `registerPlotCanvas`/`plotRendererReady` with matching TS helpers (`bindings/ts/src/index.ts`).
- **Shared WGPU context plumbing** – Accelerate’s exported context now carries the originating `wgpu::Instance`, and both the wasm renderer plus the native GUI window reuse that shared instance/adapter/device/queue combo to avoid duplicate adapters and make zero-copy plotting deterministic.
- **Stairs builtin + GPU packer** – MATLAB’s `stairs` is now a real builtin (`crates/runmat-runtime/src/builtins/plotting/stairs.rs`) with a zero-copy path via the new `runmat-plot::gpu::stairs` compute shader, so stairstep plots stay on the shared WGPU device just like `plot`/`scatter`.
- **Contour overlays (`surfc` / `meshc`)** – Added GPU marching-squares packers (`crates/runmat-plot/src/gpu/contour.rs`) and runtime helpers (`crates/runmat-runtime/src/builtins/plotting/contour.rs`) so `surfc`/`meshc` render both the surface/mesh and their contour projections straight from gpuArray buffers. CPU fallbacks now share the same figure semantics, paving the way for filled-contour work.
- **Filled contours (`contourf`)** – `contourf` now renders zero-copy on gpuArrays (triangles packed by `gpu::contour_fill`), falls back to CPU when needed, and overlays line contours to match MATLAB defaults. Legends/figures understand the new plot element.
- **Plot style parsing** – `plot` now understands MATLAB-style style strings, name-value pairs (`'LineWidth'`, `'Color'`, `'LineStyle'`, `'Marker*'`), and multiple X/Y series. Inline style strings or name-value pairs can follow each `(x,y)` pair (e.g., `plot(x1,y1,'r',x2,y2,'--')`), and `'LineStyleOrder'` accepts string scalars, string arrays, or cell arrays to cycle default styles per axes across successive calls (reset on `hold off`). Marker specs are parsed, stored, and rendered zero-copy: `LineMarkerAppearance` feeds a dedicated point pipeline so hybrid line+marker plots stay on the GPU and no longer force CPU fallbacks in native or wasm builds.
- **Line GPU styles** – The shared compute packer now emits dashed/dash-dot/dotted segments and multi-pixel widths entirely on the device. Segments expand into either line lists or extruded triangle strips, an atomic counter reports the actual vertex count, and the renderer flips between `PipelineType::Lines` and `PipelineType::Triangles` so dashed/thick plots stay zero-copy in both native and wasm builds.
- **Scatter/Surface LOD + stress harness** – Scatter packers now honor an env-configurable budget (`RUNMAT_PLOT_SCATTER_TARGET`, default 250k points) when choosing compute-side decimation, scatter3 reuses the same heuristics, and the surface packer derives stride/Lod dims from `RUNMAT_PLOT_SURFACE_VERTEX_BUDGET` (default 400k vertices) scaled by the plot extents. Runtime setters (`set_scatter_target_points`, `set_surface_vertex_budget`) let CLIs/TS bindings override these knobs without touching env vars. New GPU stress tests in `runmat-plot/src/gpu/{scatter2,scatter3,surface}.rs` allocate multi-million element datasets on a headless WGPU device (skipping automatically unless `RUNMAT_PLOT_FORCE_GPU_TESTS=1`) so we can sanity-check zero-copy perf outside CPU fallbacks.
- **Bar/Hist parser parity** – `bar` and `hist` share the new `parse_bar_style_args` helper, so MATLAB-style options (`'FaceColor'`, `'EdgeColor'`, `'FaceAlpha'`, `'BarWidth'`, `'DisplayName'`) work across CPU and gpuArray paths. GPU builders respect those styles zero-copy by patching `BarChart::apply_face_style`/`apply_outline_style`, and the histogram parser now separates bin arguments from style args so normalization + styling compose cleanly.
- **Scatter/Surface LOD + stress harness** – Scatter packers now honor an env-configurable budget (`RUNMAT_PLOT_SCATTER_TARGET`, default 250k points) when choosing compute-side decimation, and the surface packer derives stride/Lod dims from `RUNMAT_PLOT_SURFACE_VERTEX_BUDGET` (default 400k vertices). New GPU stress tests in `runmat-plot/src/gpu/scatter2.rs` and `surface.rs` allocate multi-million element datasets on a headless WGPU device (skipping automatically when `RUNMAT_PLOT_SKIP_GPU_TESTS=1`) so we can sanity-check the zero-copy path outside of CPU fallbacks.
- **Surface-style parser coverage** – `surf`, `mesh`, `surfc`, and `meshc` now accept MATLAB-style name/value pairs (`'Colormap'`, `'Shading'`, `'FaceColor'`, `'LineColor'`, `'FaceAlpha'`, `'Lighting'`, `'DisplayName'`, `'Visible'`). The new `SurfaceStyleDefaults`/`SurfaceStyle` helpers (in `crates/runmat-runtime/src/builtins/plotting/core/style.rs`) let each builtin keep its historical defaults (Parula vs. Turbo, wireframe vs. shaded) while honoring overrides across both CPU and zero-copy GPU paths; contour overlays reuse the resolved colormap so the surface + projected lines stay in sync.
- **LOD knobs exposed to hosts** – In addition to the env vars, the CLI config/flags and wasm/TypeScript bindings now accept overrides that call `set_scatter_target_points` / `set_surface_vertex_budget` directly (`crates/runmat-cli/src/config.rs`, `crates/runmat-cli/src/main.rs`, `crates/runmat-wasm/src/lib.rs`, `bindings/ts/src/index.ts`, `bindings/ts/README.md`). Browser hosts can tune plotting without mutating env vars.
- **Plotting context visibility** – `ensure_context_from_provider` now returns a typed error and both CLI + wasm init propagate actionable messages when zero-copy plotting is unavailable (`crates/runmat-runtime/src/builtins/plotting/core/context.rs`, `crates/runmat-core/src/lib.rs`, `crates/runmat-cli/src/main.rs`, `crates/runmat-wasm/src/lib.rs`).
- **Contour options** – `contour`/`contourf` parse MATLAB-style `'LevelList'`, `'LevelListMode'`, `'LevelStep'`, and `'LineColor'` arguments so scripts that customize iso-lines keep working; the GPU/CPU paths share the new `ContourLineColor` wiring and the builtin docs/tests now spell out the supported knobs (`crates/runmat-runtime/src/builtins/plotting/ops/contour*.rs`).
- **Multi-figure wasm plumbing** – The runtime can notify hosts whenever a figure updates and the wasm bindings expose `onFigureEvent` + `registerFigureCanvas(handle, canvas)` so browser/Tauri shells can drive per-handle canvases without polling (`crates/runmat-runtime/src/builtins/plotting/core/{state,engine,web}.rs`, `crates/runmat-wasm/src/lib.rs`, `bindings/ts/src/index.ts`, `bindings/ts/README.md`). Hosts now also have `figure()`, `newFigureHandle()`, `currentFigureHandle()`, `setHoldMode()`, `configureSubplot()`, `clearFigure()`, `closeFigure()`, and `currentAxesInfo()` helpers so tabs/UI widgets can push figure/hold/subplot changes straight through the wasm bridge without emitting MATLAB text commands. The bindings surface structured errors (`InvalidHandle`, `InvalidSubplotGrid`, `InvalidSubplotIndex`) so UIs can react without parsing MATLAB text.
- **Figure lifecycle builtins** – `close`, `clf`, `gcf`, and `gca` are now first-class runtime builtins that call the shared registry helpers. MATLAB scripts can close or clear handles without shell glue, and `gca('struct')` returns the axes metadata for callers that need rows/cols/index details.
- **Renderer detach signals** – `FigureEventKind::Closed` now causes the wasm bridge to drop any registered WebGPU canvas for the closed handle (`runmat-runtime/src/builtins/plotting/core/web.rs`, `crates/runmat-wasm/src/lib.rs`), so browser/Tauri hosts no longer need to poll before tearing down their DOM nodes. Native GUI windows hook into the same lifecycle: `runmat-runtime` installs a desktop observer that notifies `runmat-plot` when a figure closes, and the single-window manager, native window manager, and GUI thread manager now all listen for those close requests via the shared atomic signal so macOS/winit windows exit as soon as MATLAB code calls `close(handle)`.
- **Desktop backend routing** – The lifecycle module now tries the native window manager first, then the GUI thread manager, and finally the single-window fallback (unless `RUNMAT_PLOT_DESKTOP_BACKEND` overrides the order). Each backend receives the shared close signal, so multi-window UI shells can rely on the high-throughput native/GUI-thread renderers while CLIs still have a safe single-window fallback.
- **Stairs + histogram parser parity** – `stairs` inherits the shared line-style parser end-to-end (marker metadata now feeds both CPU + GPU renderers), and `bar`/`hist` accept MATLAB’s `'FaceColor','flat'` syntax, generating per-bar colors on the CPU path while automatically skipping the GPU packers when flat colors are requested.
- **Surface-style parser coverage** – `surf`, `mesh`, `surfc`, and `meshc` now accept MATLAB-style name/value pairs (`'Colormap'`, `'Shading'`, `'FaceColor'`, `'EdgeColor'`, `'FaceAlpha'`, `'Lighting'`, `'DisplayName'`, `'Visible'`). The new `SurfaceStyleDefaults`/`SurfaceStyle` helpers (in `crates/runmat-runtime/src/builtins/plotting/core/style.rs`) let each builtin keep its historical defaults (Parula vs. Turbo, wireframe vs. shaded) while honoring overrides across both CPU and zero-copy GPU paths; contour overlays reuse the resolved colormap so the surface + projected lines stay in sync.
- **LOD knobs exposed to hosts** – In addition to the env vars, the CLI config/flags (`plot-scatter-target`, `plot-surface-vertex-budget`) and wasm/TypeScript bindings (`scatterTargetPoints`, `surfaceVertexBudget`) now call `set_scatter_target_points` / `set_surface_vertex_budget` directly. This makes the perf budgets part of the public API (`crates/runmat-cli/src/main.rs`, `bindings/ts/src/index.ts`, `bindings/ts/README.md`), so browser shells and headless builds can tune plotting without munging process env.
- **Plotting context error surfacing** – `ensure_context_from_provider` now returns a typed error instead of silently returning `None` (`crates/runmat-runtime/src/builtins/plotting/core/context.rs`), and both the CLI and wasm init paths log/bubble failures. When a GPU was requested but the shared WGPU handle can’t be exported, `runmat-cli` warns at startup and `runmat-wasm` records the reason in `gpuStatus`, making zero-copy issues visible to hosts instead of failing later during plotting.
- **Standalone `contour` builtin** – MATLAB’s 2-D contour plot now lives in `crates/runmat-runtime/src/builtins/plotting/contour.rs` with full axis/level parsing, zero-copy gpuArray support (explicit level buffers feed the marching-squares shader), and CPU fallbacks that mirror MATLAB’s errors. Tests cover implicit axes, explicit axis validation, and level parsing corner cases.
- **Documentation** – Filesystem strategy captured in `docs/filesystem-notes.md` (providers, REST contract, test approach), and this plan stays the canonical overview.

### Remaining / Next Up
- **Plotting UX + UI wiring** – Attach the Monaco/LSP workspace, live overlay, and REPL-driven plot inspector to the new canvas bridge (current work only renders the WGPU figure; no editor plumbing yet).
- **Figure lifecycle API gaps** – ✅ Both wasm and native GUI paths now listen to `FigureEventKind::Closed`. Browser hosts drop canvases immediately, and the desktop single-window manager keeps an atomic close signal per figure handle so MATLAB `close(...)` tears down the corresponding OS window without polling. Remaining lifecycle work lives in the JS bindings (structured errors everywhere) rather than the renderer bridge.
- **Plot argument parity** – Finish the parser/renderer work for per-series styles (style strings between series, `LineStyleOrder`) and the remaining MATLAB options so scripts that mix `plot(x1,y1,'r',x2,y2,'--')` or `'Marker','o'` continue to run without manual adaptation (markers already ride the zero-copy path).
- **Website / REPL integration** – The future `runmat.org` shell and Tauri desktop client still need to consume `@runmat/wasm`, mount filesystem providers (remote + IndexedDB), stream snapshots, and surface GPU/plot signals. No code lives in this repo yet; track via `docs/wasm/plan.md` task list.
- **IO/network docs & gaps** – Produce the promised matrix of MATLAB builtins vs. browser support, including any unavoidable no-ops (sockets, raw file handles) and recommended host bridges.
- **Shape-rich MATLAB value metadata** – `matlab_class_name` serialization currently drops shape info in the wasm path (`crates/runmat-wasm/src/lib.rs`); the Monaco/LSP experience requires preserving it for hover/insight panes.
- **Remote FS configuration UX** – Need CLIs/env vars to mount remote providers natively and comparable hooks in the wasm initializer so end-users can choose memory/IndexedDB/remote without custom glue.
- **CI + packaging polish** – Add wasm target checks/lints to CI, ensure `npm run build` runs as part of release jobs, and document single-threaded runtime tests for GPU/GC constraints.
