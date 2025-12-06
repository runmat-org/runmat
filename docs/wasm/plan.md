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

5. **CLI relocation & embedding API**
   - Move the CLI crate into `crates/runmat-cli` to isolate the desktop entrypoint from the reusable runtime.
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
- [ ] Implement `runmat-plot/web` and canvas-based rendering.
- [x] Move CLI to `crates/runmat-cli` and stabilize embedding API.
- [ ] Ship `runmat-wasm` bindings plus website integration.
  - [x] Expose wasm-bindgen bindings (`crates/runmat-wasm`) with snapshot streaming + async GPU init.
  - [x] Scaffold `bindings/ts` npm package with build scripts/documented workflow.
  - [ ] Wire the website REPL + plotting demos to the new bindings.
- [ ] Document IO/network limitations and snapshot streaming guidance.

## Status Snapshot (2025-12-05)

### Completed Highlights
- **Runtime extraction & wasm entrypoint** – `runmat-core` now hosts the interpreter/GC (`crates/runmat-core/src/lib.rs`) and `runmat-wasm` exposes async init + snapshot streaming (`crates/runmat-wasm/src/lib.rs`) with optional JS `ReadableStream` sources.
- **Filesystem virtualization** – A dedicated `runmat-filesystem` crate provides pluggable providers (native, sandbox, wasm bridge, remote) and global install hooks, with runtime builtins now routed through it (`crates/runmat-runtime/src/builtins/io/**` plus `crates/runmat-runtime/src/filesystem_provider_tests.rs` for coverage).
- **Remote FS + JS parity** – High-throughput HTTP/S3 proxy via `RemoteFsProvider` (`crates/runmat-filesystem/src/remote/native.rs`) and matching TypeScript helpers (`bindings/ts/src/fs/remote.ts`) backed by a synchronous Node XHR shim (`bindings/ts/src/test/node-sync-xhr.ts`) and worker-isolated test server (`bindings/ts/src/fs/providers.spec.ts`).
- **HTTP builtins & transport refactor** – `webread`/`webwrite` flow through a shared transport layer so native uses `reqwest` while wasm leverages the browser bridge (`crates/runmat-runtime/src/builtins/io/http/transport.rs`).
- **Bindings packaging** – `bindings/ts` now publishes the `@runmat/wasm` ESM wrapper (`bindings/ts/src/index.ts`, `bindings/ts/README.md`) with IndexedDB, in-memory, and remote filesystem providers plus Vitest coverage.
- **Documentation** – Filesystem strategy captured in `docs/filesystem-notes.md` (providers, REST contract, test approach), and this plan stays the canonical overview.

### Remaining / Next Up
- **Plotting/web canvas backend** – `runmat-runtime/src/plotting.rs` still targets native; we need the `runmat-plot/web` adapter, GPU texture sharing, and fallbacks to unlock the UI mock described earlier.
- **Website / REPL integration** – The future `runmat.org` shell and Tauri desktop client still need to consume `@runmat/wasm`, mount filesystem providers (remote + IndexedDB), stream snapshots, and surface GPU/plot signals. No code lives in this repo yet; track via `docs/wasm/plan.md` task list.
- **IO/network docs & gaps** – Produce the promised matrix of MATLAB builtins vs. browser support, including any unavoidable no-ops (sockets, raw file handles) and recommended host bridges.
- **Shape-rich MATLAB value metadata** – `matlab_class_name` serialization currently drops shape info in the wasm path (`crates/runmat-wasm/src/lib.rs`); the Monaco/LSP experience requires preserving it for hover/insight panes.
- **Remote FS configuration UX** – Need CLIs/env vars to mount remote providers natively and comparable hooks in the wasm initializer so end-users can choose memory/IndexedDB/remote without custom glue.
- **CI + packaging polish** – Add wasm target checks/lints to CI, ensure `npm run build` runs as part of release jobs, and document single-threaded runtime tests for GPU/GC constraints.
