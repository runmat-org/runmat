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
   - Build a `runmat-wasm` crate with `#[wasm_bindgen]` bindings that expose init/execute APIs and integrate with TypeScript bindings for web projects.

6. **Host services & telemetry**
   - Replace direct `std::fs`, `std::net`, and thread usage with host abstractions; implement wasm adapters that use browser storage / Fetch / structured concurrency.
   - Telemetry in browsers must use HTTPS fetch only; document opt-in expectations and unsupported UDP paths.

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
- `init` accepts `{ snapshotUrl?: string | ArrayBuffer, canvas?: HTMLCanvasElement, telemetryConsent?: boolean }` so hosts control snapshot streaming, plotting surface, and telemetry.
- Execution results resolve to `{ value?: JsonValue, stdout: string[], plots: PlotArtifact[], residency: 'cpu' | 'gpu' }` to drive REPL-style UIs.
- Register callbacks (`onPlot`, `onTelemetry`, `onStatusChange`) so the website can react to GPU residency, errors, and background downloads without polling.
- Publish generated TypeScript definitions and a thin React hook (`useRunMatSession`) under `website/` for first-party tooling.

## Open Questions
- Which MATLAB IO functions must throw vs. provide JS hooks?
- How do we persist calibration/cache data when persistent storage is unavailable?
- What telemetry granularity is acceptable in a browser (per GDPR/consent requirements)?

## Task Tracker
- [ ] Add workspace features and wasm target check.
- [ ] Extract host-agnostic runtime crate and wasm snapshot loader.
- [ ] Update WebGPU provider for async wasm init and F32 enforcement.
- [ ] Implement `runmat-plot/web` and canvas-based rendering.
- [ ] Move CLI to `crates/runmat-cli` and stabilize embedding API.
- [ ] Ship `runmat-wasm` bindings plus website integration.
- [ ] Document IO/network limitations and snapshot streaming guidance.
