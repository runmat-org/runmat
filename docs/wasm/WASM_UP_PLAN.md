# WASM “Ship-Ready” Plan

Goal: deliver a standalone RunMat WASM package that shell/UI teams can consume to build the Monaco/LSP-driven experience sketched in the mock (editor + run button, plotted figures, stdout pane, variable table with shapes, fusion plan view). This plan isolates the remaining work needed on the runtime + bindings side so the UI can focus on presentation.

---

## A. Runtime / WASM Core
1. **Session bring-up**
   - [x] Verify `RunMatSession::init` (wasm) exposes all required knobs: snapshot streaming, filesystem provider selection, telemetry flags (consent + `telemetryId`), optional plot canvas registration. `bindings/ts/README.md` now documents the full option matrix so hosts know what to pass.
   - [x] Provide deterministic error reporting (`InitError` enum) so hosts can surface GPU/FS issues without parsing strings.
     - ✅ `registerFsProvider`, `registerPlotCanvas`, and `registerFigureCanvas` now wrap failures in structured `InitError` payloads (`FilesystemProvider` / `PlotCanvas`) so hosts get consistent error codes even when prereqs are invoked outside `initRunMat`.
   - ✅ Node/Vitest harness now exercises the init surface directly (`bindngs/ts/src/index.spec.ts`), covering fs registration, plot canvas sequencing, telemetry IDs, and the new `scatterTargetPoints` / `surfaceVertexBudget` knobs via a dedicated spec. `npm test` runs clean (24 specs) under Node once the shell sources `~/.zshrc`, giving us fast regression coverage without rebuilding the wasm blob for every tweak.
   - ✅ `telemetryConsent` now feeds `RunMatSession::telemetry_consent`; browsers can opt out during `initRunMat`, and the CLI threads its config into the same flag so analytics collectors stay honest. Hosts can also provide a stable `telemetryId` so any future runtime-level telemetry reuses the same CID that the UI already knows about.
2. **Execution contract**
   - [x] Define the JSON shape returned by `execute(script, opts)` for wasm usage: stdout/stderr arrays, final value preview, workspace diff, figure handles touched, profiler summaries.
   - [x] Ensure long-running executions can be cancelled (runtime interrupt flag + `session.cancelExecution()` in wasm bindings).
   - **Design Draft — `ExecutePayload` additions**
     - `stdout`: ordered list of `{ stream: "stdout" | "stderr", text: string, timestampMs: number }`. Populated incrementally; every run also returns the buffered lines.
     - ✅ `stdinRequested`: optional `{ prompt: string, echo: boolean, requestId: string, waitingMs: number }` when MATLAB code calls `input`. `waitingMs` starts at 0 and grows while the request is outstanding so hosts can surface "still waiting" notices without forcing a timeout. The host resumes execution with `resumeInput(requestId, value)`. The MATLAB-compatible `input` builtin rides this path (numeric parsing via `str2double`, `'s'` text mode), and the single-threaded async-stdin tests now cover numeric prompts, `'s'` mode, and `pause`/keypress flows.
     - `workspace`: `{ full: boolean, values: WorkspaceEntry[] }`, where each entry is `{ name, className, dtype, shape, isGpu, sizeBytes?, preview?: number[], truncated?: boolean }`. `full = false` when only the mutated variables are included.
     - `figuresTouched`: array of figure handles that changed during the run so the UI can switch tabs.
     - `stdinEvents`: ordered log of `{ prompt, kind, echo, value?, error? }` emitted during execution so the UI can render transcripts or debugging panes.
     - `fusionPlan`: `null` when disabled, else `{ nodes, edges, shaders, decisions }` with:
       - `nodes`: `{ id, kind ("tensor"|"op"|"builtin"), label, shape, residency }`.
       - `edges`: `{ from, to, reason }` mirroring the accelerator DAG.
       - `shaders`: per-kernel metadata `{ name, stage, workgroupSize, sourceHash }`.
       - `decisions`: structured log replacing the benchmark harness text `{ nodeId, fused: bool, reason, thresholds }`.
     - `profiling`: `{ totalMs, cpuMs, gpuMs, kernelCount }`.
     - `warnings`: array of MATLAB warning objects `{ identifier, message }`.
- **Streaming API hooks**
  - `subscribeStdout(listener)` + `unsubscribeStdout(id)` to drive the xterm pane in real time. ✅ Implemented via `runmat_runtime::console` forwarder + wasm/TS bindings.
  - `cancelExecution()` to stop a running script (e.g., when a user presses Ctrl+C). ✅ Wired through the runtime interrupt flag.
  - `setInputHandler(handler)` to service `input`/`pause` prompts synchronously from JS. ✅ Handlers can return scalars/objects for immediate responses or `null`/`{ pending: true }` to defer.
  - `stdinRequested`/`session.resumeInput()`/`session.pendingStdinRequests()` to complete the async handshake when hosts can't answer immediately. ✅ Implemented end-to-end (VM suspension, Rust API, wasm bridge, TS helpers, docs, tests).
     - `materializeVariable(name, options)` for lazy previews when the user expands a large tensor.
3. **Resource lifecycle**
   - [x] Expose `reset()`, `dispose()`, and snapshot reload paths so the shell can soft-reset sessions without reloading WASM modules.
     - ✅ `RunMatWasm` exposes `dispose()` (mirrored by the TypeScript wrapper) which cancels executions, drops stdin handlers, and prevents hosts from reusing dead sessions. Hosts can call `session.dispose()` when unmounting a REPL tab to free memory without reloading the wasm blob.
   - [x] Validate memory growth (wasm `memory.grow`) stays within limits and document thresholds for hosts.
     - ✅ `RunMatWasm::memoryUsage()` (surfaced as `session.memoryUsage()` in TS) reports the current heap size in bytes/pages so hosts can monitor growth. The README now explains how to use the API and which init options control GPU/memory budgets.
   - ✅ `cargo check -p runmat-wasm --target wasm32-unknown-unknown` now succeeds after aligning the wasm bindings with the plotting/runtime refactor: `detach_web_renderer` comes from the `plotting::web` module, figure errors cover `InvalidAxesHandle`, stdout subscriptions live in a wasm-only thread-local registry (no `Send + Sync` requirements on `js_sys::Function`), snapshot streams use the correct `ReadableStream` API, and the filesystem parser builds `FsMetadata`/`DirEntry` via new constructors instead of touching private fields. This unblocks the remaining “Session bring-up” verification work in Section A.

## B. Plotting & Figure Bridge
1. **Canvas management**
   - [ ] Finalize `registerFigureCanvas(handle, canvas)` / `deregisterFigureCanvas` semantics (multitab support, tab-switch latency, GPU context reuse).
   - [ ] Emit figure lifecycle events with metadata (`axes grid, titles, legend entries`) so the UI can populate the right column without re-rendering plots in JS.
2. **High-performance render loop**
   - [ ] Stress-test WebGPU plotting under the shared Accelerate context (sinusoid demo, scatter3 cloud) to confirm 60fps+ on target hardware; document fallback behavior when WebGPU is unavailable.
   - [ ] Provide a “headless” render hook so the shell can request image snapshots (for thumbnails/history) without a visible canvas.

## C. Variables / Fusion Panels
1. **Variable inspector**
   - [ ] Extend the execution response with workspace metadata (name, shape, dtype, residency, sample preview pointer). Ensure GPU tensors can be previewed via lazy gather APIs (`materializeVariable(handle, slices)`).
   - [ ] Add hover/peek helpers wired to the in-flight LSP service so Monaco can show shapes/dtypes inline.
2. **Fusion plan / shader view**
   - [ ] Surface the fusion DAG + selected GPU shaders via a structured JSON emitted after each execution. Include timestamps and compiled shader IDs so the UI can sync the lower-pane diagrams.
   - [ ] Guarantee the emission is optional (flag) to keep smaller hosts lightweight.

## D. Filesystem & Remote Providers
1. **Provider selection**
   - [ ] Confirm the JS bindings expose `createMemoryFsProvider`, `createIndexedDbFsProvider`, and `createRemoteFsProvider` with consistent signatures, plus validation for read-only vs. read-write modes.
   - [ ] Document size limits and cleanup strategy (e.g., IndexedDB quotas).
2. **Auth / S3 throughput**
   - [ ] Ensure the Remote FS hooks (HTTP proxy) support auth headers + chunked streaming so a future S3-backed host can saturate bandwidth.
   - [ ] Provide mock/stub servers + tests so shell teams can simulate remote mounts locally.

## E. Packaging & Tooling
1. **Build pipeline**
   - [ ] Script the wasm build (cargo + wasm-bindgen + npm packaging) into `bindings/ts/package-scripts` with reproducible versions.
   - [ ] Add CI sanity tests (wasm `cargo test`, Vitest suite) and publish dry-run instructions.
2. **Documentation**
   - [ ] Produce an integration guide: initialization flow, example React hook, filesystem mounting, plot canvas registration, error handling.
   - [ ] Keep `docs/wasm/plan.md` in sync by linking to this file and marking milestones as they land.

---

**Status Tracking**

| Item | Owner | Status | Notes |
| ---- | ----- | ------ | ----- |
| Session init API audit | | ☑ | Node/Vitest harness now covers fs provider registration, plot canvas wiring, telemetry IDs, and scatter/surface overrides; `npm test` runs clean once Node env sources `~/.zshrc`. |
| Execution response contract | | ☑ | Streams + workspace payloads + stdout subscription landed; cancellation still pending |
| Plot canvas lifecycle | | ☐ | |
| Variable inspector API | | ☐ | |
| Fusion plan emission | | ☐ | |
| FS provider selection docs | | ☐ | |
| WASM build script + CI | | ☐ | |
| Integration guide | | ☐ | |

Update this list as milestones land so UI teams have a single, focused reference for WASM readiness.
