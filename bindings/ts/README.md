# @runmat/wasm

TypeScript/ESM bindings for the `runmat-wasm` crate. The package exposes an async `initRunMat` helper that boots the WASM runtime, streams optional snapshot bytes, and returns a browser-friendly session wrapper.

## Developing

```bash
cd bindings/ts
npm install
npm run build   # runs wasm-pack + tsc
```

The build step expects `wasm-pack` to be installed locally and will output the generated glue code to `pkg/` and the typed wrapper bundle to `dist/`.

## Publishing

The `prepublishOnly` hook re-runs the full build so CI can simply execute `npm publish` once credentials are configured. The resulting tarball includes:

- `pkg/`: raw `wasm-bindgen` output for bundlers
- `dist/`: TypeScript-authored convenience wrapper + definitions
- `README.md`: usage notes

## Filesystem providers

Browser and hybrid hosts must forward filesystem requests to the runtime through a `fsProvider` object passed to `initRunMat`. The package now ships helpers for the two most common scenarios:

- `createInMemoryFsProvider(options?)` – zero-dependency, synchronous filesystem stored entirely in JS memory. Supports all RunMat file ops and is ideal for tests or ephemeral browser sessions.
- `createIndexedDbFsHandle(options?)` – wraps the in-memory provider and persists its state to IndexedDB. The handle exposes `.provider` (pass to `initRunMat`), `.flush()` to await persistence, and `.close()` to release the database. A convenience `createIndexedDbFsProvider` is also exported if you only need the provider.
- `createDefaultFsProvider()` – automatically tries IndexedDB → in-memory. `initRunMat` calls this for you when `fsProvider` is omitted.

Both providers implement the `RunMatFilesystemProvider` contract:

- `readFile(path) => Uint8Array | ArrayBuffer`
- `writeFile(path, data) => void`
- `removeFile(path) => void`
- `metadata(path) => { fileType, len, modified?, readonly? }`
- `readDir(path) => Array<{ path, fileName, fileType? }>`

Optional helpers such as `createDir`, `rename`, `setReadonly`, etc. unlock the full MATLAB IO surface.

See `docs/filesystem-notes.md` in the repo for the detailed contract and backend-specific guidance.

## Snapshot loading

When calling `initRunMat`, pass a `snapshot` object to control how the initial workspace is hydrated:

```ts
await initRunMat({
  snapshot: {
    url: "https://cdn.runmat.org/snapshots/core.bin"
  }
});
```

- `snapshot.bytes`: direct `Uint8Array`/`ArrayBuffer`.
- `snapshot.url`: the helper will stream the response via `fetch` (fewer copies than `arrayBuffer()`).
- `snapshot.stream`: pass a `ReadableStream` (for example `response.body`) if you already fetched the asset in JS and want wasm to consume it directly.
- `snapshot.fetcher`: custom async hook (`({ url }) => Promise<ArrayBuffer | Uint8Array | Response | ReadableStream>`) for bespoke CDNs or authenticated flows. Streams are forwarded without buffering.

If no snapshot is provided, the runtime boots with the minimal built-in seed.

## GPU detection

`enableGpu` defaults to `true`, but the wrapper now auto-detects whether `navigator.gpu` exists in the host. If a caller requests GPU but the browser lacks WebGPU support, the wrapper logs a warning and falls back to CPU execution automatically.

Once a session is running you can inspect the live GPU state via:

```ts
const session = await initRunMat();
const gpu = session.gpuStatus();
if (!gpu.active && gpu.error) {
  console.warn("GPU init failed:", gpu.error);
}

if (gpu.adapter) {
  console.log("GPU backend:", gpu.adapter.name, gpu.adapter.backend, gpu.adapter.precision);
}
```

## Telemetry consent

Browser hosts must decide whether anonymous telemetry is allowed before booting the runtime. Pass `telemetryConsent: false` to `initRunMat` to opt out:

```ts
const session = await initRunMat({
  telemetryConsent: false,
  snapshot: { url: "/snapshots/core.bin" }
});
```

When consent is disabled the runtime simply refrains from emitting analytics events (profiling data and fusion statistics are still returned locally so performance panes work). The CLI mirrors this behavior automatically by forwarding the user’s `telemetry.enabled` setting into the session, and wasm hosts can query `session.telemetryConsent()` to keep their UI in sync.

If you already have a telemetry/analytics identifier (e.g., the ID that the surrounding UI uses), pass it via `telemetryId`. The runtime stores it internally (accessible via `session.telemetryClientId()`) so any future telemetry sinks can reuse the existing CID instead of minting a second identifier.

## Execution streaming & interaction

- `subscribeStdout(listener)` / `unsubscribeStdout(id)` stream stdout/stderr events as they are emitted so hosts can drive an xterm pane without waiting for `execute()` to resolve. Every `ExecuteResult` also includes the buffered `stdout` array for easy logging or replay.
- `ExecuteResult.warnings` exposes structured `{ identifier, message }` entries pulled from MATLAB's warning store, `stdinEvents` captures every prompt/response emitted during the run for transcript panes, and `stdinRequested` is populated when the interpreter suspends while waiting for input.
- Call `session.cancelExecution()` to cooperatively interrupt a long-running script (e.g., when users press the stop button). The runtime raises `MATLAB:runmat:ExecutionCancelled`, matching desktop builds.
- `session.setInputHandler(handler)` registers a synchronous callback for MATLAB's `input`/`pause` prompts. Handlers receive `{ kind: "line" | "keyPress", prompt, echo }` and can return a string/number/boolean, `{ kind: "keyPress" }`, or `{ error }` to reject the prompt. Returning `null`, `undefined`, `{ pending: true }`, or a Promise signals that the handler will respond asynchronously.
- When a handler defers, `execute()` resolves with `stdinRequested` containing `{ id, request, waitingMs }`. Call `session.resumeInput(id, value)` once the UI collects the user's response (value follows the same shape as the input handler). `waitingMs` starts at zero and grows until the prompt is satisfied so UIs can show “still waiting…” nudges without forcing a timeout. Use `session.pendingStdinRequests()` to list outstanding prompts (useful when rehydrating a UI after refresh) — each entry carries the same `waitingMs` counter.

## Plotting surfaces

RunMat plotting now renders directly into a WebGPU-backed `<canvas>` using the same renderer as the native desktop build. Provide a canvas during initialization:

```ts
const canvas = document.getElementById("runmat-plot") as HTMLCanvasElement;
await initRunMat({ plotCanvas: canvas });
```

or attach one later via the exported helpers:

```ts
import { attachPlotCanvas, plotRendererReady } from "@runmat/wasm";

await attachPlotCanvas(canvas);
if (!await plotRendererReady()) {
  console.warn("Plotting not initialized yet.");
}
```

Once the canvas is registered, calling `plot`, `scatter`, etc. from the RunMat REPL renders directly into that surface without any additional JS shims.

## Lifecycle

Each `RunMatSessionHandle` now exposes `session.dispose()`. Call it when tearing down the editor/REPL view so the runtime can cancel pending executions, release stdin handlers, and drop any registered plot canvases. The wrapper marks the instance as disposed and throws helpful errors if a host accidentally calls `execute()` afterwards. `dispose()` is idempotent, so repeated calls are safe.

### Plotting performance knobs

Hosts can tune the GPU level-of-detail heuristics without touching environment variables. Pass either (or both) of the following when calling `initRunMat`:

- `scatterTargetPoints`: preferred number of scatter/scatter3 points to retain per dispatch before compute-side decimation kicks in (default `250_000`).
- `surfaceVertexBudget`: maximum number of surface vertices to pack before LOD sampling starts (default `400_000`).

These map directly to the runtime setters (`set_scatter_target_points`, `set_surface_vertex_budget`) so native CLI builds and the wasm bindings stay in sync.

### Multi-figure canvases & events

- `registerFigureCanvas(handle, canvas)` wires a specific `<canvas>` to a MATLAB figure handle so multiple figures can render concurrently (e.g., tabs or split panes).
- `onFigureEvent(listener)` registers a callback that receives `{ handle, axesRows, axesCols, plotCount, axesIndices[] }` whenever a figure updates. Pass `null` to unsubscribe.

The default `registerPlotCanvas` continues to serve the legacy single-canvas flow; hosts can mix both APIs as needed.

### Figure orchestration helpers

The wasm bindings now expose the same figure/axes controls that the MATLAB runtime uses so hosts can drive multi-tab canvases without issuing textual commands:

- `figure(handle?)` – selects an existing figure handle (creating it if `handle` is omitted) and returns the active handle.
- `newFigureHandle()` / `currentFigureHandle()` – explicit helpers for creating or querying handles when wiring UI tabs.
- `setHoldMode(mode)` / `hold(mode?)` / `holdOn()` / `holdOff()` – toggle MATLAB's `hold` state from JS using `"on" | "off" | "toggle"` (boolean flags also work).
- `configureSubplot(rows, cols, index)` / `subplot(rows, cols, index)` – mirror MATLAB's subplot grid selection so canvas layouts stay in sync with the runtime registry.
- `clearFigure(handle?)` / `closeFigure(handle?)` – mirror `clf`/`close` semantics. Omit the handle (or pass `undefined`) to target the current figure.
- `currentAxesInfo()` – returns `{ handle, axesRows, axesCols, activeIndex }` so hosts can surface the active subplot without scraping text output.

Each helper forwards directly to the new wasm exports, so the zero-copy renderer stays in lock-step with MATLAB semantics even when figure switches originate from the host UI.

When a lifecycle call fails (bad handle, invalid subplot index, etc.) the Promise rejects with a structured error object `{ code, message, ... }`. The wrapper converts that into a real `Error` (with `.code`, `.handle`, `.rows`, `.cols`, `.index` where applicable) so host UIs can surface meaningful messages without parsing MATLAB text.
