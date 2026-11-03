# RunMat

RunMat is a fast MATLAB-compatible runtime. Use the `runmat` package to embed RunMat in a browser, web worker, Electron app, or Node-based tool.

```bash
npm install runmat
```

The package loads RunMat's WebAssembly runtime, creates a session, and gives the host a typed API for execution, workspace inspection, plotting, filesystem integration, and diagnostics.

## Quick Start

```ts
import { initRunMat } from "runmat";

const session = await initRunMat({
  language: { compat: "matlab" }
});

const result = await session.executeRequest({
  source: {
    kind: "text",
    name: "<repl>",
    text: "A = magic(3); disp(A)"
  }
});

console.log(result.stdout);
console.log(result.workspace.values);

session.dispose();
```

`executeRequest` accepts inline source or a path resolved by the configured filesystem provider:

```ts
await session.executeRequest({
  source: { kind: "path", path: "src/main.m" }
});
```

## Initialization

`initRunMat` loads the WebAssembly module, resolves optional startup assets, creates a `RunMatSessionHandle`, and applies host settings.

| Option | Purpose |
| --- | --- |
| `enableGpu` | Request WebGPU acceleration. Defaults to GPU when WebGPU is available, with CPU fallback otherwise. |
| `enableJit` | Enable the JIT tier where supported. |
| `fsProvider` | Provide file I/O for `load`, `save`, scripts, datasets, and path execution. |
| `applicationSettings` | Set `"isolatedMemory"` for session-local preferences and user-path settings. Omit it for application-owned browser IndexedDB persistence. This setting is separate from file and package storage. |
| `plotCanvas` | Attach a default plotting canvas during startup. |
| `telemetryConsent`, `telemetryId`, `telemetryEmitter` | Configure telemetry before the runtime starts. |
| `logLevel`, `emitFusionPlan`, `callstackLimit`, `errorNamespace` | Configure diagnostics and execution metadata. |
| `language.compat` | Choose `"runmat"` (the default), `"matlab"`, or `"strict"` language policy. |

GPU initialization is opportunistic in browser contexts. If WebGPU is unavailable, RunMat logs a warning and continues on CPU.

### Node initialization

Node hosts supply the Wasm bytes and select session-isolated application settings rather than browser persistence. An in-memory filesystem keeps script files and outputs separate from application settings:

```ts
import { readFile } from "node:fs/promises";
import { createInMemoryFsProvider, initRunMat } from "runmat";

const session = await initRunMat({
  wasmModule: await readFile(new URL("./pkg-web/runmat_wasm_web_bg.wasm", import.meta.resolve("runmat"))),
  applicationSettings: "isolatedMemory",
  fsProvider: createInMemoryFsProvider(),
  enableGpu: false
});
```

Use the Wasm file distributed with the same SDK build. Session-isolated preferences and user-path changes survive `resetSession()` and are discarded when the session is disposed; they do not provide persistent Node application settings.

### Building the exact CAD browser host

The browser CAD module uses the same Rust/C++ OCCT importer as desktop for preview
and validated exact topology. Build it with Rust 1.93.0, its
`wasm32-unknown-emscripten` target, and an activated Emscripten 4.0.23 SDK:

```bash
rustup target add wasm32-unknown-emscripten --toolchain 1.93.0
npm run build:occt
npm run test:exact-cad:wasm
```

`npm run build` includes this module. Its JavaScript, Wasm, and OCCT license files
are packaged under `dist/occt`. The qualification test uses the compiled module
with the browser runtime, including invalid-input recovery. Keep the module and
runtime from the same package build. CAD import runs in the runtime worker.

## Execution Results

`executeRequest` resolves to an `ExecuteResult` with structured data for UI hosts:

| Field | Use |
| --- | --- |
| `flow`, `valueText`, `valueJson` | Returned values and displayable summaries. |
| `displayEvents` | Values displayed during execution. |
| `stdout` | Buffered stdout/stderr entries. |
| `workspace` | Variable names, classes, shapes, residency, previews, and preview tokens. |
| `figuresTouched` | Figure handles changed by the request. |
| `warnings` | Structured warning entries. |
| `stdinEvents` | Prompt/response transcript for interactive input. |
| `error` | Structured syntax, semantic, compile, or runtime error details. |
| `profiling`, `fusionPlan` | Optional performance and acceleration metadata. |

When execution rejects, wrapper calls throw `RunMatExecutionError` with the structured diagnostic attached.

## Workspace Inspection

Workspace snapshots are designed for variable panes, hover cards, and notebook state views. Each entry includes class, dtype, shape, residency, size, a small preview, and an optional `previewToken`.

```ts
const entry = result.workspace.values.find((value) => value.name === "A");

if (entry?.previewToken) {
  const full = await session.materializeVariable(
    { previewToken: entry.previewToken, name: entry.name },
    { limit: 4096 }
  );

  console.log(full.valueText);
}
```

## Filesystem Providers

RunMat routes file I/O through a JavaScript filesystem provider. If no provider is passed, the package creates a default provider that tries IndexedDB and falls back to memory.

```ts
import {
  createInMemoryFsProvider,
  createRemoteFsProvider,
  initRunMat
} from "runmat";

const session = await initRunMat({
  fsProvider: createInMemoryFsProvider()
});

const remoteSession = await initRunMat({
  fsProvider: createRemoteFsProvider({
    baseUrl: "https://api.runmat.com",
    authToken: process.env.RUNMAT_API_TOKEN
  })
});
```

Common providers:

| Provider | Use |
| --- | --- |
| `createInMemoryFsProvider` | Tests, examples, and ephemeral sessions. |
| `createIndexedDbFsHandle` / `createIndexedDbFsProvider` | Browser persistence. |
| `createDefaultFsProvider` | IndexedDB with memory fallback. |
| `createRemoteFsProvider` | HTTP-backed workspaces, object stores, and hosted filesystems. |

The provider contract covers reads, writes, metadata, directory listing, deletion, rename, directory creation, and readonly metadata used by MATLAB-compatible I/O builtins.

`createInMemoryFsProvider({ initialFiles })` seeds a ready-to-use tree, creating parent directories for nested seed paths. Later `writeFile` calls still require an existing parent directory; use `createDirAll` explicitly when needed.

The optional `renameReplace(from, to)` hook supports forced `movefile` replacement. Implement it as one provider-owned operation: reject unsupported replacement, incompatible types, and nonempty destination directories before mutation, and preserve both paths on failure. Do not implement it by deleting the destination and then calling `rename`. The built-in memory provider supports compatible file and empty-directory replacement; a host without this hook reports unsupported replacement without calling its ordinary rename or delete hooks. The remote provider does not currently expose this operation.

`MemoryVolume.createProvider(onMutate, onObserverError)` accepts synchronous post-commit notifications. Observer errors go to `onObserverError`, or to best-effort console reporting when no sink is supplied or the sink fails; they do not turn a committed mutation into a failed filesystem operation. IndexedDB persistence remains deferred: successful in-memory replacement does not mean its snapshot has been durably flushed. Use the IndexedDB handle's `flush()` when persistence completion is required.

## Package Cache

Browser sessions use a separate transactional IndexedDB package cache by default. Package state and content payloads commit atomically, concurrent tabs use revision conflicts instead of overwriting one another, and eviction is reported as a recoverable cache miss. Persistent-storage permission is optional and is never required for correctness.

```ts
import { createIndexedDbPackageCache, initRunMat } from "runmat";

const cache = await createIndexedDbPackageCache({
  dbName: "my-runmat-packages",
  requestPersistence: true
});

const session = await initRunMat({
  packageCache: { provider: cache.provider }
});

const status = await session.packageCacheSnapshot();
session.dispose();
cache.close();
```

`ImmutableBrowserPackageMount` provides a read-only projection over a Rust-validated tree manifest. It verifies payload length and SHA-256 on every cache read and reports an evicted object with `code: "PackageCacheMiss"`.

## Plotting

For a single plot surface, pass a canvas at initialization:

```ts
const canvas = document.querySelector<HTMLCanvasElement>("#plot")!;

const session = await initRunMat({
  plotCanvas: canvas
});

await session.executeRequest({
  source: {
    kind: "text",
    name: "<plot>",
    text: "x = 0:0.1:10; plot(x, sin(x))"
  }
});
```

Advanced hosts can manage multiple canvases with the exported plotting helpers:

```ts
import {
  bindSurfaceToFigure,
  createPlotSurface,
  presentSurface,
  renderFigureImage
} from "runmat";

const surfaceId = await createPlotSurface(canvas);
await bindSurfaceToFigure(surfaceId, 1);
await presentSurface(surfaceId);

const pngBytes = await renderFigureImage({ handle: 1, width: 640, height: 480 });
```

Use `onFigureEvent` to track figure creation, updates, clears, and closes. Use `exportFigureScene` and `importFigureScene` to persist or replay plot scenes.

## Streams And Interaction

Hosts can stream output before `executeRequest` resolves:

```ts
import { subscribeStdout, unsubscribeStdout } from "runmat";

const subscription = await subscribeStdout((entry) => {
  terminal.write(entry.text);
});

await unsubscribeStdout(subscription);
```

Interactive prompts are routed through a session input handler:

```ts
await session.setInputHandler(async (request) => {
  if (request.kind === "line") {
    return window.prompt(request.prompt) ?? "";
  }
  return { kind: "keyPress" };
});
```

`session.cancelExecution()` requests cooperative cancellation for a running request.

## Browser Worker Execution

An embedding host supplies isolated browser workers through `initRunMat({ executionHost })`. The host transports requests and responses; the Rust runtime owns task scheduling, callable selection, resource admission, and result validation. Without a worker host, a session does not advertise isolated worker execution.

For selected-program execution, the host provides `workerInventory`, `launchProgram`, and `cancelProgram`, and sets `capabilities.selectedProgramWork` to `true`. Obtain the inventory from the actual initialized worker with `workerProgramInventory()`, using the same pinned SDK and WASM assets as the coordinating session. Create each worker invocation with `createWorkerProgramInvocation(request.program)`. Its assignment identifies the exact task, attempt, worker, and resource leases; cancellation must carry that same assignment.

An invocation's `run()` returns a `RunMatProgramTaskTerminal`. Preserve that terminal, including originating runtime errors and its retirement state. A fulfilled or rejected transport Promise, a cancellation request, and `Worker.terminate()` are not evidence of Runtime retirement. Reuse a worker slot only after its invocation has retired; retain or quarantine unresolved work instead of replacing it with another task. The stock CPU worker admits only the host and device services its factory actually provides. It does not inherit the coordinating session's filesystem, network, foreign runtimes, or GPU resources.

`session.cancelExecution()` also requests cancellation of the session's child-task scope. Caller cancellation can complete before a worker's retirement response arrives, so the execution host must remain available to receive that response. Interpreted browser loops and supported numerical kernels yield through the adapter's cooperative task-turn checkpoints, allowing worker cancellation messages to be delivered. Cancellation does not preempt an arbitrary synchronous foreign call or device kernel.

## Diagnostics

Use `session.gpuStatus()` to inspect whether GPU acceleration was requested and whether it is active:

```ts
const gpu = session.gpuStatus();

if (!gpu.active && gpu.error) {
  console.warn(gpu.error);
}
```

The package also exposes runtime log and trace subscriptions with `subscribeRuntimeLog`, `setLogFilter`, and `subscribeTraceEvents`.

## Lifecycle

Each `RunMatSessionHandle` owns runtime state, workspace values, input handlers, filesystem callbacks, and plot resources. Call `dispose()` when tearing down a REPL, notebook kernel, worker, or editor pane.

```ts
try {
  await session.executeRequest({
    source: { kind: "text", name: "<cleanup>", text: "clear" }
  });
} finally {
  session.dispose();
}
```

## Your OpenTelemetry collector

Configure export before subscribing to log streams or creating plot surfaces.
Browser collectors must support OTLP HTTP/protobuf and permit the app's origin,
`POST`, `content-type` and your authorization headers in CORS preflight. HTTPS
apps require an HTTPS collector, and CSP `connect-src` must allow its origin.

```ts
import { initRunMat, sendOpenTelemetryTest, flushOpenTelemetry } from "runmat";

const session = await initRunMat({
  runtimeSettings: {
    userSettingsJson: JSON.stringify({
      logging: { otel: {
        enabled: true,
        endpoint: "https://collector.example.com:4318",
        protocol: "http/protobuf",
        signals: ["traces", "logs"],
      } },
    }),
  },
  // Runtime-only host input: do not save these headers in settings documents.
  openTelemetryHeaders: { authorization: `Bearer ${collectorToken}` },
  telemetryConsent: false, // Product telemetry is independent of your collector.
});
await sendOpenTelemetryTest(); // Real correlated span+log, acknowledged by collector.
await flushOpenTelemetry();
```

The awaited test/flush calls reject on disabled export, queue overflow, network,
CORS/TLS, HTTP/gRPC status, malformed response or collector partial rejection;
success confirms acceptance, not downstream indexing. Browser errors cannot
distinguish CORS from every network/TLS failure. `openTelemetryStats()` exposes
drop/failure/export counts without credentials. gRPC and metrics are not supported
in the browser and are rejected explicitly.

Sessions share one exporter per WASM runtime. Repeating identical collector
settings and headers reuses it; changing them, enabling after another API already
initialized logging, or reopening after exporter shutdown returns
`OpenTelemetryRestartRequired`. Restart the browser worker/runtime to apply those
changes. Await `shutdownOpenTelemetry()` before terminating that shared runtime,
not after each session; `session.dispose()` does not stop other sessions' exporter.
Abrupt tab or worker termination cannot guarantee final delivery.

## Development

```bash
cd bindings/ts
npm install
npm run build
npm test
```

The build expects `wasm-pack` and a Rust toolchain with the `wasm32-unknown-unknown` target installed.

## License

Business Source License 1.1 with the RunMat Free Use grant. See [LICENSE.md](./LICENSE.md) and [FREE-USE.md](./FREE-USE.md). Each version becomes available under Apache License 2.0 three years after first publication; releases through 0.6.2 retain their Apache terms.
