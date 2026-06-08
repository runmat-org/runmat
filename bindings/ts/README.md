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
| `snapshot` | Preload a runtime snapshot from bytes, a URL, a stream, or a custom fetcher. |
| `enableGpu` | Request WebGPU acceleration. Defaults to GPU when WebGPU is available, with CPU fallback otherwise. |
| `enableJit` | Enable the JIT tier where supported. |
| `fsProvider` | Provide file I/O for `load`, `save`, scripts, datasets, and path execution. |
| `plotCanvas` | Attach a default plotting canvas during startup. |
| `telemetryConsent`, `telemetryId`, `telemetryEmitter` | Configure telemetry before the runtime starts. |
| `logLevel`, `emitFusionPlan`, `callstackLimit`, `errorNamespace` | Configure diagnostics and execution metadata. |
| `language.compat` | Choose default MATLAB-compatible behavior or `"strict"` mode. |

GPU initialization is opportunistic in browser contexts. If WebGPU is unavailable, RunMat logs a warning and continues on CPU.

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

## Development

```bash
cd bindings/ts
npm install
npm run build
npm test
```

The build expects `wasm-pack` and a Rust toolchain with the `wasm32-unknown-unknown` target installed.

## License

Apache License 2.0.
