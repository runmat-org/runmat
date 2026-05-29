---
title: "WASM & TypeScript/JavaScript"
category: "WebAssembly & TypeScript"
section: "9.0"
last_updated: "May 28, 2026"
---

# WASM & TypeScript/JavaScript

Use the `runmat` npm package when embedding RunMat in a browser, web worker, Electron app, or Node-based tool.

```bash
npm install runmat
```

The public entrypoint is `initRunMat`. It loads the WASM module, resolves optional startup assets, creates a `RunMatSessionHandle`, and returns an object for executing MATLAB-compatible source.

## First Execution

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

`executeRequest` accepts either inline source or a path resolved through the configured filesystem provider:

```ts
await session.executeRequest({
  source: { kind: "path", path: "src/main.m" }
});
```

## Initialization Options

`initRunMat` options fall into a few practical groups.

| Group | Options | Purpose |
| --- | --- | --- |
| Runtime | `snapshot`, `enableJit`, `language.compat` | Control startup state, native-code tiering, and MATLAB compatibility mode. |
| GPU | `enableGpu`, `wgpuPowerPreference`, `wgpuForceFallbackAdapter` | Request WebGPU acceleration and adapter preferences. |
| Filesystem | `fsProvider` | Provide file I/O for `load`, `save`, scripts, and path execution. |
| Plotting | `plotCanvas`, `scatterTargetPoints`, `surfaceVertexBudget` | Attach a canvas and tune plotting LOD defaults. |
| Diagnostics | `telemetryConsent`, `telemetryId`, `logLevel`, `emitFusionPlan`, `callstackLimit`, `errorNamespace` | Configure logging, telemetry, error detail, and optional fusion-plan payloads. |

GPU is requested by default, but the wrapper checks for `navigator.gpu` and falls back to CPU execution when WebGPU is unavailable.

## Results

`executeRequest` resolves to an `ExecuteResult`.

| Field | Use |
| --- | --- |
| `flow`, `valueText`, `valueJson` | Returned value or output-list information. |
| `displayEvents` | Values displayed during execution. |
| `stdout` | Buffered stdout/stderr stream entries. |
| `workspace` | Workspace snapshot with names, types, shapes, residency, previews, and preview tokens. |
| `figuresTouched` | Figure handles changed by the request. |
| `warnings` | Structured MATLAB warning entries. |
| `stdinEvents` | Prompt/response transcript for interactive input. |
| `error` | Structured syntax, semantic, compile, or runtime error details. |
| `profiling`, `fusionPlan` | Optional performance and acceleration diagnostics. |

Execution errors are exposed as structured result data and are also coerced into useful JavaScript errors on rejected wrapper calls.

## Workspace Inspection

Each workspace entry includes class, shape, dtype, residency, small previews, and an optional `previewToken`. Hosts can use those entries to build variable panes without asking MATLAB to print variables.

```ts
const entry = result.workspace.values.find((value) => value.name === "A");

if (entry?.previewToken) {
  const full = await session.materializeVariable(entry.previewToken, {
    limit: 4096
  });
  console.log(full.valueText);
}
```

`materializeVariable` can also use `{ name }` or `{ previewToken, name }` selectors. Slice options are available for large arrays.

## Filesystem Providers

RunMat file I/O is backed by a JavaScript filesystem provider. If no provider is passed, the wrapper creates a default provider that tries IndexedDB and falls back to memory.

```ts
import {
  createInMemoryFsProvider,
  createIndexedDbFsHandle,
  createRemoteFsProvider,
  initRunMat
} from "runmat";

const session = await initRunMat({
  fsProvider: createInMemoryFsProvider()
});
```

Common choices:

| Provider | Use |
| --- | --- |
| `createInMemoryFsProvider` | Tests, examples, and ephemeral sessions. |
| `createIndexedDbFsHandle` / `createIndexedDbFsProvider` | Browser persistence. |
| `createDefaultFsProvider` | IndexedDB with memory fallback. |
| `createRemoteFsProvider` | HTTP-backed file operations for hosted workspaces or object stores. |

The provider contract covers reads, writes, metadata, directory listing, deletion, rename, and related file operations used by MATLAB-compatible I/O builtins.

## Streams And Interaction

Hosts can stream output before `executeRequest` resolves:

```ts
const id = await import("runmat").then(({ subscribeStdout }) =>
  subscribeStdout((entry) => {
    terminal.write(entry.text);
  })
);
```

For interactive prompts, register an input handler on the session:

```ts
await session.setInputHandler(async (request) => {
  if (request.kind === "line") {
    return window.prompt(request.prompt) ?? "";
  }
  return { kind: "keyPress" };
});
```

`session.cancelExecution()` cooperatively interrupts a running request. Use `session.dispose()` when tearing down a REPL, editor pane, worker, or notebook kernel.

## Plotting

For simple single-canvas plotting, pass a canvas during initialization:

```ts
const canvas = document.querySelector<HTMLCanvasElement>("#plot")!;

const session = await initRunMat({
  plotCanvas: canvas,
  telemetryConsent: false
});

await session.executeRequest({
  source: { kind: "text", name: "<plot>", text: "x = 0:0.1:10; plot(x, sin(x))" }
});
```

Advanced hosts can use exported helpers such as `createPlotSurface`, `bindSurfaceToFigure`, `presentSurface`, `getPlotSurfaceCameraState`, `renderFigureImage`, `exportFigureScene`, and `importFigureScene` to support multiple canvases, saved camera state, thumbnails, and figure replay.

## Fusion And Diagnostics

Set `emitFusionPlan: true` during initialization, or call `session.setFusionPlanEnabled(true)`, when a UI needs acceleration-plan details. The package also exports `createFusionPlanAdapter` for inspector panes that should subscribe to plan changes only while visible.

`session.gpuStatus()` reports whether GPU acceleration was requested, whether it is active, and which adapter was selected. `session.memoryUsage()` reports the current WebAssembly heap size.

## Lifecycle

Each `RunMatSessionHandle` owns runtime state, workspace values, input handlers, and any session-level resources. Call `dispose()` when the surrounding UI is unmounted.

```ts
try {
  await session.executeRequest({
    source: { kind: "text", name: "<cleanup>", text: "clear" }
  });
} finally {
  session.dispose();
}
```
