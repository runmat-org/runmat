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
