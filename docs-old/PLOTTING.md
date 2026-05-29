# Plotting in RunMat

RunMat plotting aims to be **MATLAB-familiar** while also being **GPU-first** and **event-loop friendly** (especially on Web/WASM).

The core design principle is:

- **Plotting builtins mutate figure state.**
- **Presentation (rendering to a surface) is coalesced and throttled.**
- **`drawnow()` (and `pause()`) are explicit “yield + present” boundaries.**

This keeps semantics clean for modelling/physics code while unlocking smooth GPU-resident animation.

## 3D depth + clipping

For details on depth modes (standard vs reversed‑Z), clip plane policies, and the 3D grid helper,
see the internal depth and clipping notes (not yet published as a standalone page).

## Semantics: state vs presentation

### `plot`, `scatter`, `hist`, …

- These builtins **update the current figure/axes state** (respecting `figure`, `gcf`, `subplot`, `hold`, etc.).
- They emit a **figure event** (`created` / `updated` / `closed`) so the host can update its UI and schedule presentation.
- They **do not guarantee that pixels change immediately** on Web/WASM. If a script does not yield, the browser cannot present frames.

### `drawnow()`

`drawnow()` is the explicit boundary that means:

- **Present the latest revision** of the current figure to any bound surface(s).
- **Yield** so the host/browser can process rendering work.

This mirrors MATLAB’s idiom and gives precise control for simulation loops:

```matlab
for t = 0:dt:T
    Y = sin(X + t);
    plot(X, Y);
    drawnow();
end
```

### `pause(dt)`

`pause` is treated as a modelling-friendly equivalent of “yield the UI”:

- `pause(dt)` performs a **`drawnow()` first**, then waits approximately `dt` seconds (cancellable).
- `pause(0)` is a useful idiom meaning **“draw once and yield”**.

This matches MATLAB intent while remaining correct on Web/WASM targets where blocking sleeps are not allowed.

## Host rendering model (Web/WASM)

On Web/WASM, rendering is fundamentally cooperative:

- The runtime executes in a Web Worker.
- The plot surface is an `OffscreenCanvas` transferred once to the worker.
- The host should treat figure events as “dirty notifications” and **present at most once per animation frame**.

Practically:

- On figure `updated`, enqueue the handle in a “dirty set”.
- On the next `requestAnimationFrame`, ask the worker to render the current scene for those handles.

This coalescing prevents redundant work when the runtime emits many updates per second.

## GPU-first data path

Plotting is designed to avoid host round-trips when data is already on a GPU provider:

- Runtime tensors on a GPU provider can be exported via `runmat_accelerate_api::export_wgpu_buffer`.
- `runmat-plot` GPU packers (`crates/runmat-plot/src/gpu/**`) can then construct vertex buffers directly on-device.
- CPU fallbacks remain available for unsupported configurations or when GPU export is unavailable.

## Where to look next

- Runtime builtins: `crates/runmat-runtime/src/builtins/plotting/**`
- `drawnow` + presentation hooks: `crates/runmat-runtime/src/builtins/plotting/core/web.rs`
- Plot surface attachment (desktop/web host): `runmat-private/desktop/src/runtime/graphics/figure-canvas-adapter.ts`
- Host scheduling of presents: `runmat-private/desktop/src/runtime/runtime-provider.tsx`
- GPU compute backend notes: `docs/GPU_BEHAVIOR_NOTES.md`

## Interaction + camera (Web/WASM)

Interactive camera control is supported on Web/WASM by forwarding pointer/wheel events
from the main thread into the worker/WASM renderer. This is intentionally **separate**
from figure mutation:

- **Figure updates** (data changes) still follow the figure-event coalescing model.
- **Camera updates** (pan/zoom/rotate) trigger a **camera-only re-render** on the bound surface,
  even if the figure revision did not change.
- Hosts can snapshot/restore camera state per surface using:
  - `getPlotSurfaceCameraState(surfaceId) -> PlotSurfaceCameraState | null`
  - `setPlotSurfaceCameraState(surfaceId, state) -> void`

This is useful for preserving camera continuity when UIs rebind figures across sessions/tabs
or virtualize plot surfaces for notebook-style layouts with many concurrent figures.

### Current controls

- **Rotate (3D)**: left-drag
- **Pan**: right-drag
- **Zoom**: mouse wheel

### 3D correctness

3D camera-based rendering uses a depth attachment so surfaces/meshes/3D scatter occlude
correctly during interaction.

## Figure scene replay payloads

RunMat now exposes a versioned figure-scene replay payload at the wasm boundary:

- `exportFigureScene(handle) -> Uint8Array | null`
- `importFigureScene(sceneBytes) -> number | null`

The payload is JSON and self-describing (schema version + kind) so hosts can persist it as an
artifact and hydrate later without coupling to renderer internals.

Scene replay roundtrip currently reconstructs these plot families directly:

- 2D: `line`, `scatter`, `errorbar`, `stairs`, `stem`, `area`
- 3D: `surface`, `scatter3`

Hosts should persist scene artifacts whenever `exportFigureScene` returns bytes and use PNG
preview artifacts only as a fallback display path.

For large 3D payloads (for example, dense `surface` grids and large `scatter3` point/color
arrays), persistence externalizes numeric buffers into dataset-style chunked blobs and
stores typed refs inside the scene JSON (`runmat-data-array-v1`).

Replay rehydration resolves refs by stable chunk identity (`artifactId`) with `src` as a hint,
then imports a hydrated scene payload. This keeps replay robust across provider root/cwd
differences while preserving chunked storage scalability.

Implementation boundaries:

- Runtime core (`runmat-runtime`) owns scene schema validation, import safety limits, and scene reconstruction.
- Shared bindings helper (`bindings/ts/src/replay/scene-resolver.ts`) owns provider-aware ref rehydration orchestration.
- Host apps (desktop/web) only provide filesystem reads and invoke scene import; they do not implement plot-kind hydration logic.

Performance sanity check (native runtime debug profile):

- Command: `cargo test -p runmat-runtime replay::scene::tests::bench_scene_ref_hydration_large_surface -- --ignored --nocapture`
- Sample local result: `512x512` surface scene decode+rehydrate in about `165 ms`.
- This benchmark is informational (ignored by default) and intended for regression spot checks, not hard pass/fail thresholds.

Runtime-level limits are enforced during decode/import:

- maximum payload bytes
- maximum plot object count

Invalid schema, oversized payloads, and rejected imports return `null` at the wasm/TS boundary.

## Native surface theming

Headless/native-surface interactive render exports support explicit theme injection.
Hosts can pass a full `PlotThemeConfig` to keep interactive plot colors aligned with
UI theme changes (including light/dark presets and custom overrides) without rerunning
the underlying script.

The same theme payload is supported by the wasm/web worker surface path so active
WebGPU plot surfaces can switch themes live while attached.

Theme application now covers overlay plot chrome consistently (frame, axes ticks/labels,
grid, legend, title, and axis labels). The default `classic_light` theme is tuned for
stronger contrast on bright backgrounds while preserving the existing dark preset.

Background fill now follows an explicit policy:

- **Theme-driven** when figure background is unspecified/default.
- **Explicit override** when a figure sets a background color directly.

This policy is applied consistently across Web/WASM and native-surface render paths to
avoid dark-default leakage during first render and theme switches.

---

## Related

- [Plotting in RunMat](/docs/plotting/plotting-in-runmat) -- the plotting workflow from first command to finished figure.
- [Plot Replay and Export](/docs/plotting/plot-replay-and-export) -- persist, replay, and export figures.
- [Graphics Handles](/docs/plotting/graphics-handles) -- inspect and update plot objects with handles.
- [GPU Residency and Precision](/docs/accelerate/gpu-behavior) -- when data moves to and from the GPU.
