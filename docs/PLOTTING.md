# Plotting in RunMat

RunMat plotting aims to be **MATLAB-familiar** while also being **GPU-first** and **event-loop friendly** (especially on Web/WASM).

The core design principle is:

- **Plotting builtins mutate figure state.**
- **Presentation (rendering to a surface) is coalesced and throttled.**
- **`drawnow()` (and `pause()`) are explicit “yield + present” boundaries.**

This keeps semantics clean for modelling/physics code while unlocking smooth GPU-resident animation.

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

### Current controls

- **Rotate (3D)**: left-drag
- **Pan**: right-drag
- **Zoom**: mouse wheel

### 3D correctness

3D camera-based rendering uses a depth attachment so surfaces/meshes/3D scatter occlude
correctly during interaction.
