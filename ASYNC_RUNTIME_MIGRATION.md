# Async Runtime Migration Plan

This document captures the current migration plan for moving RunMat’s runtime from the temporary `RuntimeControlFlow`/string‑sentinel model to a fully async, Rust‑native `Future<Result<_, RuntimeError>>` execution model. It includes an inventory of remaining “hacky” control‑flow and error surfaces, and a concrete sequence of work to eliminate them.

## Goals

- **End state**: the interpreter and builtin execution path is fully async (Rust futures), returning `Result<T, RuntimeError>` without any control‑flow enum or string sentinel tricks.
- **RuntimeControlFlow is temporary**: it exists only to bridge the old sync interpreter and the new async model. The target is to remove it entirely.
- **Error quality**: all errors should be `RuntimeError` (structured, rich diagnostics), and stringification should occur **only** at the outer UI/CLI boundary.
- **Executor agnostic**: the async execution path should work equally under WASM (JS event loop) and native (Tokio or any executor).

## Why this matters

1. **Event loop starvation**: wasm/WebGPU operations (e.g., `async_map`) can’t yield cleanly while the interpreter is sync, so the event loop is starved.
2. **Ergonomics for Rust authors**: futures + typed errors make it easier to write and extend builtins using Rust idioms.
3. **High‑quality diagnostics**: using `RuntimeError` throughout preserves identifiers, context, spans, and chaining.
4. **Compatibility with Rust ecosystem**: async + `Result` integrates cleanly with async ecosystems and tooling.

## Current temporary model

- **`RuntimeControlFlow`**: enum carrying either `Suspend(PendingInteraction)` or `Error(RuntimeError)`.
- **Suspend/Resume**: the interpreter throws “suspend” as a control‑flow return, stores state, and later resumes.
- **String edges**: some surfaces still return `Result<_, String>` or convert errors to strings.

## Target model

- All interpreter execution becomes `async fn` and returns `Result<_, RuntimeError>`.
- Builtins that must yield become `async fn` and `await` instead of “suspend.”
- The existing interpreter/execute entry points are converted to async (no parallel sync API).
- `RuntimeControlFlow`, `PendingInteraction`, resume loops, and string sentinels are removed.

## Inventory: remaining control‑flow + string error surfaces

### RuntimeControlFlow in the interpreter

- `crates/runmat-ignition/src/vm.rs`
  - `RuntimeControlFlow` types, `vm_bail!`, suspend handling.
  - `resume_with_state` and `PendingInteraction` plumbing.

### RuntimeControlFlow in core

- `crates/runmat-core/src/lib.rs`
  - `interpret_with_context` uses `Result<_, RuntimeControlFlow>`.
  - async input handling returns `Result<InputResponse, String>`.
  - pending‑frame resume logic wired to `PendingInteraction`.

### RuntimeControlFlow in runtime + builtins

- `crates/runmat-runtime/src/interaction.rs`
  - `request_line` and friends return `RuntimeControlFlow::Suspend`.
- `crates/runmat-runtime/src/dispatcher.rs`
  - Explicit `Suspend` construction.
- `crates/runmat-runtime/src/builtins/io/mat/save.rs`
  - Direct `Suspend(PendingInteraction)` paths.
- Plotting/strings builtins pattern‑match or return `RuntimeControlFlow`.

### String error surfaces (non‑exhaustive)

- `crates/runmat-runtime/src/lib.rs`: helpers return `Result<_, String>`.
- `crates/runmat-runtime/src/matrix.rs`: math helpers return `Result<_, String>`.
- `crates/runmat-runtime/src/builtins/io/filetext/*`: large surface of `Result<_, String>` helpers.
- `crates/runmat-wasm/src/lib.rs`: `js_input_request` returns `Result<_, String>`.
- `crates/runmat-core/src/lib.rs`: async input handler uses `Result<_, String>`.

## Migration checklist (sequence)

### Phase 1 — Define the new async execution boundary

1. **Convert interpreter entry points to async**
   - Convert the existing `interpret`/`execute` entry points in ignition/core to `async fn` returning `Result<InterpreterOutcome, RuntimeError>`.
   - Remove the sync wrappers rather than introducing parallel async APIs.

2. **Async builtin call chain**
   - Replace `runmat_runtime::call_builtin` with `async fn call_builtin_async(...) -> Result<Value, RuntimeError>`.
   - Convert call sites in ignition VM to `await`.

3. **Make interpreter loop async**
   - VM stepper becomes `async fn`, so any builtin call can `await`.
   - Replace `Suspend` with `await` points; no `PendingInteraction` escapes the VM.

### Phase 2 — Remove `RuntimeControlFlow` and suspend/resume

4. **Delete `RuntimeControlFlow` from VM paths**
   - Remove `vm_bail!` suspend logic, `resume_with_state`, `PendingInteraction` framing.

5. **Remove runtime `interaction` suspend model**
   - `request_line`/`wait_for_key` become async functions returning `Result<_, RuntimeError>`.
   - Interaction handlers return futures instead of immediate `Suspend`.
   - Remove any old interaction logic.

6. **Core session uses async interaction directly**
   - `Session` installs async input handler returning `Result<InputResponse, RuntimeError>`.
   - Remove pending‑frame state from core.

### Phase 3 — Replace string error surfaces

7. **Convert helpers to `RuntimeError`**
   - For each `Result<_, String>` helper, convert to `Result<_, RuntimeError>`.
   - Use `build_runtime_error(...)` or typed error enums with `thiserror` + `?`.

8. **Remove string adapters in runtime**
   - Only the CLI/UI entry points convert errors to strings.
   - `RuntimeError` stored, surfaced, and logged directly.

### Phase 4 — wasm/webgpu event loop integration

9. **Make wasm execution await interpreter**
   - `runmat-wasm` awaits the now‑async `execute` entry point directly; no resume loop.

10. **WebGPU async_map is just async**
    - `map_async` in wgpu provider returns a future awaited by builtins.
    - Remove any “suspend to event loop” special cases.

### Phase 5 — Cleanup and removal

11. **Delete `runmat-async` suspend structs**
    - Remove `PendingInteraction`, `SuspendMarker`, and any “suspend” handling code.

12. **Remove compatibility shims**
    - Remove `flow_to_string`, `RuntimeControlFlow` conversions, and any string sentinels.

## Practical notes

- **Keep adapters at the edges**: CLI/JS/FFI should handle string conversion or formatting for UI.
- **Use structured error context**: keep identifier/message/stack in `RuntimeError` and surface these in the UI layer.
- **Use `async` everywhere internally**: avoid `block_on` inside runtime or builtins; the core entry points should be async with no sync shim.

## Immediate next actions (recommended)

1. Convert existing interpreter/execute entry points to async.
2. Convert `runmat_runtime::call_builtin` to async and update VM call sites.
3. Remove `RuntimeControlFlow` from VM and core once async path works end‑to‑end.

---

When you resume work, start by re‑scanning for `RuntimeControlFlow` and `PendingInteraction` usage, and work down the checklist. This plan intentionally prioritizes **removing control‑flow hacks** and **establishing a clean async path** before sweeping every remaining `String` helper.
