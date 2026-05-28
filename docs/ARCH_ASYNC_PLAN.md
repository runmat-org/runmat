## Async architecture rollout plan (working document)

### Purpose

This is the **working plan** for implementing the async/futures architecture described in
`docs/ARCH_ASYNC.md`. It is meant to be updated continuously as work progresses.

### Guiding principles

- Prefer **typed errors** over sentinel strings.
- Use **async/await** instead of control‑flow enums.
- Keep crates layered: no Tokio/JS deps leaking into VM/runtime crates.
- Keep MATLAB‑compat “sync code” fast.
- Builtins remain **language‑synchronous**; concurrency is explicit via futures, `spawn`, and `await`.

---

## Current status snapshot

### Finalized baseline

- Execution is **async end‑to‑end** (`RunMatSession::execute` and the VM are async).
- Builtins return `Result<T, RuntimeError>` with structured diagnostics.
- `RuntimeControlFlow`, pending frames, and resume loops are removed.
- WASM input handlers are Promise‑backed and return typed responses.
- GPU map/readback currently returns a **runtime error** when pending; future awaitables are
  planned to remove this limitation.

---

## Work breakdown (phases)

### Phase 1: async execution + typed errors (done)

**Goal**: remove sentinel control flow and move to async + `RuntimeError`.

**Exit criteria**
- Interpreter is async.
- Builtins return `RuntimeError` directly.
- No control‑flow enum or resume loops remain.

---

### Phase 2: internal awaitables (GPU readback)

**Goal**: replace the current “pending readback” error with awaitable GPU futures.

- Add `WgpuReadbackFuture` in the accelerate provider.
- Await map/readback completion inside GPU paths.

**Exit criteria**
- WebGPU map/readback no longer returns a “pending” error.

---

### Phase 3: external awaitables (stdin/UI)

**Goal**: input uses awaitables rather than immediate handler responses when desired.

- Add an `InputAwaitable` and host fulfillment path.
- Preserve synchronous handlers as a fast path.

**Exit criteria**
- Input can be awaited without blocking the event loop.

---

### Phase 4: language‑level async/await

**Goal**: add RunMat language constructs.

- Parser/AST/HIR additions for:
  - async functions and async blocks that produce lazy futures
  - `await(expr)` as an explicit suspension point
  - `spawn(expr)` as the explicit scheduling/concurrency boundary
- Bytecode opcodes:
  - `FUTURE_CREATE`, `SPAWN`, `AWAIT`
- Value representation:
  - `Value::Future(...)`
  - `Value::Task(TaskHandle)`
  - optionally `Value::Awaitable(...)` for native/internal awaitables

Semantics:
- creating a future does not execute user code
- `await` polls a future/task to completion
- `spawn` starts concurrent execution and returns a task handle
- ordinary MATLAB-style code remains sequential and language-synchronous
- direct `await` without `spawn` does not introduce concurrency
- `spawn` requires spawn-safe captures and values
- mutable lexical captures from the spawning frame are rejected unless wrapped in explicit synchronized/runtime-managed handles

**Exit criteria**
- Minimal async programs run deterministically.

---

### Phase 5: cleanup + removal of transitional code

**Goal**: remove any remaining scaffolding and keep only async + `RuntimeError`.

**Exit criteria**
- No legacy control‑flow scaffolding remains.

---

## Tracking checklist (update as we go)

- [x] async `execute` end‑to‑end
- [x] builtins return `RuntimeError`
- [x] control‑flow enum removed
- [ ] awaitable GPU readback
- [ ] awaitable stdin
- [ ] language async/await

---

## Open questions (keep updated)

- **Top-level await policy**: REPL, notebook, and script-like entrypoints should permit it by default; confirm whether batch/function-file entry modes need additional host policy.
- **GC rooting API**: best representation for cross‑await handles
- **Spawn-safe shared handles**: exact user-facing synchronized/runtime-managed objects for intentional shared mutable state.
- **Cancellation semantics**: how to expose and integrate with host cancel
