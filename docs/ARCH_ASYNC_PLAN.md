## Async architecture rollout plan (working document)

### Purpose

This is the **working plan** for implementing the async/futures architecture described in
`docs/ARCH_ASYNC.md`. It is meant to be updated continuously as work progresses.

### Guiding principles

- Prefer **typed control-flow** over sentinel strings.
- Prefer **waker-driven** waiting over polling loops.
- Keep crates layered: no Tokio/JS deps leaking into VM/runtime crates.
- Keep MATLAB-compat “sync code” fast.
- Builtins remain **language-synchronous**; concurrency is explicit via tasks + `await`.

---

## Current status snapshot

### Transitional mechanism (temporary)

**As of `feat-async` (compile-green):**

- Evaluation is now **poll-driven** via `ExecuteFuture` (Phase 2 directionally complete).
- The legacy **string sentinel** suspension mechanism is removed; suspension is represented as
  **typed control-flow** (`runmat_async::RuntimeControlFlow::Suspend(PendingInteraction)`).
- WASM no longer requires any host “resume loops” for stdin; input is **Promise-driven** via
  `setInputHandler` + typed UI responses.
- WebGPU readback waiting is **waker-driven** (no polling loop), via a GPU map/readback waker hook.

Remaining transitional glue is limited to a small set of legacy `Result<_, String>` call stacks
where typed control-flow may still need to bubble through string-returning helpers; this surface is
being retired incrementally as helpers/builtins are migrated to `BuiltinResult<T>`.

### Target end state

- `RunMatSession::execute_async` returns a `Future`/`Promise` backed by `ExecuteFuture`.
- VM yields via `Poll::Pending` at await points, not via error strings.
- Host runtime drives execution through an executor adapter.

---

## Work breakdown (phases)

### Phase 0: doc + alignment (done when design is accepted)

- Ensure `ARCH_ASYNC.md` is correct and agreed upon.
- Decide if we will introduce `runmat-async` as a new crate and merge `runmat-control-flow` into it.

**Exit criteria**
- Design sign-off.

---

### Phase 1: introduce `runmat-async` crate and typed control-flow (no language syntax yet)

**Goal**: establish the shared async substrate and eliminate stringly-typed suspension.

- Create crate `runmat-async`
  - `TaskId`, `TaskHandle`
  - minimal executor trait (host-neutral)
  - awaitable wrapper types
  - deterministic local executor for tests (optional now; required before heavy testing)
- Move/merge existing `runmat-control-flow` types into `runmat-async` (and delete old crate).
- Replace any “pending interaction sentinel string” control-flow with typed control-flow:
  - no `__RUNMAT_PENDING_INTERACTION__` in the final architecture
  - define a typed `YieldReason` / `PendingRequest` model in `runmat-async`

**Exit criteria**
- Core crates compile with `runmat-async`.
- No “pending as string” used as a control-flow boundary in core runtime execution.

---

### Phase 2: `ExecuteFuture` (interpreter becomes a Future)

**Goal**: make evaluation pollable and waker-driven; still no user-facing `async/await` syntax.

- Add `RunMatSession::execute_async(...) -> ExecuteFuture`
  - `ExecuteFuture: Future<Output = Result<ExecutionResult, RunMatError>>`
  - owns VM state and roots across yields
- Modify ignition to be driven by poll:
  - `poll_execute` runs until completion/error/await-pending
- Build a minimal `LocalExecutor` and `WasmExecutorAdapter`
  - native: simple single-thread executor for tests and CLI
  - wasm: adapter that integrates with JS event loop and wakers

**Exit criteria**
- `execute_async` works in native tests and in wasm (via Promise), even if no async ops exist yet.

---

### Phase 3: internal awaitables (GPU map/readback first)

**Goal**: convert WebGPU map/readback to internal awaitables (no host polling loops).

- Add `GpuReadbackAwaitable` (or `WgpuMapFuture`) inside accelerate/wgpu provider:
  - registers waker
  - wakes on map callback
  - provides mapped bytes to decode path
- Update provider download path to return/await this awaitable internally.

**Exit criteria**
- WebGPU readback no longer uses any host “resume” loop or sentinel strings.
- The original map_async hang is solved by construction (no event-loop starvation).

---

### Phase 4: external awaitables (stdin/UI)

**Goal**: stdin uses awaitables rather than “pending requests” plumbing.

- Implement an `InputAwaitable` (oneshot-like):
  - created by runtime when input is required
  - host fulfills it
  - wake resumes execution
- Update JS/desktop bindings:
  - surface external pending awaits to UI
  - fulfill via a typed response

**Exit criteria**
- Input no longer uses special “resume” APIs; it’s just completing an awaitable.

---

### Phase 5: language-level async/await (FUTURES.md)

**Goal**: add RunMat language constructs.

- Parser/AST/HIR additions for:
  - `async @() do ... end`
  - `await(expr)` intrinsic
- Bytecode opcodes:
  - `ASYNC_CREATE`, `AWAIT`
- Value representation:
  - `Value::Task(TaskHandle)` and/or `Value::Awaitable(...)`
- Standard library async builtins:
  - `sleep_ms`, async FS primitives, etc. (these return tasks/awaitables explicitly)

**Exit criteria**
- Minimal async programs run deterministically.
- Concurrency is created by tasks (`async`/`spawn`) around existing language-synchronous builtins
  (e.g., `webread`), not by maintaining parallel `*_async` builtin name variants.

---

### Phase 6: cleanup + removal of transitional code

**Goal**: remove interim hacks introduced during the WebGPU investigation.

- Delete any sentinel string control-flow.
- Delete JS “internal drain loops” if they exist (executor/wakers should handle it).
- Remove compatibility shim code that only existed for the transitional design.

**Exit criteria**
- No “pending request” transitional protocol remains; all waiting is waker-driven.

---

## Tracking checklist (update as we go)

- [x] `runmat-async` crate created
- [x] `runmat-control-flow` merged/deleted
- [x] typed control-flow replaces sentinel strings
- [x] `ExecuteFuture` implemented
- [x] wasm executor adapter implemented (Promise-backed `execute`)
- [x] internal GPU awaitable implemented (waker-driven WebGPU map/readback)
- [x] external input awaitable implemented (Promise-driven stdin via `setInputHandler`)
- [ ] language async/await syntax shipped
- [ ] transitional code removed

---

## Open questions (keep updated)

- **Lazy vs eager tasks**: default to lazy? (`async` creates suspended task; `spawn` starts)
- **Top-level await**: allowed in REPL? in scripts?
- **GC rooting API**: best representation for cross-await handles
- **Cancellation semantics**: how to expose and how to integrate with host cancel


