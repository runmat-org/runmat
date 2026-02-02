## Architecture: Async/Await for RunMat (Futures end-to-end)

### Purpose

This document is the **north-star design** for adding async/await to RunMat by making evaluation a
Rust `Future` end-to-end. It is written to minimize hacks, avoid “sentinel” control-flow, and keep
RunMat as a thin, clean language/runtime layer that leverages Rust’s established async semantics
(`Future`, `Waker`, executors), while preserving MATLAB compatibility for existing synchronous code.

This doc is paired with `docs/ARCH_ASYNC_PLAN.md`, which tracks an incremental rollout plan.

### Goals and non-goals

- **Goals**
  - **Evaluation is a Future**: all RunMat evaluation can be driven via `poll`, returning `Pending`
    at suspension points and `Ready` at completion.
  - **Typed suspension**: no “pending as string” or implicit sentinel error messages.
  - **Host-neutral**: the core async model works in native CLI, GUI, Jupyter, and WASM.
  - **Deterministic semantics**: suspension is explicit (primarily via `await`), with predictable
    atomic regions between awaits.
  - **Good performance**: “sync-only” code remains fast; async overhead is near-zero when unused.
  - **Clean layering**: crates do not pull host dependencies (Tokio, `web_sys`) unintentionally.

- **Non-goals (v1)**
  - Preemptive scheduling (cooperative only).
  - Implicit async I/O (no hidden yields without an explicit await point, except internal awaits that
    are semantically transparent and carefully audited).
  - Full async JIT compilation on day 1 (interpret-first is acceptable).

---

## Key design: evaluation as a pollable state machine

### The core model

RunMat evaluation becomes a `Future` whose state is the interpreter (and later JIT coroutine) state:

- `ExecuteFuture` owns the VM state (instruction pointer, frames, operand stack).
- `poll()` runs until:
  - **Completion**: produces a final `ExecutionResult`.
  - **Error**: produces a typed `RunMatError`.
  - **Suspension**: returns `Poll::Pending` after registering a wake.

### Suspension points

The rule for determinism:

- **User-visible suspension points** are explicit via `await(...)` in RunMat code (and async blocks).
- **Internal suspension points** (e.g., waiting for WebGPU `map_async`) are modeled as internal
  awaitables and must be **semantically transparent** to users (no reordering surprises).

This gives predictable execution regions between awaits and avoids “hidden” re-entrancy.

#### Builtins remain language-synchronous

A key compatibility constraint is that builtins keep stable signatures/return types:

- Builtins like `isempty` always complete immediately.
- Builtins like `webread`/GPU readback remain *language-synchronous* (“evaluate expression, produce
  value, then continue”), but the evaluator may return `Poll::Pending` while waiting for host events.

This avoids “two builtins per builtin” and avoids config-driven return-type changes. Concurrency is
introduced by running multiple tasks, not by builtins returning tasks conditionally.

---

## Separation of concerns (crates and responsibilities)

This section defines what lives where, and what should *not* leak across layers.

### `runmat-parser`

- **Owns**: source text → AST.
- **Does not own**: runtime state, values, IO, executors.

### `runmat-hir` (and lowering)

- **Owns**: AST → HIR, HIR transforms, validation.
- **Does not own**: execution, async scheduling, host APIs.

### `runmat-ignition` (VM/interpreter)

- **Owns**: bytecode format and interpreter engine.
- **Exposes**: a pollable interpreter core (“run until yield”).
- **Must not depend on**: Tokio, JS/wasm bindings, GPU providers.

Ignition should understand only:
- “execute instruction stream”
- “hit an await opcode / awaitable”
- “produce Completed / Pending(awaitable-id) / Error”

### `runmat-runtime` (builtins and MATLAB semantics)

- **Owns**: builtin functions and semantic helpers.
- Builtins must not block the host. Under the futures-based engine, they may cause the evaluation
  to return `Pending` while waiting for host events, while keeping language semantics sequential.
- `await` is an intrinsic or opcode (recommended as opcode for performance and clear semantics).

### `runmat-core` (session orchestration)

- **Owns**:
  - `RunMatSession` public API and workspace model
  - execution planning, cancellation wiring, profiling/tracing
  - integration glue between ignition + runtime + GC + async substrate
- **Exposes**:
  - `execute_async(...) -> ExecuteFuture`
  - convenience wrappers for native hosts (e.g., `execute_blocking` using a host executor)

### `runmat-accelerate` (GPU providers and scheduling)

- **Owns**: GPU kernels, provider abstractions, and GPU-backed awaitables.
- Must not know about the VM; it can expose awaitables that wake when GPU work completes.

### New: `runmat-async` (unified async substrate)

We introduce a new crate **`runmat-async`** as the single home for:

- typed yielding/suspension types (what used to be “control-flow”)
- executor trait(s)
- task identifiers/handles
- “awaitable value” representations used by the runtime/core
- deterministic local executor for tests (either in this crate or a submodule)

We should fold (or delete) the prior `runmat-control-flow` crate into `runmat-async` to avoid dual
control-flow APIs.

### Host bindings crates

- `runmat-wasm`: WASM bindings and *host adapter* to `runmat-async` (JS timers, promise wakeups).
- Desktop/worker TypeScript: thin transport layer only; should not implement policy beyond “drive
  the runtime/executor”.

---

## The async substrate (`runmat-async`)

### Why a dedicated crate

We want:
- a stable dependency boundary for the VM/runtime/core to share async semantics
- to avoid pulling host dependencies (Tokio, `web_sys`) into core crates

### Proposed API surface (initial)

#### Task identity

- `TaskId` (copyable id)
- `TaskHandle` (language-level value referencing a task)

#### Executor trait

This is intentionally minimal and host-neutral:

- `spawn(fut) -> TaskId`
- `wake(task_id)` (optional if wakers directly queue tasks)
- `register_timer(deadline, task_id)` (or timer awaitable)
- `poll_task(task_id, cx) -> Poll<Result<Value, RunMatError>>`

We can refine the trait, but the key is: **RunMat does not hardcode Tokio**.

#### Awaitables

At the language/runtime boundary, `await(x)` needs a protocol:

- if `x` is a `TaskHandle`, poll that task
- else if `x` is a native awaitable wrapper (`Value::Awaitable(...)`), poll that
- else error: “not awaitable”

Implementation strategy:
- store `Pin<Box<dyn Future<Output=Result<Value, RunMatError>> + 'session>>` for tasks
- store non-task awaitables similarly or via a small vtable if we need to avoid trait objects

#### Cancellation

Cancellation is best modeled as:
- a cancellation token (shared state + waker list)
- cancellation checks at poll boundaries and await points

---

## VM integration (Ignition) and bytecode changes

### Bytecode opcodes (minimum)

- `ASYNC_CREATE <closure>`: create a task (lazy or eager; we recommend lazy-by-default).
- `AWAIT`: pop awaitable; if ready push value; if pending yield from interpreter poll.

Optional but useful:
- `YIELD` (explicit cooperative yield; mainly for runtime fairness/testing)

### Ignition poll contract

Ignition should expose something like:

- `fn poll_execute(&mut self, cx: &mut Context<'_>) -> Poll<InterpreterDone>`

Where `InterpreterDone` is either:
- Completed(value(s))
- Error(RunMatError)

When it hits `AWAIT` on a pending awaitable, it returns `Poll::Pending` and ensures the awaitable
has registered `cx.waker()`.

### State ownership

The VM state that must live across yields includes:
- operand stack
- call frames
- instruction pointer(s)
- “spilled” locals across await boundaries

This state is owned by `ExecuteFuture` or by a VM struct owned by it.

---

## GC rooting across suspension

This is the highest-risk correctness topic in async runtimes.

### Rule

**No raw pointers to GC-managed objects may live across an await.**

Instead:
- store stable handles (indices/ids) into the GC arena
- maintain an explicit “async frame root set” for all live handles

### What must be rooted

Across any suspension:
- stack values
- locals
- captured closure env values
- temporary values that might be used after resuming

### Proposed mechanism

- `ExecuteFuture` owns an `AsyncFrame` object:
  - contains `Vec<GcHandle>` (or equivalent) for all rooted values
  - is registered as a GC root for the lifetime of the future/task

This should integrate with existing GC root registration infrastructure (not duplicate it).

---

## Error model (typed, no sentinel)

### Requirements

- Suspension must not be representable as an error string.
- Error adaptation layers must not be able to “wrap away” suspension.

### Proposed

At the Rust level:
- `RunMatError` (real errors)
- `Poll::Pending` (suspension)

At the language level:
- `await` either returns a value or raises the awaited task’s error.

For external requests (stdin/UI), `await(input(...))` produces a pending request awaitable that the
host fulfills; the task is woken and resumes.

---

## Host integration

### WASM

WASM is the forcing function: you cannot block the event loop. The correct model is:

- `execute_async` returns a JS `Promise` (from TS bindings) that awaits completion of the
  underlying `ExecuteFuture`
- internal awaitables (GPU map/readback) wake via JS callbacks → waker → resume poll
- external awaitables (stdin prompt) are surfaced to JS/UI; resolution triggers wake

Key property: **no synchronous busy-wait** in WASM.

### Native

Native hosts can choose:
- `execute_blocking` (implemented via a local executor or Tokio `block_on`)
- `execute_async` integrated into an existing async runtime

### Desktop worker / TS transport

The worker should be a thin transport:
- start execution
- forward stdout events
- forward external pending requests to UI
- fulfill external requests when UI responds

It should not implement “special GPU loops”; internal awaitables are executor-driven.

---

## GPU/WebGPU integration (WGPU)

### Desired shape

GPU readback is an internal awaitable:

- scheduling `map_async` returns an awaitable/future that:
  - registers a waker
  - wakes on callback completion
  - yields mapped bytes to the provider code to decode into host tensors

This avoids:
- blocking in wasm
- host polling loops
- leaking GPU details into unrelated runtime code

### Provider contract

Providers should expose async-friendly operations by returning either:
- immediate values, or
- awaitables (internal) representing pending completion

---

## Migration notes (from current code)

This repository currently contains transitional mechanisms to avoid event-loop starvation:

- “pending request” plumbing and host auto-drain loops
- error-sentinel preservation in a few wrapper layers

These are **temporary** and should be removed once:
- the interpreter is a future
- suspension is typed (`Pending` via wakers)
- host uses a real executor adapter

In particular, any solution that encodes suspension as a string/sentinel is considered transitional
and must be removed as part of the futures architecture rollout.

The plan in `docs/ARCH_ASYNC_PLAN.md` describes how to retire the transitional mechanisms safely.


