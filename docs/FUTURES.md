# Proposal: Async/await for RunMat via Rust Futures end-to-end

## 1) Motivation and goals

### Why async now

RunMat’s current execution model (parse → HIR → bytecode → interpret/JIT) assumes “evaluation completes or errors.” That makes:

* I/O-heavy workloads (files, network, UI) awkward (blocking stalls the runtime)
* structured concurrency (fan-out/fan-in) non-idiomatic
* integration with modern Rust ecosystem (Tokio, async FS, async HTTP, GPU scheduling) harder than it needs to be

Moving “execute onward” to a Rust future state machine unlocks:

* non-blocking I/O and timers
* cooperative scheduling with predictable performance
* better host integration (CLI, Jupyter, WASM, GUI)
* tracing spans that naturally flow across suspension points

### Goals

1. **Matlab compatibility preserved**: any valid MATLAB should run unchanged.
2. **RunMat adds new capabilities**: RunMat code need not run in MATLAB.
3. **Uniform async substrate**: *every* evaluation is a Rust future, even “sync” code (it just completes immediately).
4. **Deterministic semantics**: async suspension points are explicit (`await`) and predictable.
5. **Good performance**: interpreter fast path stays fast; async adds minimal overhead when not used.

### Non-goals (v1)

* Implicit concurrency without `await` / task creation.

> Note: under the futures-based execution model, the engine may return `Pending` internally while
> waiting for host events (e.g. HTTP, timers, WebGPU callbacks) even when user code does not
> explicitly call `await`. This is *not* “implicit concurrency”: language semantics remain
> sequential; the runtime merely avoids blocking the host event loop.

---

## 2) User-facing language design

### 2.1 Minimal surface primitives

Introduce 2 intrinsics (syntactically normal calls, semantically special):

* `async <block>` → returns `Task<T>`
* `await(expr)` → suspends until `expr` (a Task/Future-like value) is ready

**Block form**: we need a multi-statement closure. Proposed RunMat-only block lambda:

```matlab
% Parallelism comes from tasks around synchronous builtins.
t1 = async @() do
  webread(url1)
end

t2 = async @() do
  webread(url2)
end

a = await(t1);
b = await(t2);
a + b
```

Key principle:

- Builtins like `webread` remain **language-synchronous** (they return values, not tasks).
- The runtime may yield internally while they complete, but program order remains sequential.
- Concurrency is explicit: create multiple tasks and `await` them.

This requires:

* `do ... end` block closure syntax (RunMat-only)
* `async` recognized in “call position” or “prefix position” when followed by a closure/block
* `await(...)` recognized as an intrinsic

### 2.2 Optional sugar: `async function`

Add contextual keyword only when preceding `function`:

```matlab
async function y = download(url)
  y = await(fetch(url));
end
```

This is not MATLAB-compatible; it’s fine. It does not break MATLAB parsing because MATLAB doesn’t accept `async function`.

Desugaring:

* `async function f(args) body end`
* becomes: `function f(args) f = async @() do body end; end`
  (or returns a Task directly if you want `f(args)` to be a Task—see below).

### 2.3 Return types and calling conventions

Two viable models:

**Model A: async functions return Task**

* `async function y = f(...)` means calling `f(...)` yields a Task resolving to `y`.
* `await(f(...))` returns the value.

**Model B: async functions are “implicitly awaited” in sync contexts**

* Avoid. It’s magic, and it complicates determinism and tooling.

**Recommendation: Model A**. It’s explicit and maps cleanly to Rust.

#### Builtins are language-synchronous

To preserve MATLAB compatibility and avoid “two builtins per builtin”, builtins should keep stable
signatures/return types:

- `webread(url)` returns the downloaded value (sequential semantics).
- When executed under the futures-based engine, the evaluator may internally suspend while the
  request completes, but the *language* remains sequential.
- Users opt into concurrency via tasks (`async`/`spawn`) and `await`, not by choosing different
  builtin names.

### 2.4 Core library async utilities (v1 set)

Provide builtins returning Tasks:

* `sleep_ms(ms) -> Task<void>`
* `read_text_async(path) -> Task<string>`
* `write_text_async(path, s) -> Task<void>`
* `spawn(block) -> Task<T>` (alias of `async` or wrapper)
* `join(tasks...) -> Task<tuple>` (fan-in)
* `race(tasks...) -> Task<first>` (optional)
* `with_timeout_ms(ms, task) -> Task<T>`
* `cancellation_token() -> [cancel_fn, token]`
* `with_cancel(token, task) -> Task<T>`

These allow real use without forcing users into raw polling loops.

---

## 3) Semantics: evaluation as an async state machine

### 3.1 Big idea: all evaluation is a Future

Define: the “execute” API returns a Rust future:

```rust
pub fn execute(session: &mut Session, code: Source) -> impl Future<Output=Result<Value, Error>>
```

Even if code contains no `await`, the future completes immediately (poll once → Ready).

### 3.2 Suspension points

Only `await(expr)` can suspend (v1). That gives:

* predictable “atomic regions” between awaits
* simple debugging semantics (“where can I yield?”)

### 3.3 Concurrency model

Cooperative: tasks yield only at await points (and internal runtime awaits like timers). No preemption.

### 3.4 Determinism / ordering

* Within a single task: semantics are sequential, like today.
* Across tasks: scheduling order is executor-defined but should be stable enough for tests (we can provide a deterministic executor for testing).

---

## 4) Runtime architecture

### 4.1 Core traits and types

#### RunMat Task value

At language level, Tasks are Values:

* `Value::Task(TaskHandle)` where `TaskHandle` points to an owned task in the scheduler.
* Tasks resolve to `Value`.

Rust-side:

```rust
pub struct TaskHandle { id: TaskId, /* ... */ }

pub enum PollResult {
  Ready(Value),
  Pending,
}
```

But if you’re going “Rust futures all the way,” the natural internal representation is:

* tasks are `Pin<Box<dyn Future<Output=Result<Value, Error>> + Send>>` (or !Send for single-thread)
* store them in a scheduler arena keyed by TaskId
* the language `TaskHandle` is a reference to that TaskId, with refcount/liveness tracked.

#### Executor abstraction

Introduce a runtime executor trait:

```rust
pub trait RunMatExecutor {
  fn spawn(&mut self, fut: RunMatFuture) -> TaskId;
  fn wake(&mut self, id: TaskId);
  fn register_timer(&mut self, at: Instant, id: TaskId);
  fn poll_task(&mut self, id: TaskId, cx: &mut Context<'_>) -> Poll<Result<Value, Error>>;
}
```

Implementations:

* `LocalExecutor` (single-threaded, deterministic option)
* `TokioExecutor` (native integration; uses tokio timers, fs, net)
* `WasmExecutor` (uses JS promises/timers)

Make this pluggable via `RunMatSession` config.

### 4.2 “Execute onward” future chain

Today: execute drives interpretation until completion.

New: execute returns a future that *is* the interpreter state machine.

Conceptually:

* `ExecuteFuture` owns:

  * VM state (stack, frames, instruction pointer)
  * references to GC arena
  * current function/code object refs
  * scheduler handle/executor
  * “pending await” state if currently suspended

Polling `ExecuteFuture`:

1. runs bytecode until:

   * finishes (return) → Ready(result)
   * throws error → Ready(Err)
   * hits `AWAIT` and the awaited task is Pending → returns Pending after registering wake
2. on resume (wake), continue from saved PC.

This is a classic “async interpreter loop”.

---

## 5) VM / bytecode changes

### 5.1 New HIR nodes

Add:

* `AsyncBlock { body, captures }`
* `Await { expr }`
* `Spawn { closure }` (if you keep `async` and `spawn` distinct)
* Optional: `Join`, `Race`, `Timeout` as library calls (not IR nodes)

### 5.2 Bytecode opcodes

Minimum set:

* `ASYNC_CREATE <closure>` → pushes `TaskHandle` (task scheduled but not necessarily run yet)
* `AWAIT` → pops `TaskHandle` or awaitable Value, and:

  * if ready: pushes resolved Value
  * if pending: saves current frame state and returns Pending from the interpreter future

Also helpful:

* `YIELD` (optional explicit cooperative yield)
* `CANCEL_TASK` (optional; could be builtin)

### 5.3 Awaitable protocol

Not everything is a TaskHandle. Users may get “awaitables” from builtins. Define a protocol:

* `await(x)` checks:

  1. if `x` is `TaskHandle`, poll that task
  2. else if `x` is a runtime “native future wrapper” (Value::NativeFuture), poll it
  3. else error: “not awaitable”

This allows returning “native future” from builtins without spawning a separate task object, if desired.

---

## 6) JIT (Turbine) implications

### 6.1 v1 strategy: async remains interpreted

For first iteration:

* compile normal functions as today
* for async functions/blocks: interpret them (Ignition) until stable semantics exist

This de-risks the first release.

### 6.2 v2: JIT state-machine compilation

Later, compile async functions into native code that includes a state enum:

* each `await` becomes a state transition
* locals spilled into a heap frame (or stack frame + spill area)
* return Pending maps to a trampoline

This is essentially “lower to a coroutine state machine,” same as Rust does.

### 6.3 Integrating with Cranelift

Cranelift doesn’t directly model coroutines; you implement:

* a function `poll(frame_ptr, cx_ptr) -> Poll<Value>`
* a `switch(state)` at top
* each suspension sets `frame.state = next_state` and returns Pending

---

## 7) GC and async safety

This is one of the biggest correctness risks: values can live across yields.

### 7.1 Rooting across suspension

At any `await`, the task’s frame must be treated as GC roots:

* stack slots
* locals
* captured variables
* any pending temporaries

Implement an explicit “async frame object” that holds spilled references and is registered as a GC root while task is alive.

### 7.2 Moving GC + pinned references

If your GC is moving/compacting, you must not store raw pointers to GC objects inside Rust futures across yield points unless they’re handles that survive moves (you mentioned handle-based access—good).

Rule:

* async frames store `GcHandle<Value>` or equivalent stable handles
* no raw `*const` to GC-managed memory across awaits

### 7.3 Finalization and cancellation

When a task is dropped/cancelled:

* remove it from scheduler
* drop its async frame roots
* allow GC to reclaim objects

---

## 8) Cancellation, timeouts, and structured concurrency

### 8.1 Cancellation token

Make cancellation explicit and composable:

* `cancellation_token()` returns `[cancel_fn, token]`
* `with_cancel(token, task)` returns a new Task that resolves:

  * task result if completes first
  * error/cancelled if token triggered

Under the hood:

* token is a shared atomic + waker list
* triggering wakes tasks waiting on it

### 8.2 Structured concurrency defaults

Encourage patterns that avoid “fire and forget” leaks:

* `async` creates a task
* tasks are values; if user discards them, they still run (maybe) — that’s dangerous

Options:

* require `spawn(...)` for detached tasks; `async` alone creates a “lazy task” that runs only when awaited
* or keep eager semantics but warn in lint/tooling

**Recommendation**: make tasks **lazy-by-default**:

* `async @() do ... end` creates a suspended task that starts on first `await` or `spawn`
* `spawn(task)` schedules it immediately and returns a handle

This mirrors some structured concurrency designs and reduces accidental background tasks.

### 8.3 Timeout

`with_timeout_ms(ms, task)`:

* schedules a timer
* race task vs timer
* if timer wins: cancel task and error

---

## 9) Tracing integration (since you’re planning it anyway)

Async + tracing should be designed together so spans “flow” across awaits.

### 9.1 Task-local span

Attach a `tracing::Span` to each task. When polling:

* enter span for the duration of poll
* exit before returning Pending

This yields correct structured traces across suspensions.

### 9.2 Instrumentation points

* task creation: `trace.event("task.spawned", ...)`
* await start / await ready: optionally at debug level
* cancellation: event + reason
* long polls: event if a poll takes > threshold

Keep default verbosity low.

---

## 10) Host integration

### 10.1 CLI / REPL

* top-level execution returns a future
* CLI can `block_on` with chosen executor
* REPL can run a local executor; optionally allow top-level `await(...)` by making REPL evaluation an async context

### 10.2 Jupyter

Notebook cells can naturally be async:

* cell evaluation future resolves to display value
* integrate with the notebook event loop (often async-friendly)

### 10.3 WASM

* implement executor backed by JS promises and `setTimeout`
* restrict blocking operations; require async builtins

---

## 11) Backwards compatibility story

* MATLAB code parses the same; no new reserved keywords in general expression positions.
* `await` and `async` are introduced as **intrinsics**, but only recognized in specific syntactic forms:

  * `await(<expr>)` call form
  * `async <closure/block>` or `async(<closure>)` form
* If user had variables named `await` in old MATLAB code, only `await(...)` becomes special; a bare identifier `await` remains normal.

---

## 12) Implementation plan (phased)

### Phase 0: plumbing

* Introduce `RunMatExecutor` abstraction
* Implement `LocalExecutor` + deterministic testing mode
* Convert `execute` to return a future, but initially it always completes immediately (no async ops)

### Phase 1: async runtime + opcodes

* Add `Value::Task`
* Add `ASYNC_CREATE`, `AWAIT` opcodes
* Add `async @() do ... end` and `await(expr)` parsing + HIR nodes
* Implement interpreter as a future state machine (poll loop that can return Pending)

### Phase 2: async builtins

* Add `sleep_ms`, `read_text_async`, `write_text_async`
* Provide `spawn`, `join`, `with_timeout_ms`, cancellation token
* Add basic tracing hooks (task spans), even if tracing surface language comes later

### Phase 3: ergonomics + tooling

* `async function` sugar
* better errors: “await used outside async context” (if you enforce that)
* debugging: show current await site, task backtrace-ish info
* snapshot support: ensure stdlib async functions compile into snapshot cleanly

### Phase 4: performance

* reduce interpreter overhead for await checks
* optional JIT compilation for async state machines
* optimize task storage, wake queues, timer wheel

---

## 13) Key design decisions to lock down early

1. **Lazy vs eager tasks**

   * Lazy (`async` creates suspended task; `spawn` starts it) is safer.
2. **Allowed await contexts**

   * Only inside async blocks/functions (strict) vs allow top-level await (REPL convenience).
3. **Blocking I/O behavior**

   * Provide async-only APIs for I/O (best) or support spawn_blocking under the hood.
4. **Task lifetime**

   * What happens if task is GC’d without being awaited? cancel it? detach it? warn?
5. **Error propagation**

   * `await(task)` returns value or throws the task’s error in the awaiting context.

---

## 14) Testing strategy

* Deterministic executor for unit tests:

  * manual time advancement for timers (`advance_time(ms)`)
  * predictable wake ordering
* Property tests for GC safety across awaits:

  * allocate many objects, yield often, force GC, ensure handles remain valid
* Concurrency tests:

  * join/race correctness
  * cancellation/timeout edge cases
* Host tests:

  * CLI block_on
  * Jupyter cell completion semantics

---

## 15) Example: polling file equals “x” (idiomatic with the model)

User code:

```matlab
async function v = wait_file_equals(path, expected, interval_ms)
  while true
    v = await(read_text_async(path));
    if v == expected
      return v;
    end
    await(sleep_ms(interval_ms));
  end
end
```

This compiles to:

* async function → task state machine
* loop contains two await points
* GC roots include `path`, `expected`, `interval_ms`, `v` across suspensions

---

If you want, I can turn this into a tighter “RFC format” with explicit sections for:

* grammar changes (EBNF-ish)
* HIR additions
* opcode definitions and interpreter pseudo-code
* executor + waker mechanics
* GC rooting API for async frames

…and I can propose a concrete Rust API surface for `RunMatSession::execute_async`, `spawn_task`, and the internal `ExecuteFuture` struct layout.
