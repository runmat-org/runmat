# Module Map

`runmat-vm` owns RunMat bytecode compilation, interpreter execution, runtime semantics, and acceleration integration.

## Top-level layout

```text
src/
  accel/
  bytecode/
  call/
  compiler/
  indexing/
  interpreter/
  object/
  ops/
  runtime/
```

## Ownership

### `bytecode/`
- Bytecode contracts and compile entrypoints.
- Owns `Instr`, `EndExpr`, `Bytecode`, `UserFunction`, `ExecutionContext`, and `compile(...)`.

### `compiler/`
- HIR-to-bytecode lowering.
- `core.rs` holds compiler state and shared emit/error helpers.
- `expressions.rs`, `statements.rs`, `lvalues.rs`, `functions.rs`, `classes.rs`, and `imports.rs` own the main lowering concerns.
- `end_expr.rs` owns lowering-time `end` expression construction.

### `interpreter/`
- Shared interpreter shell and dispatch.
- `runner.rs` owns the interpreter entrypoints and main loop.
- `engine.rs` owns loop setup and execution prelude helpers.
- `dispatch/` owns grouped opcode routing.
- `debug.rs` owns interpreter debug tracing helpers.

### `runtime/`
- Shared runtime state outside the main interpreter loop.
- `call_stack.rs` owns call stack limits and error namespace.
- `workspace.rs` owns workspace snapshot/import/export plumbing.
- `globals.rs` owns global/persistent storage.
- `gc.rs` owns interpreter GC root registration.

### `ops/`
- Concrete opcode-family semantics.
- Arithmetic, comparison, arrays, stack, cells, and control-flow execution helpers live here.

### `call/`
- Call semantics shared by interpreter and lowering.
- Builtin dispatch, user-function preparation, closures, `feval`, and output shaping.

### `indexing/`
- MATLAB-compatible indexing read/write semantics.
- Selector normalization, `end` resolution, linear indexing, slice gather, and slice scatter.

### `object/`
- Object/class member semantics.
- Member reads/writes, static dispatch, method loading, and runtime class registration.

### `accel/`
- Acceleration-specific compile/runtime support.
- `graph.rs` and `stack_layout.rs` hold fusion graph metadata.
- `idioms/` owns deterministic math-idiom detection/lowering and runtime execution hooks.
- `fusion.rs` owns fusion execution helpers.
- `residency.rs` owns GPU residency policy helpers.
- `auto_promote.rs` owns accel-aware argument/value promotion.
