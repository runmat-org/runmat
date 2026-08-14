# Module Map

`runmat-vm` owns RunMat bytecode compilation, interpreter execution mechanics, and acceleration integration. Executor-neutral language semantics live in `runmat-runtime`; the VM adapts bytecode stacks, slots, and control flow to those shared operations.

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
- Retains the complete analyzed region inventory and maps each stable MIR entry/exit point to an exact empty-stack bytecode PC; missing boundaries fail executable construction.

### `compiler/`
- HIR-to-bytecode lowering.
- `core.rs` holds compiler state and shared emit/error helpers.
- `exceptions.rs` derives explicit try-scope regions and normal-exit edges from the MIR control-flow graph.
- `expressions.rs`, `statements.rs`, `lvalues.rs`, `functions.rs`, `classes.rs`, and `imports.rs` own the main lowering concerns.
- `end_expr.rs` owns lowering-time `end` expression construction.

### `interpreter/`
- Shared interpreter shell and dispatch.
- `runner.rs` owns the interpreter entrypoints and main loop.
- `engine.rs` owns loop setup and execution prelude helpers.
- `dispatch/` owns grouped opcode routing.
- `debug.rs` owns interpreter debug tracing helpers.

### `runtime/`
- VM adapters for runtime state outside the main interpreter loop.
- `call_stack.rs` adapts Runtime-owned call-stack limits and attaches bytecode source frames to errors.
- `workspace.rs` owns workspace snapshot/import/export plumbing.
- `globals.rs` synchronizes bytecode slots with Runtime-owned named global/persistent storage.
- `gc.rs` owns interpreter GC root registration.

### `ops/`
- Concrete opcode-family semantics.
- Arithmetic, comparison, arrays, stack, cells, and control-flow execution helpers live here.
- Try/catch handlers carry an explicit compiler-assigned scope. Normal control-flow edges leave only the scopes whose protected MIR regions they exit, while exceptional edges consume the innermost handler and preserve enclosing handlers.

### `call/`
- Bytecode call decoding and interpreter-frame preparation.
- Stack argument specifications are materialized here, while comma-list expansion, callable descriptors, object brace dispatch, function-entry preparation, argument validation/defaults, and output contracts are delegated to Runtime so VM, native, and browser executors share one language implementation.

### `indexing/`
- Bytecode indexing adapters and VM-specific stack/slot coordination.
- Selector normalization, `end` evaluation, reads, writes, and object/cell dispatch consume the corresponding Runtime-owned semantics.

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
