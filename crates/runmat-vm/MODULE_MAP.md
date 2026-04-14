# runmat-vm Module Map

`runmat-vm` is a parity-first rename and refactor target for `runmat-ignition`.

The rule for this migration is simple: preserve semantics first, improve structure second.

## Migration goals

- Keep the bytecode and interpreter semantics exactly aligned with `runmat-ignition` while porting.
- Move away from god-files, especially `src/vm.rs`.
- Give the VM a module tree that reflects semantic ownership.
- Port subsystem-by-subsystem with tests staying green throughout.

## Current source of truth

Until the port is complete, `runmat-ignition` remains the behavior source of truth.

## Migration status

Substantial ownership has already moved into `runmat-vm` for the following areas:

- `bytecode/`: `Instr`, `Bytecode`, `UserFunction`, `ExecutionContext`, `CompileError`, and `compile(...)`.
- `runtime/`: call-stack limit/namespace state and workspace import/export state.
- `interpreter/`: public interpreter state types, current-PC/span helpers, timing helpers, stack helpers, and runtime error helpers.
- `ops/`: stack op family, control-flow op family, and cell creation/read/write helpers.
- `call/`: builtin shell helpers, user-function prep/output shaping, `feval` helpers, closure/method/static call helpers, and expansion helpers.
- `indexing/`: selector normalization, `end` evaluation, linear read, most slice reads, cell indexing helpers, linear writes, and most slice writes.
- `object/`: member read/write resolution, static property/method resolution, and runtime class registration.

This means `runmat-ignition/src/vm.rs` is now primarily retaining execution shells and a shrinking set of fallback paths rather than owning the full subsystem logic.

Key current files:

- `runmat-ignition/src/instr.rs`: bytecode IR contract.
- `runmat-ignition/src/functions.rs`: `Bytecode`, `UserFunction`, `ExecutionContext`.
- `runmat-ignition/src/compiler.rs`: HIR to bytecode lowering.
- `runmat-ignition/src/bytecode.rs`: compile orchestration and accel graph attachment.
- `runmat-ignition/src/vm.rs`: interpreter, runtime state, opcode execution, accel integration.
- `runmat-ignition/src/accel_graph.rs`: fusion graph builder.
- `runmat-ignition/src/gc_roots.rs`: interpreter GC roots.

## Target structure

The target structure keeps files small by splitting by subsystem and then by operation family.

```text
runmat-vm/
  src/
    lib.rs
    accel/
      mod.rs
      idioms/
        mod.rs
        stochastic_evolution.rs
      fusion.rs
      graph.rs
      residency.rs
      stack_layout.rs
    bytecode/
      mod.rs
      instr.rs
      program.rs
    call/
      mod.rs
      builtins.rs
      closures.rs
      feval.rs
      shared.rs
      user.rs
    compiler/
      mod.rs
      classes.rs
      end_expr.rs
      expressions.rs
      functions.rs
      imports.rs
      lvalues.rs
      statements.rs
    indexing/
      mod.rs
      end_expr.rs
      read_linear.rs
      read_slice.rs
      selectors.rs
      write_linear.rs
      write_slice.rs
    interpreter/
      mod.rs
      api.rs
      dispatch.rs
      engine.rs
      errors.rs
      stack.rs
      state.rs
      timing.rs
    object/
      mod.rs
      class_def.rs
      member_read.rs
      member_write.rs
      method_call.rs
      resolve.rs
      static_dispatch.rs
    ops/
      mod.rs
      arithmetic.rs
      arrays.rs
      cells.rs
      comparison.rs
      control_flow.rs
      stack.rs
    runtime/
      mod.rs
      call_stack.rs
      gc.rs
      globals.rs
      workspace.rs
```

## Ownership map

### `bytecode/`

- Owns instruction definitions and bytecode program data.
- First port target because it is a stable contract used by compiler, VM, and dependents.

Port source:

- `runmat-ignition/src/instr.rs`
- parts of `runmat-ignition/src/functions.rs`

### `compiler/`

- Owns lowering from HIR to bytecode.
- Split by lowering concern rather than by AST type count.

Current internal ownership:

- `core.rs`: compiler state, setup, emit/patch/error helpers, and thin span-guarded entrypoints.
- `end_expr.rs`: lowering-time `end` expression construction and numeric `end` helper parsing.
- `lvalues.rs`: assignment-target lowering including indexed/member lvalues and store-back chains.
- `functions.rs`: function statement registration, closure capture analysis, and anonymous-function lowering.
- `classes.rs`: class-definition lowering and class/member attribute parsing.
- `statements.rs`: general statement lowering including control flow, imports, globals/persistents, and multi-assign.
- `expressions.rs`: general expression lowering including calls, indexing, member access, literals, and operators.
- `imports.rs`: compile-time import/static resolution helpers for unqualified names and `Class.*` lookup.

Port source:

- `runmat-ignition/src/compiler.rs`
- `runmat-ignition/src/bytecode.rs`

### `interpreter/`

- Owns public interpreter API, execution loop, dispatch, state, timing, and error attachment.
- Should not directly contain family-specific opcode behavior beyond dispatch wiring.

Port source:

- `runmat-ignition/src/vm.rs`

### `runtime/`

- Owns cross-cutting runtime state used by the interpreter.
- Pulls workspace/global/persistent/call-stack/GC machinery out of the dispatch loop.

Port source:

- thread-local and state sections from `runmat-ignition/src/vm.rs`
- `runmat-ignition/src/gc_roots.rs`

### `call/`

- Owns builtin, user function, closure, and `feval` invocation semantics.
- Exists because call semantics are currently duplicated in several paths.
- `shared.rs` is the home for argument collection, expansion handling, frame setup, and output shaping.

Current migrated ownership:

- builtin shell helpers and import-resolution helpers in `builtins.rs`
- user-function prep/output shaping in `shared.rs`
- closure/method/static dispatch helpers in `closures.rs`
- `feval` dispatch helpers in `feval.rs`

Port source:

- call-related sections from `runmat-ignition/src/vm.rs`

### `indexing/`

- Owns selector normalization, `end` evaluation, slice read/write, and linear read/write.
- This is one of the highest-risk areas and gets its own top-level subtree.
- `selectors.rs` is the home for selector materialization, selector classification, and shared index planning.

Current migrated ownership:

- selector normalization and slice-plan building in `selectors.rs`
- runtime `end` expression evaluation in `end_expr.rs`
- linear reads in `read_linear.rs`
- most read-side slice handling in `read_slice.rs`
- linear writes in `write_linear.rs`
- most slice writes in `write_slice.rs`

Port source:

- indexing and slice sections from `runmat-ignition/src/vm.rs`
- `EndExpr` contract from `runmat-ignition/src/instr.rs`

### `object/`

- Owns object, struct, member, method, static dispatch, and class registration behavior.
- Keeps class semantics away from generic stack dispatch.
- `resolve.rs` is the shared semantic center for field, property, method, and static target resolution.

Current migrated ownership:

- member read/write resolution in `resolve.rs`
- static property/member resolution in `resolve.rs`
- runtime class registration in `class_def.rs`

Port source:

- member/method/class sections from `runmat-ignition/src/vm.rs`
- class lowering portions of `runmat-ignition/src/compiler.rs`

### `ops/`

- Owns straightforward opcode families that do not justify their own top-level subsystem.
- If any file grows too large, it should split again by opcode group.
- These files should stay thin and should delegate shared rules to stack, error, runtime, indexing, and object resolver helpers.

Port source:

- arithmetic/comparison/control-flow/stack/cell/array sections from `runmat-ignition/src/vm.rs`

### `accel/`

- Owns optional fusion graph and native acceleration integration.
- Keeps feature-gated acceleration logic separated from the core host interpreter.
- `residency.rs` is the shared home for residency policy and fusion/barrier decisions used by both analysis and execution.
- `idioms/` is the home for deterministic detection/lowering of accel-specific math idioms into specialized VM opcodes.

Current migrated ownership:

- stochastic-evolution idiom detection/lowering in `idioms/stochastic_evolution.rs`
- statement-idiom dispatch in `idioms/mod.rs`

Port source:

- `runmat-ignition/src/accel_graph.rs`
- accel-related sections from `runmat-ignition/src/vm.rs`
- `runmat-ignition/src/fusion_stack_layout.rs`

## Port order

1. `bytecode/`
2. `interpreter/api.rs` public parity layer
3. `runtime/`
4. `call/`
5. `indexing/`
6. `ops/`
7. `object/`
8. `compiler/`
9. `accel/`

## Risk areas to keep semantically frozen

- user-function invocation and output-count behavior
- `varargin` / `varargout`
- indexing and slicing, especially `end`, colon, logical selection, and shape rules
- store paths for indexed assignment
- class, struct, member, and static dispatch fallbacks
- global / persistent / workspace behavior
- try/catch error translation and source span attachment
- accel graph stack contracts and fusion residency behavior

## Shared semantic centers

These modules exist to prevent the refactor from recreating the same logic across many opcode handlers.

### `call/shared.rs`

Use for:

- collecting call arguments from the stack in source order
- cell expansion and multi-arg expansion handling
- user-function frame setup and teardown
- common output shaping and padding behavior
- shared call-site error wrapping

### `indexing/selectors.rs`

Use for:

- selector normalization and classification
- GPU-backed selector gathering when required
- colon, logical, and range selector handling
- shared shape and index planning helpers
- residency rules tied to the indexed base

### `interpreter/errors.rs`

Use for:

- attach span from PC
- attach call frames
- namespace-aware runtime error construction
- try/catch exception translation

### `interpreter/stack.rs`

Use for:

- small helpers for popping fixed operand bundles
- collecting `N` stack arguments while preserving source order
- shared operand decoding helpers for opcode handlers

### `runtime/workspace.rs`

Use for:

- workspace import/export and pending workspace state
- updated-workspace tracking returned to callers
- assignment/remove semantics that should not live in opcode handlers

### `runtime/globals.rs`

Use for:

- global and persistent storage
- named alias handling and shared variable resolution helpers
- lifecycle rules for global and persistent values

### `object/resolve.rs`

Use for:

- classify base values as struct, object, or class reference
- resolve fields vs dependent properties vs methods
- common access-check and fallback lookup rules
- static target resolution shared by read, write, and call paths

### `accel/residency.rs`

Use for:

- value residency policy helpers
- fusion barrier and eligibility helpers
- shared interpretation of plan metadata used by graph building and runtime execution

## Working rule for each port step

For any subsystem move:

1. move code with minimal edits
2. keep names until the move is stable
3. run the relevant tests
4. only then clean up local structure
