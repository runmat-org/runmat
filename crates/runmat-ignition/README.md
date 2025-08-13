RunMat Ignition (Interpreter)
================================

Overview
--------
Ignition compiles HIR (`runmat-hir`) to a compact bytecode (`Instr`) and executes it with a fast stack VM. It is the reference, fully-featured interpreter backend for RunMat with a strong focus on MATLAB compatibility and performance.

Module structure
----------------
- `instr.rs`: Instruction set enum `Instr` (bytecode opcodes).
- `functions.rs`: `UserFunction`, call frames, and `ExecutionContext`.
- `compiler.rs`: HIR → bytecode compiler; exhaustive lowering for all `HirStmt`/`HirExprKind`.
- `bytecode.rs`: Public compile helpers (`compile`, `compile_with_functions`).
- `vm.rs`: Interpreter loop (`interpret`, `interpret_with_vars`, `execute`).
- `gc_roots.rs`: RAII GC root management for stack, locals, and globals.

Public API
----------
- `compile(&HirProgram) -> Bytecode`
- `compile_with_functions(&HirProgram, &HashMap<String, UserFunction>) -> Bytecode`
- `interpret(&Bytecode) -> Result<Vec<Value>, String>`
- `interpret_with_vars(&Bytecode, &mut [Value]) -> Result<Vec<Value>, String>`
- `execute(&HirProgram) -> Result<Vec<Value>, String>`

What’s implemented today
------------------------
- Expressions: numbers, strings, identifiers, constants, unary (+, -, transpose, non-conjugate transpose .'), logical not (~), arithmetic (+, -, *, /, ^), elementwise (.*, ./, .^), comparisons (==, ~=, <, <=, >, >=), ranges (start[:step]:end).
- Tensors and cells: 2D tensor and cell literals, concatenation, creation, transpose.
- Indexing (generic): scalar and vector/range selectors, `:` colon, `end` arithmetic, logical masks, N-D gather via `IndexSlice`.
- Slice assignment (generic): N-D scatter via `StoreSlice` with per-dimension selectors.
- Linear indexing and 2D A(i, j) variants preserved with column-major semantics.
- Control flow: if/elseif/else, while, for (positive/negative/zero step handling), break/continue, switch/case/otherwise.
- Try/catch: core semantics via `EnterTry`/`PopTry`. Catch variable bound to `Value::MException`.
- Functions: user-defined functions, recursion, local variable isolation, multiple calls.
- Multi-assign: `[a,b] = f(...)` via `CallFunctionMulti` + stores.
- Function handles and closures: `Value::FunctionHandle`, `Value::Closure` with captured environment; nested captures supported; `feval` and direct invocation supported via `CallFeval`.
- Object system (initial runtime semantics): `Value::Object`, property read/write (`LoadMember`/`StoreMember`), instance method calls (`LoadMethod`/`CallMethod`), static property/method access on class references (`LoadStaticProperty`/`CallStaticMethod`). Access control checks enforced.
- Meta-class and class refs: `LoadClassRef`/`LoadMetaClass` integration.
- Imports: parsed/lowered; currently a no-op at runtime.

Key bytecode opcodes (high level)
---------------------------------
- Data/stack: `LoadConst`, `LoadString`, `LoadVar`, `StoreVar`, arithmetic/relational ops, elementwise ops, `Transpose`.
- Tensors/cells: `CreateMatrix`, `CreateMatrixDynamic`, `CreateCell2D`.
- Indexing: `Index`, `IndexCell`, `IndexSlice(dims, numeric_count, colon_mask, end_mask)`; stores: `StoreIndex`, `StoreIndexCell`, `StoreSlice`.
- Functions: `CallBuiltin`, `CallFunction`, `CallFunctionMulti`, `Return`, `ReturnValue`, `CreateClosure`, `CallFeval`, `LoadFunctionHandle`.
- Objects: `LoadMember`, `StoreMember`, `LoadMethod`, `CallMethod`, `LoadStaticProperty`, `CallStaticMethod`, `LoadClassRef`, `LoadMetaClass`.
- Control flow: conditional jumps, `EnterTry`, `PopTry`.

Execution model
---------------
- The compiler lowers HIR into a linear instruction stream. The VM executes it using a value stack, a locals array, and GC roots that cover both.
- Column-major ordering is used throughout to match MATLAB semantics.

Feature status grid (tracked subset)
------------------------------------
| Area                 | Status       | Notes |
|----------------------|-------------:|------|
| Arithmetic/Elemwise  | Implemented  | Delegates to `runmat-runtime` where applicable |
| Comparisons          | Implemented  | |
| Logical (&&, ||, ~)  | Implemented  | Short-circuiting in compiler |
| Transpose/.'         | Implemented  | |
| Ranges/Colon         | Implemented  | `CreateRange`, colon via `IndexSlice` |
| End keyword          | Implemented  | Encoded via `end_mask` in `IndexSlice` |
| 1D/2D Indexing       | Implemented  | Scalar, vector, logical |
| N-D Indexing (gather)| Implemented  | `IndexSlice` generic path |
| N-D Slice assignment | Implemented  | `StoreSlice` generic path |
| Cells (2D)           | Implemented  | `CreateCell2D`, `IndexCell`, stores |
| Multi-assign         | Implemented  | `CallFunctionMulti` + stores |
| Functions/Recursion  | Implemented  | |
| Closures/Handles     | Implemented  | Nested captures; `feval` |
| Try/Catch            | Implemented  | Basic propagation; `MException` binding |
| Switch/case/otherwise| Implemented  | |
| Objects: members     | Implemented  | Load/store with access checks |
| Objects: methods     | Implemented  | Instance/static, access checks |
| Class refs/metaclass | Implemented  | |
| Imports              | Implemented  | No-op at runtime |
| JIT (Turbine)        | Fallback     | New opcodes explicitly fallback to interpreter |

Active work-in-progress (context to resume)
-------------------------------------------
At the time of writing, the following multi-dimensional indexing tests were failing and are being addressed next:

- `tests/multid_indexing.rs`:
  - `mixed_selectors_basic_2d_range`: ensure `(I, scalar)` produces a column vector in column-major order; current gather returns a broader shape. Fix: adjust `IndexSlice` 2D shape logic.
  - `logical_mask_rows_select`: confirm shape and element order for row logical masks in column-major; fix gather to build correct output tensor shape.
  - `reshape_and_index_3d_element`: verify `reshape` uses column-major mapping; fix either test expectation or `reshape` usage in tests.
  - `slice_assignment_3d_entire_slice`: scatter semantics produce interleaved zeros (column-major). Adjust scatter or test to index-based assertions that match MATLAB column-major placement.

Immediate next steps
--------------------
1. Indexing semantics
   - Tighten `vm.rs` `IndexSlice` gather for 2D selections:
     - `(I, scalar)` → shape `[len(I), 1]` (column vector);
     - `(scalar, J)` → shape `[1, len(J)]` (row vector).
   - Ensure logical mask selection preserves column-major order and correct shape computation.
   - Verify 3D reshape/index tests against MATLAB’s column-major semantics; update expectations if needed.

2. Exceptions
   - Expand `try/catch` to cover nested rethrow, consistent `MException` propagation across builtins.

3. OOP essentials (second pass)
   - Constructors with args, `subsref`/`subsasgn` dispatch hooks, operator overloading. Keep access checks intact.

4. Multi-assign and comma-list expansion
   - Complete behavior for cells and function returns in all contexts.

5. Performance
   - Specialize hot `IndexSlice/StoreSlice` cases (common 2D forms) and avoid unnecessary allocations.

Running tests
-------------
- All interpreter tests:
  - `cargo test -p runmat-ignition --tests`
- Focus a single file:
  - `cargo test -p runmat-ignition --test multid_indexing -- --test-threads=1`
- Enable test-only classes used by OOP tests:
  - Add feature: `--features test-classes`
  - Example: `cargo test -p runmat-ignition --test functions --features test-classes`

Notes
-----
- Column-major ordering is a core invariant; tests and semantics are aligned with MATLAB.
- New instructions unsupported by Turbine JIT are intentionally marked to fallback to the interpreter.
