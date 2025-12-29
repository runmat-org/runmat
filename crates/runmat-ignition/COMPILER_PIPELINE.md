# Ignition Compiler Pipeline (HIR → Bytecode)

This document specifies how `runmat-ignition` lowers the high-level IR (HIR) emitted by the parser into VM bytecode and how names, functions, objects, and indexing semantics are realized at compile-time.

The VM is a stack machine: each instruction pops values from the top of the stack and pushes results back. Heavyweight numeric and object semantics are delegated to `runmat-runtime` builtins; the compiler focuses on control flow, name resolution, structural lowering, and correct stack discipline.

## Inputs and Outputs

- Input: `runmat_hir::HirProgram`
- Output: `Bytecode` (in `functions.rs`):
  - `instructions: Vec<Instr>`: linear VM program
  - `var_count: usize`: number of global/local slots required by this unit
  - `functions: HashMap<String, UserFunction>`: user-defined functions discovered in this unit (including synthesized closures)

`Compiler::new` pre-scans the HIR to compute `var_count` by visiting variable IDs found in statements and expressions.

## High-level Pass Structure

Compilation is single-pass over statements with small local analyses. A preliminary sweep records imports and global/persistent declarations so name resolution and storage bindings are stable.

1) Validate program invariants
   - `validate_imports(prog)` for duplicate/ambiguous imports
   - `validate_classdefs(prog)` for class attribute/name conflicts
2) Pre-collect declarations and emit them:
   - `Import`: `RegisterImport { path, wildcard }`
   - `Global/Persistent`: named forms are emitted so the VM can bind thread locals at runtime
3) Compile all other statements sequentially.

## Name Resolution and Imports

Unqualified names at call-sites are resolved at compile-time using the following precedence:

1. Local variables / constants handled directly by HIR kinds (`Var`, `Constant`)
2. User functions defined in the current compilation unit
3. Specific imports: `import pkg.foo` resolves `foo` → `pkg.foo`
4. Wildcard imports: `import pkg.*` resolves `foo` → `pkg.foo`
5. Static class methods: `import MyClass.*` may resolve `foo` → `CallStaticMethod("MyClass", "foo", ...)` if unambiguous

Builtins are looked up first by unqualified name, then via specific imports, then via wildcard imports. Static properties can also be compiled under `Class.*` wildcard when unambiguous. Ambiguities produce compile-time errors.

Constants (`HirExprKind::Constant`) are resolved against `runmat_builtins::constants()`. If not found, the compiler attempts unqualified static property lookup via `Class.*` imports.

## Expressions

The compiler is responsible for stack shaping and choosing appropriate instructions. Representative cases:

- Numbers/strings/chars:
  - `Number` → `LoadConst`
  - `String('...')` → `LoadCharRow` (char row vector)
  - `String("...")` → `LoadString` (scalar string)
  - `Constant` → constant lookup (`LoadConst`/`LoadBool`/`LoadComplex`) or `LoadStaticProperty` via Class.* imports

- Unary and binary ops:
  - Emit left then right then the operator instruction (e.g., `Add`, `ElemMul`, `Pow`)
  - Logical `!` lowers to `x == 0`
  - Short-circuit `&&`/`||` use conditional jumps emitting only the necessary side (see “Short-circuit lowering” below)

- Ranges:
  - `start[:step]:end` → push components then `CreateRange(has_step)`

- Indexing and slicing:
  - Pure numeric: `Index(n)`
  - Mixed `:`, `end`, vector/range/logical: `IndexSlice(dims, numeric_count, colon_mask, end_mask)`
  - `end - k` in numeric positions: `IndexSliceEx(..., end_offsets)`
  - Range endpoints using `end` per-dimension: `IndexRangeEnd` or 1-D fast-path `Index1DRangeEnd`

- Literals (tensor/cell):
  - Pure numeric rectangular matrices: `CreateMatrix(rows, cols)`
  - Mixed/dynamic: push all elements row-major + row lengths then `CreateMatrixDynamic(rows)`
  - Cells: `CreateCell2D(rows, cols)` (rectangular) or ragged fallback using the same opcode with `(1,total)`
  - Special case: `[C{:}]` lowers to `CallBuiltinExpandMulti("cat", specs)` with first arg fixed `2` and second argument expand-all from the cell

- Function calls:
  - User function: `CallFunction(name, argc)`
  - Builtin: `CallBuiltin(name, argc)`
  - Comma-list expansion from `C{...}` arguments uses `Call*ExpandMulti` with `ArgSpec` per argument
  - `feval(f, ...)` uses `CallFeval` / `CallFevalExpandMulti` for dynamic dispatch (closures, handles, strings)
  - If an argument is a user function call, the compiler can “inline-expand” its multiple outputs into the caller’s argument list via `CallFunctionMulti` + packing

- Anonymous functions and closures:
  - Free variables are discovered (`collect_free_vars`) and captured by value in the order of first appearance
  - A synthetic `UserFunction` is created with captures prepended to parameters
  - `CreateClosure(synth_name, capture_count)` is emitted with capture values on the stack

- Member access and methods:
  - Instance field: `LoadMember(field)` / `StoreMember(field)`
  - Dynamic field: `LoadMemberDynamic` / `StoreMemberDynamic`
  - Instance method call: `CallMethod(name, argc)`
  - Static property/method via `classref('T')` or metaclass literal: `LoadStaticProperty` / `CallStaticMethod`

## Statements

- Expression statement: compile expr then `Pop`
- Assignment: compile RHS then `StoreVar(var_id)`
- If / ElseIf / Else: emit `JumpIfFalse` guards per branch and backpatch end jumps
- While: loop header guard with a single `JumpIfFalse` to loop end; keep `break`/`continue` stacks to backpatch
- For-range: requires a `Range` expression; the compiler emits:
  - Initialize `var`, `end_var`, `step_var` (with default step 1)
  - If `step == 0` jump to loop end
  - Conditional form depends on step sign: `var <= end` for non-negative, `var >= end` otherwise
  - Body, handle `continue`, then increment `var += step`
- Switch: compare scrutinee against each case, chain `JumpIfFalse`, collect end jumps
- Try/Catch: `EnterTry(catch_pc, catch_var?)`, compile try body, `PopTry`, jump over catch; then compile catch and backpatch
- Global/Persistent/Import: named variants emitted up-front and repeated if they occur in function bodies for binding in VM
- Function definitions: materialize `UserFunction` entries (not executed inline)
- Class definitions: lowered to a single `RegisterClass` carrying static metadata for runtime registration

### Multi-assign

`[a,b,c] = rhs` lowers as follows:

- If `rhs` is a user function: emit `CallFunctionMulti(name, argc, outc)` where `outc == len([a,b,c])`, then `StoreVar`/`Pop` right-to-left
- If `rhs` is builtin/unknown: `CallBuiltinMulti(name, argc, outc)` and distribute results similarly
- If `rhs` is `C{...}` (cell indexing): `IndexCellExpand(num_indices, outc)`
- Otherwise: first real variable gets `expr`, others receive `0` (matlab-compatible defaulting in many test paths)

### L-values (Index and Member assignment)

- Numeric-only indexing: push base and indices, compile RHS (with user-function packing optimization for 1-D), `StoreIndex(n)` then write-back to variable/member
- Slices (`:`, `end`, ranges, vectors, logical): compute masks and numeric-count, compile RHS (attempting vector packing for function returns or cell expand), then `StoreSlice`/`StoreSliceEx`/`StoreRangeEnd`/`StoreSlice1DRangeEnd` as appropriate, finally store back to the base variable or member
- Cell assignment: `StoreIndexCell(n)`
- Member and dynamic-member assignment re-evaluate the base when necessary to perform `StoreMember`/`StoreMemberDynamic` and then store updated object back to the root variable if applicable

## Short-circuit lowering

`a && b`:

1. compile `a`, compare to 0 (`NotEqual`), `JumpIfFalse` over RHS path
2. compile `b`, compare to 0, unconditional jump to end
3. false path pushes `0`

`a || b`:

1. compile `a`, compare to 0, `JumpIfFalse` to RHS path
2. true path pushes `1`, jump to end
3. RHS path compiles `b != 0`

## Objects and Classes

Objects are mediated by runtime registries:

- Instance access: `LoadMember`/`StoreMember` check access and dependent-property behavior; when absent, fall back to `subsref`/`subsasgn` if provided by the class
- Methods: `CallMethod` dispatches to `ClassName.method` or generic `name` builtins; `LoadMethod` returns a closure bound to the receiver
- Static members: `LoadStaticProperty`/`CallStaticMethod` with class names from `classref('T')` or metaclass literal; `RegisterClass` installs definitions at runtime

## Diagnostics

Compiler errors are returned as `Err(String)`. Runtime errors are normalized by the VM via the mex error model (see `ERROR_MODEL.md`).

For instruction-by-instruction semantics see `INSTR_SET.md`. For gather/scatter details see `INDEXING_AND_SLICING.md`.
