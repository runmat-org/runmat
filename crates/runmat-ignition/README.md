# RunMat Ignition (Interpreter)

Ignition is RunMat’s reference interpreter. It compiles the high‑level IR (HIR) emitted by `runmat-hir` into a compact bytecode and executes it on a fast, non‑allocating stack virtual machine. The design goal is full MATLAB grammar/semantic compatibility with clear, testable semantics and a codebase that is easy to extend.

Sibling docs for deep dives:
- INSTR_SET.md – opcode semantics, stack effects, failure modes
- COMPILER_PIPELINE.md – HIR → bytecode lowering strategy
- INDEXING_AND_SLICING.md – column‑major rules, `end` arithmetic, N‑D gather/scatter, expansion
- ERROR_MODEL.md – MException and mex identifier normalization
- OOP_SEMANTICS.md – objects, properties/methods, `subsref`/`subsasgn`, operator overloading

## Module structure
- `instr.rs`: Instruction set enum `Instr` (bytecode opcodes)
- `functions.rs`: `UserFunction`, frames, and `ExecutionContext`
- `compiler.rs`: exhaustive lowering for `HirStmt`/`HirExprKind`
- `bytecode.rs`: `compile`, `compile_with_functions`
- `vm.rs`: Interpreter loop and semantics (`execute`)
- `gc_roots.rs`: RAII GC roots for stack/locals/globals

## Public API
- `compile(&HirProgram) -> Bytecode`
- `compile_with_functions(&HirProgram, &HashMap<String, UserFunction>) -> Bytecode`
- `interpret(&Bytecode) -> Result<Vec<Value>, String>`
- `interpret_with_vars(&Bytecode, &mut [Value]) -> Result<Vec<Value>, String>`
- `execute(&HirProgram) -> Result<Vec<Value>, String>`

## Execution model
- Each function compiles to bytecode + constants. Calls create frames holding PC, locals, return arity, and a base stack pointer.
- The VM uses a single operand stack of `Value`. Locals are a separate array per frame. Both are GC roots.
- Column‑major semantics are global invariants. All indexing, reshape, broadcast and assignment follow MATLAB order.

## Compatibility surface
- Expressions: numbers/strings/identifiers/constants; unary (+, -, transpose, non‑conjugate transpose .'), logical not (~); arithmetic (+, -, *, /, ^), elementwise (.*, ./, .^); comparisons; ranges (`start[:step]:end`).
- Tensors/Cells: 2‑D literals, transpose, indexing, stores.
- Indexing: scalar/vector/range/logical/colon/`end`; N‑D gather via `IndexSlice`; 1‑D/2‑D fast‑paths preserved.
- Slice assignment: N‑D scatter via `StoreSlice` with broadcast/shape checks.
- Control flow: if/elseif/else, while, for (±/0 step), break/continue, switch/case/otherwise.
- Try/catch: nested via `EnterTry`/`PopTry`; catch variable bound to `Value::MException`.
- Functions/Recursion: fixed arity + varargs; `nargin`/`nargout` accounted; multi‑assign via `CallFunctionMulti`.
- Handles/Closures: captures and `feval` supported.
- OOP: properties/methods (instance/static), access checks, class refs/metaclass, overloaded indexing routed to `subsref`/`subsasgn`.

## Compiler principles
- Evaluation order is preserved (left‑to‑right) to match MATLAB side‑effects.
- Control flow uses structured blocks with patched jumps.
- Multi‑assign lowers to `CallFunctionMulti` with explicit arity then sequential stores.
- Indexing lowers to 2‑D fast paths where safe, otherwise `IndexSlice`. Stores mirror via `StoreSlice`.

## Indexing and slicing (summary)
- Column‑major gather: cartesian enumeration in column‑major with shape normalization (2‑D `(I,scalar)` → `[|I|,1]`, `(scalar,J)` → `[1,|J|]`).
- `end` arithmetic per dimension; logical masks are validated and expanded to indices.
- Slice assignment enforces exact shape/broadcast rules; scalar broadcasts along selected extents.
- Cells and function returns expand in argument lists and (when applicable) into slice targets.

## Error model (mex)
- All interpreter errors are normalized: `mex(id, message)` returns `"<ID>: <message>"`.
- Identifiers include: `MATLAB:UndefinedFunction`, `MATLAB:UndefinedVariable`, `MATLAB:NotEnoughInputs`, `MATLAB:TooManyInputs`, `MATLAB:TooManyOutputs`, `MATLAB:VarargoutMismatch`, `MATLAB:SliceNonTensor`, `MATLAB:IndexOutOfBounds`, `MATLAB:CellIndexType`, `MATLAB:CellSubscriptOutOfBounds`, `MATLAB:ExpandError`, `MATLAB:MissingSubsref`, `MATLAB:MissingSubsasgn`.

## OOP semantics (summary)
- Instance: `LoadMember`/`StoreMember`, `LoadMethod`/`CallMethod` with access checks.
- Static: `LoadStaticProperty`/`CallStaticMethod` when base is a class reference.
- Overloaded indexing: the VM constructs selector cells and routes to `subsref`/`subsasgn`. Missing hooks surface mex identifiers above.
- Operator overloading: delegated through runtime dispatch (e.g., `plus`, `mtimes`), with numeric fallbacks normalized.

## Testing strategy
- Unit/property tests cover:
  - Gather/scatter round‑trip invariants, column‑major mapping, broadcast laws.
  - Negative paths: invalid indices, arity/output errors, expansion/type mismatches, OOP dispatch failures.
  - Parser+compiler+VM end‑to‑end for command‑form, metaclass postfix, imports.
- Tests assert mex identifiers to prevent semantic drift.

## Performance
- Correctness‑first generic paths, plus 2‑D fast paths for `A(:, j)` and `A(i, :)` with strict shape checks and column‑major writes.
- Unhandled opcodes in JIT (Turbine) explicitly fall back to the interpreter.

## Contributing
- Add builtins in `runmat-runtime`; wire special handling in the VM only if semantics require.
- Prefer centralizing MATLAB rules in `IndexSlice`/`StoreSlice` rather than scattering logic.
- When adding opcodes, document in INSTR_SET.md and ensure mex errors are uniform.
