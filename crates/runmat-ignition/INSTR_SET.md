# Instruction Set (Instr)

This document describes each opcode’s semantics, stack/locals effects, and failure modes. The VM is stack-based with column-major numeric semantics. For high-level lowering, see `COMPILER_PIPELINE.md`. For gather/scatter semantics, see `INDEXING_AND_SLICING.md`.

Notation:
- Stack top is on the right. `[...]` shows stack content before → after
- `Value` denotes a runtime value (Num, Int, Tensor, String, StringArray, Cell, Object, Struct, LogicalArray, Closure, HandleObject, ClassRef, CharArray, FunctionHandle)
- Errors are normalized via mex identifiers (see `ERROR_MODEL.md`)

## Data/Stack and Locals

- LoadConst(c): [] → [Num(c)]
- LoadBool(b): [] → [Bool(b)]
- LoadString(s): [] → [String(s)]
- LoadCharRow(s): [] → [CharArray(1×len(s))]
- LoadVar(i): [] → [vars[i]]
- StoreVar(i): [v] → []  assigns `vars[i] = v` and writes through to globals if bound
- EnterScope(n): [] → []  append `n` local slots to `ExecutionContext.locals`
- ExitScope(n): [] → []   pop `n` locals
- LoadLocal(i): [] → [locals[i]]  relative to current frame; falls back to `vars` in <main>
- StoreLocal(i): [v] → []  relative to current frame; writes through to persistents if bound
- Swap: [a, b] → [b, a]
- Pop: [v] → []

## Arithmetic / Logical / Relational

- UPlus: [+x] → [x] (object overload `uplus` if available)
- Neg: [x] → [-x] (object overload `uminus`, else numeric elementwise)
- Transpose: [V] → [V'] (delegates to runtime transpose)
- Add/Sub/Mul/Div/Pow: binary numeric; object overloads (`plus`, `minus`, `mtimes`, `mrdivide`, `power`) attempted first
- ElemMul/ElemDiv/ElemLeftDiv/ElemPow: elementwise ops with object overloads (`times`, `rdivide`, `ldivide`, `power`)
- Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual: numeric and array comparisons; object overloads (`eq`, `ne`, `lt`, `le`, `gt`, `ge`) with fallbacks; handle objects compare by identity via runtime
- AndAnd(target), OrOr(target): short-circuit variants (currently unused by the compiler; `JumpIfFalse` lowering is used instead)

## Control Flow

- JumpIfFalse(tgt): [cond] → []  jump if `cond == 0`
- Jump(tgt): [] → []
- EnterTry(catch_pc, catch_var?): [] → []  push try frame; on error jump to `catch_pc` and bind exception into `catch_var` if provided
- PopTry: [] → []
- Return: [] → exit current unit
- ReturnValue: [v] → [v] and exit

## Construction

- CreateMatrix(rows, cols): [... row-major elements] → [Tensor(rows×cols)] (reordered to column-major)
- CreateMatrixDynamic(rows): [rowlen_r, ..., elems...] → [Tensor] with ragged row handling delegated to runtime
- CreateCell2D(rows, cols): [... row-major values] → [Cell(rows×cols)]
- CreateRange(has_step): [start, [step], end] → [Tensor indices]

## Indexing (gather)

- Index(n): [base, i1, ..., in] → [value] numeric-only; objects route to `subsref(obj,'()',cell)
- IndexSlice(dims, numeric_count, colon_mask, end_mask): generic gather with colon/end and vector/logical
- IndexSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets): numeric `end-k` arithmetic
- IndexRangeEnd{dims, numeric_count, colon_mask, end_mask, range_dims, range_has_step, end_offsets}: N-D range gather with end arithmetic
- Index1DRangeEnd{has_step, offset}: 1-D range gather with `end-k`
- IndexCell(n): [base, idx...] → [contents] (1-D or 2-D supported; objects route to `subsref(obj,'{}',cell)`)
- IndexCellExpand(n, outc): expand cell contents into exactly `outc` stack values (pad with 0 if needed)

## Stores (scatter)

- StoreIndex(n): [base, i1..in, rhs] → [updated_base]
- StoreIndexCell(n): [base, i1..in, rhs] → [updated_base]
- StoreSlice(dims, numeric_count, colon_mask, end_mask): broadcasting-aware scatter
- StoreSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets): numeric `end-k` arithmetic + scatter
- StoreSlice1DRangeEnd{has_step, offset}: 1-D range with `end-k`

## Packing and Comma-lists

- PackToRow(n): [v1, ..., vn] → [Tensor 1×n]  values coerced to numeric
- PackToCol(n): [v1, ..., vn] → [Tensor n×1]

## Calls and Expansion

- CallBuiltin(name, argc): pops `argc`, pushes 1; tries imports (specific then wildcard) on failure
- CallBuiltinMulti(name, argc, outc): invoke builtin and push up to `outc` values (from tensors/cells or scalar+pads)
- CallBuiltinExpandLast(name, fixed_argc, num_indices): expand last argument from `C{...}`
- CallBuiltinExpandAt(name, before_count, num_indices, after_count): expand an argument in the middle
- CallBuiltinExpandMulti(name, specs: Vec<ArgSpec>): multi-position expansion; each `ArgSpec { is_expand, num_indices, expand_all }`
- CallFunction(name, argc): resolve user function and call; checks `nargin` mismatch (unless `varargin`)
- CallFunctionMulti(name, argc, outc): push `outc` user function results; supports `varargout` merging semantics
- CallFunctionExpandAt/CallFunctionExpandMulti: user function analogs of builtin expansion
- CallFeval(argc): dynamic function value (`Closure`, `'@name'`, `CharArray('@name')`, function handle)
- CallFevalExpandMulti(specs): as above with argument expansion

## Objects and Classes

- LoadMember(field): [obj] → [value] with access checks; dependent properties route to `get.<field>` builtin when present
- LoadMemberDynamic: [obj, name] → [value]
- StoreMember(field): [obj, rhs] → [updated_obj] with access checks; dependent properties route to `set.<field>` builtin when present
- StoreMemberDynamic: [obj, name, rhs] → [updated_obj]
- LoadMethod(name): [obj] → [Closure(Class.method, captures=[obj])]
- CallMethod(name, argc): [obj, args...] → [ret] via qualified builtin or registry; static misuse errors
- LoadStaticProperty(class, prop): [] → [value] with static/access checks (uses registry and defaults)
- CallStaticMethod(class, method, argc): [] → [ret] with static/access checks
- RegisterClass { name, super_class, properties, methods }: registers class metadata in runtime registry

## Imports, Globals, Persistents

- RegisterImport { path, wildcard }: record for VM-time builtin and static resolution
- DeclareGlobal(ids) / DeclareGlobalNamed(ids, names): bind local slots to thread-local global table
- DeclarePersistent(ids) / DeclarePersistentNamed(ids, names): bind slots to persistent tables keyed by function name

## Closures and Scoping

- CreateClosure(name, capture_count): captures popped then reversed into closure; later used by `CallFeval` and can be called as builtins if registered

## Error model

All runtime failures are surfaced via mex identifiers (e.g., `MATLAB:IndexOutOfBounds`, `MATLAB:TooManyInputs`, `MATLAB:MissingSubsref`). See `ERROR_MODEL.md` for principles and `vm.rs` for exact messages.

## Lowering notes

See `compiler.rs` for how high-level constructs map to opcodes (e.g., short-circuiting, ranges with end arithmetic, multi-assign/varargout, object/property access, and argument expansion composition).
