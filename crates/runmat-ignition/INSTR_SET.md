# Instruction Set (Instr)

This document describes each opcode’s semantics, stack/locals effects, and failure modes. It complements the high‑level README.

Notation:
- Stack top is on the right. `[...]` shows stack content before → after.
- `V` is a `Value`. Errors use mex identifiers.

Data/Stack and Locals
---------------------
- LoadConst(c): [] → [c]
- LoadString(s): [] → [String(s)]
- LoadCharRow(s): [] → [CharArray(1,len(s))]
- LoadVar(i): [] → [locals[i]]
- StoreVar(i): [v] → [] (locals[i] = v)
- EnterScope(n): [] → [] (reserve n new locals)
- ExitScope(n): [] → [] (release n locals)
- LoadLocal(i): [] → [frame[i]]
- StoreLocal(i): [v] → [] (frame[i] = v)
- Swap: [a, b] → [b, a]
- Pop: [v] → []

Arithmetic/Logical/Relational
-----------------------------
- UPlus: [+x] → [x]
- Neg: [x] → [-x]
- Transpose: [V] → [V'] (delegates to runtime)
- Binary ops: Add, Sub, Mul, Div, LeftDiv, Pow, ElemMul, ElemDiv, ElemLeftDiv, ElemPow
- Relations: Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual
- Short-circuit lowering uses JumpIfFalse and constants; see compiler.
- Errors: shape/broadcast mismatches route to runtime mex identifiers.

Control Flow
------------
- JumpIfFalse(tgt): [cond] → [] (0 is false)
- Jump(tgt): [] → []
- EnterTry(catch_pc, catch_var?): [] → [] (push try frame)
- PopTry: [] → [] (pop try frame)
- Return: [] → (exit)
- ReturnValue: [v] → [v] (return with top value)

Construction
------------
- CreateMatrix(rows, cols): [... elements row-major] → [Tensor]
- CreateMatrixDynamic(rows): rows with varying lengths pushed as constants precede elements
- CreateCell2D(rows, cols): [... elements row-major] → [Cell]
- CreateRange(has_step): [start, [step], end] → [Tensor indices]

Indexing (gather)
-----------------
- Index(n): [base, i1, ..., in] → [V] (numeric only)
- IndexSlice(dims, numeric_count, colon_mask, end_mask): base-first, then numeric indices; colon/end per masks
- IndexSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets): end_offsets are (numeric_position, offset) for end-k
- IndexRangeEnd { dims, numeric_count, colon_mask, end_mask, range_dims, range_has_step, end_offsets }: base, per-range start[,step], then numeric
- Index1DRangeEnd { has_step, offset }: base, start[,step]
- IndexCell(n): [base, idx1..idxn] → [contents]
- IndexCellExpand(n, outc): expand `C{...}` into `outc` stack values (col-major)

Stores (scatter)
----------------
- StoreIndex(n): [base, i1..in, rhs] → [updated_base]
- StoreIndexCell(n): [base, i1..in, rhs] → [updated_base]
- StoreSlice(dims, numeric_count, colon_mask, end_mask): [base, numeric..., rhs] → [updated_base]
- StoreSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets): like StoreSlice with end‑k offsets
- StoreSlice1DRangeEnd { has_step, offset }: [base, start[,step], rhs] → [updated_base]

Packing and Comma‑lists
-----------------------
- PackToRow(n): [v1, ..., vn] → [Tensor 1×n]
- PackToCol(n): [v1, ..., vn] → [Tensor n×1]

Calls and Expansion
-------------------
- CallBuiltin(name, argc): pops `argc`, pushes 1
- CallBuiltinExpandLast(name, fixed_argc, num_indices)
- CallBuiltinExpandAt(name, before_count, num_indices, after_count)
- CallBuiltinExpandMulti(name, specs: Vec<ArgSpec>)
- CallFunction(name, argc): compile‑time resolved user function
- CallFunctionMulti(name, argc, outc): push `outc` return values
- CallFunctionExpandAt(name, before_count, num_indices, after_count)
- CallFunctionExpandMulti(name, specs)
- CallFeval(argc): dynamic resolution at runtime (closures, handles)
- CallFevalExpandMulti(specs)

Objects and Classes
-------------------
- LoadMember(field): [obj] → [value]
- LoadMemberDynamic: [obj, name] → [value]
- StoreMember(field): [obj, rhs] → [updated_obj]
- StoreMemberDynamic: [obj, name, rhs] → [updated_obj]
- LoadMethod(name): [obj] → [closure]
- CallMethod(name, argc): [obj, args...] → [ret]
- LoadStaticProperty(class, prop): [] → [value]
- CallStaticMethod(class, method, argc): [] → [ret]
- RegisterClass { name, super_class, properties, methods }

Imports and Globals
-------------------
- RegisterImport { path, wildcard }
- DeclareGlobal(ids)
- DeclarePersistent(ids)
- DeclareGlobalNamed(ids, names)
- DeclarePersistentNamed(ids, names)

Error model
-----------
All runtime failures are surfaced via mex identifiers (e.g., `MATLAB:IndexOutOfBounds`, `MATLAB:TooManyInputs`, 
`MATLAB:MissingSubsref`). See error model document and `vm.rs` for exact messages.

Lowering notes
--------------
See `compiler.rs` for how high‑level constructs map to opcodes (e.g., ranges with end arithmetic, multi‑assign, 
object/property access, feval/expansion composition).
