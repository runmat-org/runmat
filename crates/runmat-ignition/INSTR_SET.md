# Instruction Set (Instr)

This document describes each opcode’s semantics, stack/locals effects, and failure modes. It complements the high‑level README.

Notation:
- Stack top is on the right. `[...]` shows stack content before → after.
- `V` is a `Value`. Errors use mex identifiers.

## Data/Stack
- LoadConst(c): [] → [c]
- LoadString(s): [] → [String(s)]
- LoadVar(i): [] → [locals[i]]
- StoreVar(i): [v] → [] (locals[i] = v)

## Arithmetic/Logical/Relational
- Binary ops pop two, push one. Numeric fallbacks; object cases route to runtime overloads.
- Errors: `MATLAB:ShapeMismatch` for incompatible tensor elementwise shapes.

## Transpose
- Transpose: [V] → [V'] (delegates to runtime). Errors propagate.

## Indexing
- Index: legacy scalar `A(i)` and `A(i,j)` fast paths.
- IndexCell: cell content indexing `C{...}`; builds linear or 2‑D shapes as needed.
- IndexSlice(dims, numeric_count, colon_mask, end_mask): N‑D gather. Validates masks, `end`, ranges; returns scalar or tensor. Errors: `MATLAB:IndexOutOfBounds`, `MATLAB:UnsupportedIndexType`.
- IndexRangeEnd/Index1DRangeEnd: range forms with `end` arithmetic.

## Stores
- StoreIndex/StoreIndexCell: legacy fast paths.
- StoreSlice(dims, numeric_count, colon_mask, end_mask): N‑D scatter with broadcast; errors on shape mismatch.

## Functions
- CallBuiltin(name, argc): pops args, pushes 1 result.
- CallBuiltinExpandMulti/At/Last: expansion variants.
- CallFunction(name, argc): user function call, 1 result.
- CallFunctionMulti(name, argc, outc): multi‑assign; produces `outc` results on the stack.
- Return/ReturnValue: return from current frame.

## Objects
- LoadMember/StoreMember: instance property get/set with access checks.
- LoadMethod/CallMethod: instance method lookup and call.
- LoadStaticProperty/CallStaticMethod: classref static access.

## Control Flow
- Conditional jumps, EnterTry/PopTry for try/catch.

See `vm.rs` for exact semantics and `compiler.rs` for lowering patterns.
