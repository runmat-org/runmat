# Compiler Pipeline (HIR → Bytecode)

This document explains how `runmat-ignition` lowers HIR to bytecode.

## Overview
- The compiler performs a single forward walk over HIR statements with small local analyses (e.g., short-circuit lowering, multi-assign shaping).
- No global SSA or register allocation: we use a stack machine model for clarity and debuggability.

## Expressions
- Emit code for left operand, then right, then apply operator. For short-circuit `&&`/`||`, emit conditional jumps over RHS emission.
- Function calls: push args left-to-right, emit `CallBuiltin`/`CallFunction`. Multi-assign sites drive `CallFunctionMulti` with explicit `outc`.
- Indexing: when patterns match 2-D fast forms `(I, scalar)`/`(scalar, J)`, emit specialized shape logic; otherwise emit `IndexSlice` with masks.

## Statements
- Assign/ExprStmt: compile RHS then store or leave on stack; statement results not captured are dropped.
- If/ElseIf/Else: compile condition; branch via jumps; patch labels after blocks.
- While/For: loop headers and bodies with patched back-edges; for ranges normalize step and direction.
- Switch: chain of comparisons + blocks; optional otherwise.
- Try/Catch: wrap try body with `EnterTry`/`PopTry`; catch body runs with the caught exception bound in locals.
- Multi-assign: `[a,b]=f(...)` compiles to `CallFunctionMulti + StoreVar` sequence.

## Objects
- Members/methods compile to dedicated opcodes; static accesses require class references (`classref(...)`) or metaclass `?T` lowering in the parser + HIR.
- Overloaded indexing: emit helpers that build selector cells and route to `subsref`/`subsasgn` in the VM.

## Names and locals
- Locals are assigned contiguous indices; closures capture by value in `Value::Closure`.

For precise opcode semantics, refer to `INSTR_SET.md`. Indexing details live in `INDEXING_AND_SLICING.md`.
