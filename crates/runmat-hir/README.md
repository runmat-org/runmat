# RunMat HIR

High-level Intermediate Representation for MATLAB code. HIR is the semantic hub
between parsing and execution (interpreter/JIT). It resolves identifiers to
`VarId`s, annotates types, and normalizes constructs for efficient downstream
execution.

## Goals

- Provide a typed, SSA-friendly structure for the engine
- Preserve MATLAB semantics (indexing, cells, methods, class members)
- Enable type inference and optimizations (constant folding, dispatch)

## Core types

- `VarId(usize)`: stable variable identifiers after name binding
- `Type`: imported from `runmat-builtins` (Num, Bool, String, Matrix, Unknown, Void, etc.)
- `HirExpr { kind, ty }` with variants:
  - Numbers, strings, variables, constants
  - Unary and binary ops (incl. element-wise, logical, transpose)
  - Matrix and cell literals; indexing `Index`, `IndexCell`
  - Ranges (`start[:step]:end`), colon (`:`) and `End` sentinel
  - Member/method access; function calls; anonymous functions; function handles
- `HirStmt` with variants:
  - Expression statements (with suppression flag)
  - Assignment and multi-assignment
  - (Lowered placeholder) Complex lvalue assignments (member/paren/brace) are currently represented as effectful `ExprStmt` for side effects; interpreter implements write semantics
  - Control flow: if/elseif/else, while, for, switch/otherwise, try/catch
  - Declarations: function, global, persistent
  - Break/continue/return
  - Class definitions: name, optional superclass, members
- `HirClassMember`: properties, methods (lowered body), events, enumeration, arguments
- `HirProgram { body }`

## Lowering (AST → HIR)

- Name resolution: `Ctx` maintains nested scopes and maps identifiers to `VarId`
- Type tracking: `var_types[VarId]` updated on assignments; expressions carry `ty`
- Indexing vs calls: parser disambiguates; HIR keeps both forms (`Index`, `FuncCall`)
- Cells, methods, members lowered to dedicated variants
- Globals/persistents mapped to VarIds for consistent runtime handling
- Import statements are lowered as no-ops (carried as `ExprStmt` placeholder) since they affect name resolution rather than runtime
- Metaclass `?Qualified.Name` lowers to a string literal for now (future: dedicated HIR node if needed)

## Type inference

- Expressions infer types via operator rules and builtin signatures
- Function return types:
  - Builtins: taken from registry signature
  - User-defined: flow-sensitive dataflow across a CFG-like analysis
    - Track VarId → Type through statements
    - At control-flow joins (if/switch/loops), join types (Unknown ⊔ T = T; T ⊔ U ≠ T = Unknown)
    - Collect environments at return sites and fallthrough; compute per-output types
    - Unassigned outputs remain Unknown (or Empty if modeled)

## Remapping utilities

- `remapping::create_function_var_map` and `create_complete_function_var_map`
- `remap_function_body`, `remap_stmt`, `remap_expr`: rewrite VarIds for function-local contexts
- Variable collectors for building complete maps and analyses

## Testing

- Ensure parity with parser tests for constructs
- Add tests to validate inference joins across if/else, switch, loops, early returns, and multi-assign
- Validate cell/indexing/method lowering and class member lowering

## Notes

- HIR remains conservative where semantics depend on runtime values; Unknown types are permitted and handled by the runtime
- Class members are lowered structurally; semantic checks (access, attributes) are future work

## How HIR differs from MATLAB (and why)

- MATLAB is dynamically typed; HIR attaches static types to expressions and variables to enable optimization. This is an internal representation only:
  - Type inference is conservative and never rejects programs that MATLAB would accept. Unknown is used when information is insufficient.
  - No user‑visible “compile‑time” type errors are introduced by HIR; runtime semantics remain MATLAB‑compatible.
- Control‑flow–sensitive return‑type inference for user functions (CFG‑like join of types at merges) goes beyond MATLAB’s dynamic behavior and exists to guide optimizations and codegen.
- Normalization choices:
  - `end` index sentinel is a first‑class `Expr` in HIR.
  - Member/method access, cells, and function handles/anonymous functions are represented explicitly to simplify codegen.
  - Class blocks are carried structurally in HIR; access control/attributes are deferred to later phases.
  - Command syntax is normalized at the parser layer into `FuncCall`; HIR treats both uniformly.
- Disambiguations (e.g., call vs indexing, `.'` vs member access) are made explicitly in HIR to avoid ambiguity in later stages, but semantics are preserved.

