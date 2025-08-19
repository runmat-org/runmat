# RunMat HIR

High-level Intermediate Representation for MATLAB code. HIR is the semantic hub between parsing and execution (interpreter/JIT). It resolves identifiers to `VarId`s, attaches static types, normalizes constructs, and runs early semantic validations so downstream components can be simpler and faster.

## Goals

- Provide a typed, SSA-friendly structure for the engine
- Preserve MATLAB semantics (indexing, cells, classes, methods, metaclass)
- Enable flow-sensitive inference and optimizations (constant folding, dispatch)
- Catch structural and attribute errors early (classdef attributes, imports)

## Core data structures

- `VarId(usize)`: stable variable identifiers after name binding
- `Type` (from `runmat-builtins`):
  - `Int`, `Num`, `Bool`, `String`
  - `Tensor { shape: Option<Vec<Option<usize>>> }` (column-major semantics)
  - `Cell { element_type: Option<Box<Type>>, length: Option<usize> }`
  - `Function { params: Vec<Type>, returns: Box<Type> }`
  - `Struct { known_fields: Option<Vec<String>> }` (inference-only)
  - `Void`, `Unknown`, `Union(Vec<Type>)`
- `HirExpr { kind, ty }` (selected variants):
  - Literals and names: `Number`, `String`, `Var(VarId)`, `Constant`
  - Ops: `Unary`, `Binary`
  - Aggregates: `Tensor`, `Cell`, `Range`, `Colon`, `End`
  - Indexing: `Index`, `IndexCell`
  - Calls and members: `FuncCall`, `FuncHandle`, `AnonFunc`, `Member`, `MemberDynamic`, `MethodCall`
  - Metaclass: `MetaClass("pkg.Class")`
- `HirStmt` (selected variants):
  - `ExprStmt(expr, suppressed)` (semicolon suppression)
  - `Assign(VarId, expr, suppressed)`
  - `MultiAssign(Vec<Option<VarId>>, expr, suppressed)` with `~` as `None`
  - `AssignLValue(HirLValue, expr, suppressed)` where `HirLValue` ∈ { `Var`, `Member`, `MemberDynamic`, `Index`, `IndexCell` }
  - Control flow: `If`, `While`, `For`, `Switch`, `TryCatch`
  - Declarations: `Function { name, params, outputs, body, has_varargin, has_varargout }`, `Global`, `Persistent`
  - Flow control: `Break`, `Continue`, `Return`
  - Class: `ClassDef { name, super_class, members }`
  - Imports: `Import { path: Vec<String>, wildcard: bool }`
- `HirClassMember`: `Properties`, `Methods`, `Events`, `Enumeration`, `Arguments` (carry `parser::Attr` attributes)
- `HirProgram { body }`

## Lowering (AST → HIR)

- `Ctx` manages scopes, binds names to `VarId`, and maintains `var_types` for flow typing.
- Variables shadow constants; bare identifiers that are known functions lower to `FuncCall(name, [])`.
- Indexing vs calls is already disambiguated by the parser; HIR keeps `Index`/`IndexCell` and `FuncCall` distinct.
- L-values lower to `HirLValue` for dot/paren/brace writes. Plain `A(…) = v` is `AssignLValue`.
- `Function` statements record `has_varargin`/`has_varargout` flags.
- `ClassDef` lowers structurally into `HirClassMember` blocks with attributes preserved.
- `Import` lowers to a dedicated `HirStmt::Import` (no runtime effect; used by name resolution/validation).
- Metaclass `?Qualified.Name` lowers to `HirExprKind::MetaClass("Qualified.Name")`; postfix is handled in the compiler.
- Function-level `arguments ... end` blocks (when present) are parsed; names are accepted and exposed to later validation. Constraint checking (types/defaults/ranges) is enforced at HIR/VM time rather than parsing time.

## Early validations and helpers

- `validate_classdefs(&HirProgram)` runs during `lower()`:
  - Detects duplicate properties/methods and name conflicts between them
  - Enforces attribute constraints (e.g., Methods: `Abstract` ∧ `Sealed` invalid; Properties: `Static` ∧ `Dependent` invalid; `Access`/`GetAccess`/`SetAccess` values limited to `public|private`)
  - Performs basic sanity checks for `Events`, `Enumeration`, and `Arguments` (unique names; no conflicts with props/methods)
- Imports:
  - `collect_imports(&HirProgram)`
  - `normalize_imports(&HirProgram) -> Vec<NormalizedImport { path, wildcard, unqualified }]`
  - `validate_imports(&HirProgram)` checks duplicates and ambiguity among specifics with the same unqualified name
- Multi-LHS structural validation: lowering rejects invalid LHS shapes early (e.g., empty LHS vectors, unsupported mixed forms); shape/size rules are enforced by the interpreter at assignment.
- Globals/Persistents: a per-program symbol set is collected across units to model lifetimes and name binding consistently.

## Type inference (expressions)

- Numbers/strings/booleans map to `Num`/`String`/`Bool`.
- Arithmetic/elementwise ops: if any operand is `Tensor`, result is `Tensor` (shape may unify when known).
- Range/colon produce `Tensor`.
- Indexing computes output type conservatively. For tensors with known rank, scalar indices drop dimensions.
- Cells compute a unified element type across literals when possible.
- Member/Method calls are `Unknown` by default (value-dependent at runtime).
- Metaclass expression has `String` type.

## Flow-sensitive inference

Two complementary passes exist:

1) Inter-procedural return summaries

- `infer_function_output_types(&HirProgram) -> HashMap<String, Vec<Type>>`
  - Gathers all function names (top-level and class methods)
  - Seeds summaries from each function's own exits/fallthrough, then iterates to a small fixed point (cap at 3 iters)
  - Merges types at joins; Unknown ⊔ T = T; otherwise unify
  - Uses an internal `analyze_stmts(outputs, …, func_returns)` whose env joins propagate return types

2) Per-function variable environments

- `infer_function_variable_types(&HirProgram) -> HashMap<String, HashMap<VarId, Type>`
  - Similar dataflow that produces a final environment for each function
  - Uses return summaries from (1) to type `FuncCall`
  - Includes a simple callsite fallback for direct callees: when a callee's summary is missing/Unknown, a single-pass analysis of the callee body (seeding parameter types conservatively) infers direct output assignments. This stabilizes per-position types for `[a,b]=f(...)` at callers.

### Struct-field flow inference

- HIR uses `Type::Struct { known_fields: Option<Vec<String>> }` to conservatively track observed fields on variables.
- The analysis refines struct knowledge in two ways:
  - Writes: `s.field = expr` marks `s` as Struct and adds `"field"` to `known_fields`.
  - Conditions (then-branch refinement): detect any of the following and add asserted fields:
    - `isfield(s, 'x')`
    - `ismember('x', fieldnames(s))` or `ismember(fieldnames(s), 'x')`
    - `strcmp(fieldnames(s), 'x')` / `strcmpi(…)`, including `any(strcmp(…))` or `all(strcmp(…))`
    - Conjunctions using `&&` or `&` are traversed; negations are ignored (no refinement)
- Refinements are applied to the then-branch env only and merged back at joins using `Type::unify` for Structs.

## Multi-assign typing

- `[a,b] = f(...)` is typed per-position using the callee's return summary when available.
- If a summary is incomplete or missing, a simple fallback (single-pass over the callee) infers direct assignments to outputs and fills Unknowns conservatively.
- Mixed forms like `[~,b] = f(...)` are handled by storing `None` in the LHS vector and skipping the slot.

## Function call typing

- Builtins: signatures come from the registry (`runmat-builtins`).
- User functions: return summaries and the per-position logic above are used for accurate call result typing in both expression and `MultiAssign` contexts.

## Remapping utilities

- `remapping::create_function_var_map`, `create_complete_function_var_map`
- `remapping::remap_function_body` / `remap_stmt` / `remap_expr` to rewrite `VarId`s for local execution frames
- `remapping::collect_function_variables` scans bodies to compute complete maps

## Public entry points

- `lower(&AstProgram) -> Result<HirProgram, String>`: lowers AST, runs return-summary inference (for seeding), then validates classes
- `lower_with_context` / `lower_with_full_context`: lowering for REPL with preexisting variables/functions
- Validation helpers: `validate_classdefs`, `collect_imports`, `normalize_imports`, `validate_imports`
- Inference helpers: `infer_function_output_types`, `infer_function_variable_types`

## Testing

- Mirrors parser coverage for syntax constructs; adds HIR-specific tests:
  - L-value lowering (member/paren/brace), multi-assign and `~` placeholder
  - Control-flow joins across if/elseif/else, switch/otherwise, while/for loops, try/catch
  - Class attribute validation (invalid combos, duplicates, conflicts)
  - Import normalization/ambiguity checks
  - Fuzz seeds for lowering edge cases

## Notes and differences from MATLAB

- MATLAB is dynamically typed; HIR attaches conservative static types for optimization only. Programs acceptable to MATLAB remain acceptable; Unknown is used when insufficient info.
- Column-major Tensor semantics are preserved throughout indexing/slicing/shape operations.
- Class blocks are carried structurally; access/attribute validations run during lowering; advanced OOP attributes may have future passes.
- Metaclass expressions are represented explicitly; postfix static member/method usage is compiled appropriately downstream.

## Roadmap / future enhancements

- Inter-procedural propagation of struct field knowledge across calls
- Deeper OOP attribute validations (Hidden/Constant/Transient interplay; static/instance access rules)
- Richer import resolution summaries for static method/property lookup in the HIR stage
- Shape reasoning improvements for Tensor broadcasting and indexing

## Remaining edges

- Arguments metadata: carry `arguments ... end` declared names/constraints (when available from parser) and surface to runtime validation. Current parser accepts names; HIR will add optional metadata structs without breaking format.
- Multi-LHS validation: parser structurally restricts to identifiers/`~`; HIR enforces shape semantics at runtime. Additional unit tests exist; no further work is blocking.
- Globals/Persistents: cross-unit name binding is wired; additional tests around nested functions/closures will be added.

## Minimal example

MATLAB:

```
function y = f(s)
  if isfield(s, 'x') && any(strcmp(fieldnames(s), 'y'))
    s.y = 1;
  end
  y = g(s.x);
end
```

HIR sketch:

```
Function { name: "f", params: [s], outputs: [y], ... }
  If { cond: FuncCall("isfield", [Var(s), String('x')]) && any(strcmp(fieldnames(s),'y')), then: [ AssignLValue(Member(Var(s),'y'), Number(1)) ] }
  Assign(Var(y), FuncCall("g", [Member(Var(s), "x")]))
```

Return summaries infer type of `g`'s first output if available; variable analysis refines `s` as a Struct with fields `{x,y}` along the then-branch.