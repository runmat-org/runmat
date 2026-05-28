# Plan 4: Complete MATLAB Core Semantics

## Objective

Complete the MATLAB compatibility semantics introduced in the target model on top of the semantic HIR, MIR, analysis, and restored runtime pipeline from Plans 0-3.

Plan 3 brings the repo back to green on the new compiler architecture. Plan 4 turns that architecture into a complete MATLAB core semantics implementation rather than a set of point fixes.

## Desired Resting State

RunMat correctly represents, analyzes, lowers, and executes MATLAB core language semantics for:

- compatibility modes
- source unit kinds
- resolution precedence
- function handles and dynamic callable effects
- requested-output calls and comma-separated lists
- function ABI details including `nargin`, `nargout`, `varargin`, and `varargout`
- assignment growth, deletion, scalar expansion, and place mutation
- indexing reads/writes including symbolic `end`
- empty-array roles and concatenation
- scalar and implicit expansion
- matrix and elementwise operators
- numeric classes and string/char distinctions
- struct arrays, cell arrays, and object arrays
- control-flow semantics including `for`-over-columns, scalar conditions, short-circuit operators, and `switch/case`
- workspace and environment effects such as `clear`, `load`, `eval`, `assignin`, `addpath`, `cd`, `which`, and `exist`

## Core Invariants

- MATLAB compatibility semantics are modeled as source/runtime semantics, not ad hoc builtin patches.
- HIR preserves source context; MIR normalizes semantics without losing required MATLAB behavior.
- Runtime behavior follows the Function ABI, value-flow, indexing, assignment, workspace-effect, and environment-effect products from analysis/lowering.
- Dynamic fallback is explicit and conservative.
- Unsupported dynamic features produce structured diagnostics, not runtime traps.
- Provider placement, GPU residency, and host details remain outside language semantics.

## Primary Crates

- `runmat-hir`
- `runmat-mir`
- `runmat-static-analysis`
- `runmat-vm`
- `runmat-runtime`
- `runmat-core`

## Secondary Crates

- `runmat-parser`
- `runmat-builtins`
- `runmat-lsp`
- `runmat-snapshot`
- `runmat-turbine`

## Implementation Plan

1. Implement compatibility mode plumbing.

Thread `CompatibilityMode` through parsing, lowering, diagnostics, LSP, and entrypoint policy.

Modes should gate RunMat-only syntax, top-level await, interactive workspace behavior, and extension diagnostics.

2. Implement source unit kind semantics.

Classify source units as scripts, function files, class files, package function files, class-folder method files, REPL submissions, or notebook cells.

Enforce source-unit rules for local function visibility, primary function identity, script entry functions, class files, `return`, and `arguments` blocks.

3. Implement MATLAB-compatible resolution precedence.

Resolver products should encode precedence across lexical bindings, nested/local functions, imports, private functions, class constructors/static members, package-qualified names, currently available source-path/host-policy functions, runtime metadata, builtins, and dynamic fallback.

Full manifest-backed project composition and dependency graph lookup remain Plan 5 work; Plan 4 should stabilize the resolver semantics that Plan 5 plugs into.

Variables should shadow functions in contexts where MATLAB does so. Package-qualified calls should remain distinct from member access.

4. Complete function handle semantics.

Implement `FunctionHandleTarget` for direct functions, builtins, methods, anonymous functions, stable `DefPath`s, and dynamic names.

Ensure `feval`, `arrayfun`, `cellfun`, `str2func`, and `func2str` use the same callable identity and dynamic-effect model.

5. Harden requested-output and comma-list semantics.

Finish output-list ABI behavior across user functions, builtins, function handles, object methods, assignment target lists, discarded outputs, `varargout`, `varargin{:}`, cell expansion, and struct-array field access.

6. Complete assignment, indexing, and aggregate semantics.

Implement assignment creation/growth, deletion with `[]`, scalar expansion, slice assignment shape rules, dynamic field names, struct array behavior, cell `()` vs `{}` behavior, object array indexing/property access, and object `end` dispatch through source-defined or currently available class metadata.

7. Complete empty-array, concatenation, and expansion semantics.

Implement `EmptyArrayRole`, horizontal/vertical/N-D concatenation, empty concat identity behavior, char/string/cell/struct/object/numeric concat rules, scalar expansion, and implicit expansion.

8. Complete numeric, operator, and string/char semantics.

Implement `NumericClass`, default double literal behavior, logical condition/index behavior, integer casting/promotion where supported, sparse as language-visible array representation, matrix vs elementwise operators, transpose vs conjugate transpose, and char/string compatibility. Runtime-provided builtin/class metadata completion remains Plan 6 work.

9. Complete control-flow semantics.

Implement `for x = A` column iteration, scalar condition validation for `if`/`while`, `&&`/`||` short-circuit behavior, elementwise logical operators, `switch/case` matching, and context-specific diagnostics for `break`, `continue`, and `return`.

10. Complete workspace and environment effects.

Implement or explicitly diagnose `load`, `clear`, `global`, `persistent`, `eval`, `evalin`, `assignin`, `addpath`, `rmpath`, `path`, `cd`, `pwd`, `rehash`, `which`, and `exist` according to compatibility and project policy.

Workspace/environment effects must invalidate or barrier analysis/runtime products where needed.

11. Add compatibility diagnostics and regression coverage.

Diagnostics should cover arity mismatch, invalid comma-list use, invalid assignment growth/deletion, invalid `end`, invalid condition shape/type, command syntax misuse, resolution precedence ambiguity, unsupported dynamic environment mutation, and class dispatch/access errors.

## Tests

Add tests for:

- compatibility mode gating and diagnostics
- source unit kind classification and local function visibility
- resolution precedence and shadowing
- function handles and dynamic callable behavior
- requested-output-sensitive builtins and methods
- comma-list expansion and consumption
- `varargin`, `varargout`, `nargin`, and `nargout`
- indexed assignment growth, deletion, scalar expansion, and symbolic `end`
- struct arrays, cell arrays, object arrays, and dynamic fields
- empty concat identity and deletion marker behavior
- scalar/implicit expansion and matrix vs elementwise operators
- numeric classes, logicals, sparse, char, and string behavior
- `for` column iteration, scalar conditions, short-circuit, and `switch/case`
- workspace/environment effects and invalidation behavior

## Acceptance Criteria

- MATLAB core semantics are represented by compiler/runtime products, not point fixes.
- Representative MATLAB scripts using multi-output calls, varargs, function handles, command syntax, indexing, structs/cells, concat/growth, control flow, and workspace/environment effects execute correctly.
- Unsupported dynamic or compatibility-mode-specific features produce structured diagnostics.
- The repo remains green after Plan 4.
- Plan 5 can build project/source composition on top of stable source-unit and resolution-precedence semantics.

## Explicit Non-Goals

- Do not implement full project manifests or dependency composition; that is Plan 5.
- Do not complete runtime class/builtin metadata generation; that is Plan 6.
- Do not complete accelerate/fusion/GC lifetime hardening; that is Plan 7.
- Do not require every MATLAB toolbox/domain object to be complete; model them through metadata and nominal/runtime classes where needed.
