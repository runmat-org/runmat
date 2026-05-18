# Plan 1: Rewrite Lowering Around Semantic Ownership

## Objective

Make `runmat-hir` lower parser ASTs directly into the new semantic HIR model introduced in Plan 0, including an explicit resolver product.

This stage turns the new HIR type surface into a real compiler product. It should establish modules, entrypoints, functions, classes, bindings, captures, statement IDs, expression IDs, resolver products, diagnostics, and best-effort call references from actual source code.

## Desired Resting State

`runmat_hir::lower` returns a semantic `HirAssembly` or a lowering result whose primary payload is a semantic `HirAssembly`.

The emitted HIR should represent:

- script-like top-level code as a synthetic entry function
- top-level functions as module-owned `HirFunction`s
- nested functions as `HirFunction`s with parent relations
- anonymous functions as `HirFunction`s referenced by `FunctionId`
- class definitions as `HirClass` items
- class methods as ordinary `HirFunction`s referenced by `HirClass`
- variables as semantic `HirBinding`s
- globals and persistents as binding storage metadata plus source-fidelity declaration statements
- captures as explicit relations on the capturing function
- imports as module-level semantic imports and source-fidelity statements where needed
- context-sensitive MATLAB constructs such as compatibility modes, source unit kinds, command syntax, requested-output calls, comma-list expansion, indexed assignment growth, symbolic `end` indexing, function handles, empty-array roles, expansion semantics, operators, aggregates, and control-flow semantics
- resolver products for lexical bindings, functions, classes, imports, builtins, and unresolved/dynamic references
- structured diagnostics for resolution and lowering failures

## Core Invariants

- Every executable body belongs to exactly one `HirFunction`.
- Top-level executable source code belongs to a synthetic entry function, not to a raw module body.
- Functions are not statements.
- Classes are not statements.
- Every parameter, output, local, global, persistent, module binding, and implicit `ans` has a `BindingId`.
- Nested functions reference parent bindings directly through capture relations.
- Captured bindings are not duplicated.
- `isolated` functions are normal functions with an `isolated` modifier and zero captures.
- Any attempted capture by an `isolated` function is a lowering or early semantic error.
- Ambiguous or dynamic name resolution is explicitly represented.
- Async-capable calls are preserved semantically for MIR/analysis rather than hidden behind runtime callbacks.
- Compatibility mode, source unit kind, requested-output count, command syntax, assignment context, indexing context, function handle identity, operator kind, aggregate kind, and control-flow semantics are preserved semantically for MIR/analysis.

## Primary Files

- `runmat/crates/runmat-hir/src/lowering/ctx.rs`
- `runmat/crates/runmat-hir/src/lowering/stmt.rs`
- `runmat/crates/runmat-hir/src/lowering/expr.rs`
- `runmat/crates/runmat-hir/src/lowering/mod.rs`
- `runmat/crates/runmat-hir/src/hir.rs`
- `runmat/crates/runmat-hir/src/error.rs`

## Secondary Files

- `runmat/crates/runmat-hir/src/validation/`
- `runmat/crates/runmat-hir/tests/`
- `runmat/crates/runmat-parser/src/ast.rs` if parser support for `isolated` is not already present
- `runmat/crates/runmat-parser/src/parser/` if parser support for `isolated` is not already present

## Implementation Plan

1. Redesign lowering context around semantic arenas.

The lowering context should allocate and store:

- modules
- entrypoints
- functions
- classes
- bindings
- statement IDs
- expression IDs
- evaluation context stack
- compatibility mode
- source unit kind
- requested-output context
- command statement context

2. Add lexical lowering state.
Track:

- current module
- current function
- current class, if any
- lexical scope stack
- name-to-binding maps per scope
- function declarations visible in the current module/function scope
- class declarations visible in the current module scope
- source-unit-local declaration inventory
- resolution precedence state
- capture candidates and capture relations
- implicit function ABI state for `nargin`, `nargout`, `varargin`, and `varargout`

3. Lower source/program root to module plus entrypoint.

For script-like input:

- create one module
- create one synthetic entry function
- create one `HirEntrypoint` targeting that function with source/host origin and workspace export policy
- lower top-level executable statements into the synthetic entry body
- mark assigned top-level bindings as `WorkspaceVisibility::TopLevel`

For function-oriented input:

- create one module
- create top-level function items
- if execution requires an entrypoint, create one targeting the selected root function

4. Split declarations from executable statements.

Pre-scan statement lists where needed to collect function and class declarations before lowering executable bodies. This allows calls to local functions to resolve to `FunctionId` even when declarations appear later in source.

5. Lower function declarations to `HirFunction`.

For each function:

- allocate `FunctionId`
- allocate parameter bindings
- allocate output bindings
- build `FunctionAbi` with fixed inputs, fixed outputs, optional `varargin`, optional `varargout`, and implicit `nargin`/`nargout` bindings
- reuse one binding when an output name matches an input name
- set parent function for nested functions
- lower body into `HirBlock`
- record local bindings
- record captures
- attach to module, parent, or class inventory as appropriate

6. Lower anonymous functions to `HirFunction`.

For each anonymous function expression:

- allocate a new `FunctionId`
- allocate parameter bindings
- lower expression body as the function body or expression-return block
- record captures from enclosing lexical scopes
- return `HirExprKind::AnonymousFunction(function_id)`

7. Lower bindings semantically.

Binding roles:

- parameter
- output
- local
- module binding
- implicit `ans`

Binding storage:

- lexical
- global
- persistent

Workspace visibility:

- hidden
- top-level
- module-visible
- implicit `ans`

8. Lower global and persistent declarations.

For declarations:

- ensure a binding exists
- set binding storage to global or persistent
- emit declaration statements preserving source information
- avoid requiring later passes to infer storage semantics by walking raw statements

9. Lower classes to semantic class items.

For each class:

- allocate `ClassId`
- determine class kind when possible
- lower properties into class property metadata
- lower methods into `HirFunction`s
- attach method functions through `ClassMethod`
- preserve events, enumerations, and argument blocks structurally
- preserve class dispatch metadata for constructors, static members, operators, property attributes, and overloaded indexing where source syntax exposes it

10. Lower calls to semantic call forms.

Call resolution should be best-effort in this stage:

- local function calls should resolve to `HirCallableRef::Function`
- class constructor calls should resolve when obvious
- builtin calls should resolve to stable builtin IDs and metadata when known
- imported calls may remain `Imported` or `Unresolved` depending on available information
- dynamic expression calls should use `DynamicExpr`
- unresolved calls should retain qualified spelling without pretending to be resolved
- requested-output context should be attached to call sites
- command calls should lower from command syntax into ordinary semantic calls with string/option arguments
- comma-list expansion sites should be explicit rather than hidden in runtime argument passing
- function handle targets should preserve direct, builtin, anonymous, method, `DefPath`, or dynamic-name identity
- operator syntax should lower to explicit `OperatorKind`

Resolution should explicitly classify MATLAB-style ambiguous forms such as `foo(x)` as binding indexing, direct function call, builtin call, imported call, class construction, dynamic call, or unresolved reference.

Resolution should preserve precedence-relevant facts for lexical bindings, local/nested functions, imports, private functions, package-qualified names, class/static/member lookups, source-path functions, runtime classes, builtins, and dynamic fallback.

11. Lower MATLAB assignment, indexing, and workspace-effect semantics.

Lowering should preserve:

- assignment target lists and discarded outputs
- assignment creation policy for plain bindings, indexed array growth, and struct field creation
- deletion assignment with `[]`
- paren, brace, and dot indexing mode
- colon, logical indexes, and symbolic `end` components
- command syntax statement boundaries
- workspace-effecting operations such as `load`, `clear`, `global`, `persistent`, `eval`, `evalin`, and `assignin` as explicit effects or effect candidates
- environment-effecting operations such as `addpath`, `rmpath`, `path`, `cd`, and `rehash` as explicit effects or effect candidates
- empty-array role context for expression, concatenation, and deletion uses
- scalar and implicit expansion context
- string vs char literal identity
- aggregate construction/access for structs, cells, and object arrays
- control-flow semantics for `for`, `if`, `while`, short-circuit operators, and `switch/case`

12. Add capture analysis during or immediately after lowering.

Capture behavior:

- a reference to an outer function-owned binding from a nested function records a capture relation
- assignments to captured bindings target the same original `BindingId`
- captures do not create new bindings
- `isolated` functions reject any lexical capture
- captures that may cross a `spawn` boundary are marked for spawn-safety analysis
- mutable lexical captures from the spawning frame are rejected for `spawn` unless wrapped in an explicit synchronized/runtime-managed object

13. Remove or replace legacy inference hooks inside lowering.

The old lowering currently computes inferred globals, function envs, and function returns as part of `LoweringResult`. In this stage, either remove those fields or replace them with minimal placeholders until Plan 2 introduces the new analysis model.

14. Keep runtime placement out of semantic lowering.

Lowering may preserve source constructs such as `gpuArray` calls or explicit runtime-domain class references, but it should not infer concrete device residency, provider choice, buffer identity, or materialization state. Those are runtime/provider decisions.

15. Preserve async and host-interaction candidates.

Lowering should retain enough semantic information for Plan 2 to mark async boundaries in MIR. Examples include provider-backed calls, plotting/host interaction calls, input calls, filesystem/package operations, and any builtin metadata that indicates async behavior.

Lowering should also preserve user-facing Rust-like async semantics directly:

- async functions and async blocks lower as lazy future-producing constructs
- async function calls produce future values and do not execute the body immediately
- `await` lowers as an explicit source suspension expression, not an ordinary builtin call
- `spawn` lowers as an explicit scheduling/concurrency boundary that consumes a future and returns a task handle
- `spawn` also records a task-boundary use requiring spawn-safe captures and values
- ordinary MATLAB-style calls stay language-synchronous even if their implementation may internally suspend

Diagnostics should reject `await` in non-async-capable ordinary function bodies, while allowing top-level `await` when the selected `EntrypointPolicy` permits it.

Diagnostics should reject obviously non-spawn-safe futures at lowering time when the capture relation is local and unambiguous. More complex cases can be finalized by Plan 2 analysis.

## Tests To Add Or Rewrite

Add focused `runmat-hir` tests for:

- script-like top-level code lowers to module plus synthetic entry function
- script plus local function creates one synthetic entry function and one module-owned function
- function parameters and outputs create explicit bindings
- `varargin` and `varargout` populate `FunctionAbi`
- `nargin` and `nargout` lower as implicit function-local bindings
- shared input/output names reuse one semantic binding
- multi-output assignment records requested output count and target list
- discarded outputs are represented explicitly
- `varargin{:}` and cell expansion sites are represented explicitly
- command syntax lowers only in statement context and terminates at newline/semicolon
- indexed assignment to an undefined root is represented as creation policy rather than an early undefined-variable error
- `end` in indexing is preserved with operand/dimension context
- compatibility mode and source unit kind are preserved
- function handles preserve target identity and captures
- empty array roles, expansion semantics, operator kind, numeric class, string/char distinctions, aggregate kind, and loop/control-flow semantics are represented where visible in source
- nested shared lexical capture records one captured parent binding
- captured mutation targets the same parent binding
- `isolated` nested function with no capture succeeds
- invalid `isolated` capture fails with a clear diagnostic
- `spawn` of a future with mutable parent lexical capture fails with a clear diagnostic
- anonymous function lowers to a real `HirFunction`
- global and persistent declarations set binding storage metadata
- class method lowers to a `HirFunction` referenced by `HirClass`
- imports are attached to the module
- simple local function call resolves to `HirCallableRef::Function`
- ambiguous/unresolved references are represented or diagnosed explicitly
- async-capable call metadata is preserved for MIR lowering
- async function/block creation lowers without eager execution semantics
- `await` and `spawn` are represented explicitly instead of as string-keyed calls

## Explicit Non-Goals

- Do not restore all downstream crates in this stage.
- Do not implement MIR in this stage.
- Do not implement the full project manifest model.
- Do not implement final import/package/dependency resolution.
- Do not implement complete type, shape, effect, and execution fact stores.
- Do not encode concrete runtime residency or provider state in HIR.
- Do not optimize VM slot layout yet.

## Acceptance Criteria

- `runmat-hir` lowers representative MATLAB-style programs into semantic HIR.
- The new semantic HIR tests pass.
- Functions, classes, bindings, captures, and entrypoints are populated from source.
- `runmat-hir` no longer depends on statement-walking reconstruction as its primary semantic model.
- Downstream crates may still be broken, but the HIR producer itself is coherent and testable.

## Known Temporary Breakage

Downstream crates still need MIR/analysis restoration in Plan 2 and active consumer restoration in Plan 3. The expected failures are imports, type mismatches, and removed legacy HIR variants in:

- `runmat-vm`
- `runmat-core`
- `runmat-static-analysis`
- `runmat-lsp`
- `runmat-snapshot`
- `runmat-turbine`
