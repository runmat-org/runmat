# Plan 0: Replace the Core HIR Type Model

## Objective

Replace the current `VarId`-centered HIR surface with the new semantic HIR model directly, without introducing a separate long-lived parallel HIR.

This stage is about establishing the new compiler vocabulary, ownership invariants, diagnostic scaffolding, and stable identity vocabulary in `runmat-hir`. It is acceptable for downstream crates to be temporarily broken during this stage because the repo is intentionally paused for the migration.

## Desired Resting State

`runmat-hir` defines the new semantic source of truth:

- `HirAssembly`
- `HirModule`
- `HirEntrypoint`
- `HirFunction`
- `HirClass`
- `HirBinding`
- semantic statement, expression, place, call, import, and class metadata types
- MATLAB compatibility semantic vocabulary for compatibility modes, source units, evaluation context, requested outputs, comma-separated lists, function ABI, indexing, place mutation, command syntax, workspace/environment effects, function handles, core array semantics, operators, numeric classes, strings/chars, aggregates, and control flow
- semantic async constructs for async functions/blocks, future creation, `await`, and `spawn`
- spawn-safety scaffolding for captures and values crossing task boundaries
- stable symbol identity scaffolding such as `DefPath`
- diagnostic structures that can carry primary/secondary spans and fix/help information

The old public HIR model is removed or renamed out of the primary path:

- `HirProgram` is no longer the primary compiler product
- `HirStmt::Function` is removed as the durable function representation
- `HirStmt::ClassDef` is removed as the durable class representation
- `HirExprKind::FuncCall(String, ...)` is replaced by semantic call structure
- inline anonymous function bodies are replaced by `FunctionId` references
- `VarId` no longer represents semantic binding identity

## Core Invariants

- `HirAssembly` owns canonical tables for modules, functions, classes, bindings, and entrypoints.
- `HirModule` owns module identity, imports, top-level function IDs, class IDs, and optional synthetic entry function identity.
- `HirFunction` is the uniform executable representation for named functions, nested functions, anonymous functions, synthetic entry functions, and class methods.
- `HirFunction` owns a `FunctionAbi` describing fixed/variadic inputs, fixed/variadic outputs, and implicit `nargin`/`nargout` bindings.
- `HirClass` owns class metadata and method membership through `FunctionId` references.
- `HirBinding` represents semantic binding identity, not VM slot layout.
- Captures are relations on `HirFunction`, not duplicate bindings.
- VM slots are not represented in core HIR.
- Workspace visibility is semantic binding metadata, not a consequence of slot numbering.
- Local arena IDs are not cross-session cache identities.
- Stable qualified identities use package/module/item paths.

## Primary Files

- `runmat/crates/runmat-hir/src/ids.rs`
- `runmat/crates/runmat-hir/src/hir.rs`
- `runmat/crates/runmat-hir/src/lib.rs`

## Secondary Files Likely To Break

- `runmat/crates/runmat-hir/src/lowering/ctx.rs`
- `runmat/crates/runmat-hir/src/lowering/stmt.rs`
- `runmat/crates/runmat-hir/src/lowering/expr.rs`
- `runmat/crates/runmat-hir/src/inference/`
- `runmat/crates/runmat-hir/src/remapping.rs`
- Downstream crates that import old HIR names directly

## Implementation Plan

1. Add the semantic ID set in `ids.rs`.

Required IDs:

- `ModuleId`
- `FunctionId`
- `ClassId`
- `EntrypointId`
- `BindingId`
- `ExprId`
- `StmtId`
- keep `SourceId`
- remove or quarantine `VarId` from the primary HIR model

2. Replace the top-level HIR product.

Introduce:

- `HirAssembly`
- `HirModule`
- `HirEntrypoint`
- `SourceUnitKind`
- `EntrypointOrigin`
- `EntrypointPolicy`
- `QualifiedName`

3. Replace function representation.

Introduce:

- `HirFunction`
- `FunctionKind`
- `FunctionName`
- `FunctionModifiers`
- `FunctionAbi`
- `CapturedBinding`

Remove `HirStmt::Function` from the target statement model.

4. Replace variable identity with binding identity.

Introduce:

- `HirBinding`
- `BindingOwner`
- `BindingRole`
- `BindingStorage`
- `WorkspaceVisibility`

5. Replace statement shape.

Introduce:

- `HirBlock`
- `HirStmt { id, kind, span }`
- `HirStmtKind`

Preserve source-fidelity statements for declarations such as `global`, `persistent`, and `import`, while moving semantic ownership to bindings/modules.

6. Replace assignment targets.

Introduce `HirPlace` with variants for:

- binding
- member
- dynamic member
- index
- cell index
- indexing components including colon, logical indexes, and symbolic `end`
- place mutation metadata including creation and deletion policies

Remove the split between plain assignment and lvalue assignment in the target HIR.

7. Replace expression shape.

Introduce:

- `HirExpr { id, kind, span }`
- `HirExprKind`
- `HirCall`
- `HirCallableRef`
- `CallSyntax`
- `RequestedOutputs`
- `OutputTargetList`
- `ValueFlowFact`
- command-call syntax nodes
- function handle target metadata
- explicit async expression/function markers
- explicit await expression form
- explicit spawn expression/call classification

Replace string-only calls with semantic call references where possible and unresolved qualified names where not yet resolved.

MATLAB compatibility vocabulary should include:

- `EvaluationContext`
- `CompatibilityMode`
- `SourceUnitKind`
- `RequestedOutputCount`
- `OutputTarget`
- `ValueFlowFact`
- `FunctionAbi`
- `PlaceMutation`
- `IndexingSemantics`
- `ReferenceKind`
- `CallKind`
- `WorkspaceEffect`
- `EnvironmentEffect`
- `FunctionHandleTarget`
- `EmptyArrayRole`
- `ExpansionSemantics`
- `OperatorKind`
- `NumericClass`
- string/char semantic markers
- `AggregateKind`
- `LoopIterationSemantics`
- tensor element-domain facts

Async source forms should follow the target Rust-like model:

- async functions and async blocks produce lazy future values
- creating a future does not execute user code
- `await` is an explicit suspension point
- `spawn` is the explicit concurrency boundary that schedules a future and returns a task handle
- `spawn` requires spawn-safe captures and must not alias mutable parent frame storage
- ordinary MATLAB-style calls remain sequential source constructs

8. Replace class representation.

Introduce:

- `HirClass`
- `ClassKind`
- `ClassProperty`
- `ClassPropertyModifiers`
- `ClassMethod`
- class dispatch metadata for constructors, static members, operators, and overloaded indexing
- class event, enumeration, argument block, and access-level metadata

Remove `HirStmt::ClassDef` from the target statement model.

9. Update `lib.rs` exports.

Export the new semantic HIR surface as the primary public API of `runmat-hir`.

10. Add stable identity scaffolding.

Introduce initial qualified identity concepts such as:

- package name
- module qualified name
- `DefPath`
- source identity

These do not need full project composition yet, but the HIR model should not imply that local numeric IDs are stable outside a compiler product.

11. Add diagnostic scaffolding.

Ensure diagnostics can represent:

- stable code
- severity
- primary span
- secondary spans
- notes/help
- optional suggestions

Full diagnostic production can come later, but the type model should not bake in string-only errors.

12. Add type-level smoke tests.

Tests should verify that the new structures can be constructed and that the key ownership invariants are expressible. These tests should not attempt full lowering behavior yet.

## Explicit Non-Goals

- Do not preserve a fully operational legacy HIR path.
- Do not migrate VM lowering in this stage.
- Do not implement full semantic lowering in this stage.
- Do not implement project manifests, source roots, dependencies, or entrypoint config.
- Do not implement MIR in this stage; that is Plan 2.
- Do not implement the complete analysis store yet.
- Do not encode runtime/provider residency in HIR types or bindings.
- Do not implement async scheduling or future polling in this stage.

Type/analysis scaffolding note:

- It is acceptable to define lightweight `TypeFact`, `ShapeFact`, `ValueFlowFact`, `WorkspaceEffect`, `EnvironmentEffect`, `SpawnSafetyFact`, function-handle, tensor element-domain, and execution-fact shells if needed to keep HIR definitions and downstream planning aligned.
- The full type lattice, shape lattice, and analysis propagation rules belong in Plan 2.
- Any acceleration-related fact in this stage must be eligibility or policy-hint oriented, not concrete runtime residency.

## Acceptance Criteria

- `runmat-hir` has the new semantic HIR type surface as its primary public model.
- The old `HirProgram` / `VarId` / `HirStmt::Function` shape is no longer the primary API.
- The new ownership invariants are documented in code near the relevant type definitions.
- Local IDs and stable qualified identities are explicitly distinguished.
- Diagnostic scaffolding supports structured diagnostics rather than string-only errors.
- `runmat-hir` type-level tests pass, even if downstream crates are still broken.

## Known Temporary Breakage

Downstream crates should be expected to fail until Plans 2-3 restore the MIR/analysis layer and active runtime/tooling consumers:

- `runmat-vm`
- `runmat-core`
- `runmat-static-analysis`
- `runmat-lsp`
- `runmat-snapshot`
- `runmat-turbine`

This breakage is acceptable only on the migration branch.
