# Plan 6: Runtime Class, Builtin Metadata, And Nominal Dispatch

## Objective

Make Rust-defined runtime domain objects, source-defined RunMat classes, builtins, and graphics/domain objects participate in one coherent nominal class/type/builtin metadata system.

Plans 0-5 make classes, `ClassId`, nominal type facts, MIR, complete MATLAB core semantics, project composition, and resolver products real. Plan 6 makes runtime-provided classes and builtins first-class language metadata instead of ad hoc enum variants and string-based special cases.

## Desired Resting State

Rust runtime code can declare language-facing classes, methods, properties, dispatch behavior, builtin behavior, docs, async behavior, and analysis summaries once, and that metadata feeds:

- runtime class registration
- nominal type identity
- property and method lookup
- constructor, static member, operator, and indexing dispatch
- requested-output-sensitive builtin and method behavior
- argument parser modes such as name-value and line-spec parsing
- type and shape analysis summaries
- effect and async summaries
- LSP hover/completion/navigation
- docs/help output
- generated bindings such as TypeScript

Domain types such as `DataDataset`, `DataArray`, and `DataTransaction` move toward nominal runtime classes with metadata and summary hooks rather than permanent core type enum variants.

## Core Invariants

- Runtime Rust implementation details do not directly become language semantics.
- Rust declarations generate language-facing metadata.
- Source-defined classes and Rust-defined runtime classes share nominal class identity concepts.
- Structs remain structural.
- Classes remain nominal.
- Method/property lookup uses class metadata and respects inheritance where known.
- Constructor/static/operator/indexing dispatch uses class metadata rather than scattered string checks.
- Builtin metadata describes arity, requested-output behavior, comma-list behavior, workspace effects, environment effects, name-value parsing, and command syntax compatibility where applicable.
- Rich domain behavior can use analysis hooks without hard-coding every domain type into the core type lattice.
- Async behavior is metadata, not an incidental implementation detail.

## Primary Crates

- `runmat-builtins`
- `runmat-runtime`
- `runmat-hir`
- `runmat-static-analysis`
- `runmat-macros`

## Secondary Crates

- `runmat-lsp`
- `runmat/bindings/ts`
- docs/help generation surfaces
- `runmat-core`
- `runmat-mir`

## Metadata Model

Suggested language-facing metadata:

```rust
pub struct RuntimeClassMetadata {
    pub qualified_name: QualifiedName,
    pub kind: ClassKind,
    pub parent: Option<QualifiedName>,
    pub properties: Vec<PropertyMetadata>,
    pub methods: Vec<MethodMetadata>,
    pub dispatch: ClassDispatchMetadata,
    pub docs: DocsMetadata,
}

pub struct MethodMetadata {
    pub name: MethodName,
    pub is_static: bool,
    pub access: AccessLevel,
    pub params: Vec<ParameterMetadata>,
    pub outputs: Vec<OutputMetadata>,
    pub requested_output_behavior: RequestedOutputBehavior,
    pub effects: EffectSummary,
    pub workspace_effects: Vec<WorkspaceEffect>,
    pub environment_effects: Vec<EnvironmentEffect>,
    pub async_behavior: AsyncBehaviorFact,
    pub inference_hook: Option<InferenceHookId>,
    pub docs: DocsMetadata,
}

pub struct BuiltinBehaviorMetadata {
    pub id: BuiltinId,
    pub arity: BuiltinArity,
    pub requested_output_behavior: RequestedOutputBehavior,
    pub argument_parser: ArgumentParserMode,
    pub comma_list_behavior: CommaListBehavior,
    pub effects: EffectSummary,
    pub workspace_effects: Vec<WorkspaceEffect>,
    pub environment_effects: Vec<EnvironmentEffect>,
    pub async_behavior: AsyncBehaviorFact,
    pub inference_hook: Option<InferenceHookId>,
}
```

## Implementation Plan

1. Define a central language metadata representation.

This representation should be consumed by runtime registration, analysis, MIR lowering metadata, LSP, docs, and bindings. Avoid building separate per-consumer metadata formats as sources of truth.

2. Bridge existing runtime class registry.

Align the current registry concepts of class name, parent, properties, and methods with semantic class metadata.

3. Add nominal runtime class IDs.

Map Rust-defined runtime classes into the same nominal identity space used by source-defined `HirClass` items. If runtime metadata loading and source assembly allocation happen at different times, introduce stable class symbol identities that resolve to local `ClassId`s inside an assembly.

4. Add class-aware type facts.

Ensure `TypeFact` can represent class instances, handle class instances if needed, class references/metaclasses, and nominal class identity.

5. Add method and property summary lookup.

Analysis and MIR lowering should resolve instance property access, static property access, instance method calls, static method calls, constructors, overloaded operators, and overloaded indexing through metadata rather than scattered string checks.

6. Add declarative summaries.

Simple methods, properties, and builtins should be described declaratively with parameter facts, requested-output behavior, output facts, shape transfer rules where simple, comma-list behavior, effects, workspace/environment effects, async behavior, argument parser mode, and docs.

7. Add custom inference hooks.

Support hooks that receive semantic context, receiver facts, argument facts, requested-output count, value-flow facts, project/source context, and literal/value facts.

8. Migrate `DataDataset`, `DataArray`, and `DataTransaction`.

Move these toward nominal classes:

- `data.Dataset`
- `data.Array`
- `data.Transaction`

Keep compatibility shims temporarily if needed, but stop growing core `TypeFact` or old `Type` with new domain-specific variants.

9. Update LSP/docs/bindings consumers.

Consumers should derive completion items, hover text, method/property docs, TypeScript bindings, and help output from the same metadata source.

10. Add macro sugar after manual metadata works.

Do not start with proc macros. First prove the metadata model with manual declarations, then add macros to reduce boilerplate.

## Tests

Add tests for runtime class metadata registration, method lookup, property lookup, inheritance lookup, constructor/static/operator/indexing dispatch metadata, class instance type facts, class reference type facts, requested-output-sensitive method/builtin summaries, workspace/environment-effect metadata, argument parser metadata, async metadata propagation, data API inference through hooks, and LSP/docs/bindings metadata projection sanity.

## Acceptance Criteria

- Rust-defined runtime classes can expose language-facing metadata.
- Source-defined and runtime-defined nominal classes share lookup/type concepts.
- `DataDataset`, `DataArray`, and `DataTransaction` have a real nominal metadata path.
- Analysis uses class and builtin metadata plus hooks for method/property/operator/indexing/builtin summaries.
- Async behavior from runtime metadata reaches MIR/analysis summaries.
- LSP/docs/bindings consume shared metadata instead of duplicating class/builtin knowledge.

## Explicit Non-Goals

- Do not expose arbitrary Rust struct layout as RunMat language semantics.
- Do not require all builtins to migrate to metadata in one pass.
- Do not remove every legacy domain special case until the metadata-backed path is fully validated.
- Do not complete acceleration/GC lifetime hardening; that is Plan 7.
