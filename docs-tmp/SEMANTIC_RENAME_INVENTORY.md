# Semantic Prefix Rename Inventory

Date: 2026-05-21

## Current Footprint

- Files containing `SemanticXxx` in `crates/`: 102
- Files containing `semantic_xxx` in `crates/`: 59
- Non-test Rust occurrences:
  - `SemanticXxx`: 627
  - `semantic_xxx`: 532

Primary concentration:

- `runmat-vm`: 230 (`SemanticXxx`) / 313 (`semantic_xxx`)
- `runmat-runtime`: 159 / 72
- `runmat-hir`: 90 / 61
- `runmat-mir`: 51 / 13
- `runmat-lsp`: 41 / 14
- `runmat-turbine`: 38 / 106
- `runmat-core`: 36 / 85

## Public API Surface Using `Semantic*`

- `runmat-hir`
  - `SemanticError` ([error.rs](./../crates/runmat-hir/src/error.rs))
  - `SemanticIndex` ([hir.rs](./../crates/runmat-hir/src/hir.rs))
  - `semantic_resolution_name_for(...)` ([hir.rs](./../crates/runmat-hir/src/hir.rs))
- `runmat-runtime`
  - `SemanticCallableRequest`
  - `SemanticFunctionInvoker`
  - `SemanticFunctionResolver`
  - guard types for resolver/invoker
  - ([user_functions.rs](./../crates/runmat-runtime/src/user_functions.rs))
- `runmat-vm`
  - `SemanticFunctionBytecode`
  - `SemanticFunctionRegistry`
  - `SemanticAsyncMetadata`
  - `SemanticAsyncRuntimeModel`
  - `SemanticSpawnSite`
  - `SemanticAwaitSite`
  - `SemanticFusionMetadata`
  - `SemanticFusionCandidateGroup`
  - `SemanticFusionInstructionKind`
  - `SemanticFusionInstructionWindow`
  - ([program.rs](./../crates/runmat-vm/src/bytecode/program.rs))
- `runmat-lsp`
  - `SemanticModel`
  - semantic token exports (`semantic_tokens*`)

## High-Volume `Semantic*` Symbols

- `SemanticError`: 131
- `SemanticFunctionRegistry`: 60
- `SemanticFusionInstructionKind`: 57
- `SemanticFunctionBytecode`: 44
- `SemanticFusionInstructionWindow`: 40
- `SemanticFusionCandidateGroup`: 32
- `SemanticCallableRequest`: 28

## Keep As-Is (Confirmed)

These stay unchanged because they are protocol vocabulary, not migration leftovers.

1. LSP semantic token API names in `runmat-lsp`:
   - `semantic_tokens`, `semantic_tokens_full`, `semantic_tokens_legend`
   - `SemanticToken*` types from LSP.

## Rename-Now (Final Resting Names)

Decision: previous “rename later” items are promoted to **rename now**. The target is long-lived compiler/runtime names with no migration-era `Semantic` prefix.

### Core compiler/runtime types

- `SemanticError` -> `HirError`
- `SemanticIndex` -> `HirIndex`
- `SemanticCallableRequest` -> `CallableRequest`
- `SemanticFunctionInvoker` -> `FunctionInvoker`
- `SemanticFunctionResolver` -> `FunctionResolver`
- `SemanticFunctionInvokerGuard` -> `FunctionInvokerGuard`
- `SemanticFunctionResolverGuard` -> `FunctionResolverGuard`
- `SemanticFunctionBytecode` -> `FunctionBytecode`
- `SemanticFunctionRegistry` -> `FunctionRegistry`
- `SemanticAsyncMetadata` -> `AsyncMetadata`
- `SemanticAsyncRuntimeModel` -> `AsyncRuntimeModel`
- `SemanticSpawnSite` -> `SpawnSite`
- `SemanticAwaitSite` -> `AwaitSite`
- `SemanticFusionMetadata` -> `FusionMetadata`
- `SemanticFusionCandidateGroup` -> `FusionCandidateGroup`
- `SemanticFusionInstructionKind` -> `FusionInstructionKind`
- `SemanticFusionInstructionWindow` -> `FusionInstructionWindow`

### Internal/private helpers

- `SemanticCtx` -> `LoweringCtx`
- `SemanticScope` -> `ScopeFrame`
- `semantic_resolution_name_for` -> `resolution_name_for`
- `semantic_registry()` -> `function_registry()`
- Private helper prefixes:
  - `semantic_diagnostic` -> `hir_diagnostic` (or equivalent domain-specific naming)
  - `semantic_output_value` -> `output_value`
  - `semantic_callback_literal` -> `callback_literal`
  - `semantic_display_context` -> `display_context`
  - `semantic_expr_emit_disposition` -> `expr_emit_disposition`

### Test/support helpers

- `compile_semantic_source` -> `compile_source`
- `execute_semantic_source` -> `execute_source`

## Execution Plan (Updated)

### Commit 1: internal/private rename sweep

- Rename `SemanticCtx`, `SemanticScope`, and private `semantic_*` helpers.
- No public API changes in this commit.

### Commit 2: public type/API rename sweep

- Apply all type and public function renames above across crates.
- Update re-exports and call sites in dependent crates.

### Commit 3: tests/docs/support sweep

- Rename test helpers (`compile_semantic_source`, `execute_semantic_source`).
- Update tests, docs-tmp audits, and references.

Validation gates after each commit:

- `cargo fmt --all --check`
- `cargo test -p runmat-core --test semicolon_suppression -- --nocapture`
- `cargo check --workspace`
- `git diff --check`

## Naming Principle (Locked)

Use domain-stable names that age well:

1. Module/context communicates semantic provenance.
2. Type names communicate role/product, not migration path.
3. Keep standards/protocol terminology unchanged (LSP semantic tokens).
