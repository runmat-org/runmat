# Semantic Prefix Rename Inventory

Date: 2026-05-21

## Current Footprint

- Files containing `SemanticXxx` in `crates/`: 6
- Files containing `semantic_xxx` in `crates/`: 10
- Rust occurrences:
  - `SemanticXxx`: 44
  - `semantic_xxx`: 67

Primary concentration:

- `runmat-builtins/src/semantics.rs`: 42 hits
- `runmat-lsp/src/core/semantic_tokens.rs`: 21 hits
- `runmat-lsp/src/backend.rs`: 13 hits
- `runmat-lsp/src/core/analysis.rs`: 5 hits
- `runmat-runtime/src/lib.rs`: 4 hits

## Completed In This Rename Campaign

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
- `SemanticCtx` -> `LoweringCtx`
- `SemanticScope` -> `ScopeFrame`
- `semantic_resolution_name_for` -> `resolution_name_for`
- `semantic_registry()` -> `function_registry()`
- `compile_semantic_source` -> `compile_source`
- `execute_semantic_source` -> `execute_source`
- `CallableIdentity::SemanticFunction` -> `CallableIdentity::BoundFunction`
- `semantic_function` -> `bound_function`
- `semantic_functions` -> `bound_functions`
- `semantic_function_registry` -> `function_registry`
- `semantic_async_metadata` -> `async_metadata`
- `semantic_fusion_metadata` -> `fusion_metadata`
- `semantic_instruction_windows` -> `instruction_windows`
- `semantic_candidate_groups` / `semantic_candidates` -> `candidate_groups` / `candidates`
- `semantic_index` (HIR index field/local) -> `hir_index`
- `spawn_semantic_lifecycle.rs` -> `spawn_function_lifecycle.rs`
- `semantic_lowering.rs` -> `hir_lowering.rs`
- Remaining migration-era `semantic_*` test names and local helper names were de-prefixed.

## Keep As-Is (Confirmed)

These stay unchanged because they are protocol vocabulary, not migration leftovers.

1. LSP semantic token API names in `runmat-lsp`:
   - `semantic_tokens`, `semantic_tokens_full`, `semantic_tokens_legend`
   - `SemanticToken*` types from LSP.

2. Builtin semantics vocabulary:
   - `semantic_kind` and `BuiltinSemanticKind` in builtin metadata.

3. Compatibility-preserved runtime identifiers:
   - `RunMat:SemanticFunctionUnavailable`
   - `SemanticFunctionArity`

## Compatibility Policy (Final)

Public error identifiers containing `SemanticFunction` are intentionally retained for compatibility with existing downstream checks and scripts.

If we ever change them, ship aliases and a compatibility window first.

## Execution Plan (Updated)

### Completed

- Commit 1: internal/private rename sweep
- Commit 2: public type/API rename sweep
- Commit 3: tests/docs/support sweep
- Commit 4: rename callable identity + bound function field names.
- Commit 5: rename async/fusion metadata field names.
- Commit 6: rename remaining test symbols/file names and decide error identifier compatibility policy.

### Next

- Optional follow-up only if we choose a compatibility break:
  - `RunMat:SemanticFunctionUnavailable` -> `RunMat:FunctionUnavailable`
  - `SemanticFunctionArity` -> `FunctionArity`

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
