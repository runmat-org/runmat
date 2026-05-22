# Semantic Prefix Rename Inventory

Date: 2026-05-21

## Current Footprint

- Files containing `SemanticXxx` in `crates/`: 20
- Files containing `semantic_xxx` in `crates/`: 53
- Rust occurrences:
  - `SemanticXxx`: 93
  - `semantic_xxx`: 638

Primary concentration:

- `runmat-vm/src/bytecode/compile.rs`: 130 hits
- `runmat-turbine/tests/jit.rs`: 70 hits
- `runmat-vm/src/bytecode/program.rs`: 45 hits
- `runmat-builtins/src/semantics.rs`: 42 hits
- `runmat-vm/tests/functions.rs`: 41 hits
- `runmat-runtime/src/lib.rs`: 36 hits

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

## Keep As-Is (Confirmed)

These stay unchanged because they are protocol vocabulary, not migration leftovers.

1. LSP semantic token API names in `runmat-lsp`:
   - `semantic_tokens`, `semantic_tokens_full`, `semantic_tokens_legend`
   - `SemanticToken*` types from LSP.

## Rename-Now (Next Tranche, Resting Names)

Decision: keep draining migration-era `semantic_*` where the symbol denotes bytecode/runtime plumbing rather than language semantics.

### Bytecode/runtime function identity

- `CallableIdentity::SemanticFunction` -> `CallableIdentity::BoundFunction`
- `semantic_function` field -> `bound_function`
- `semantic_functions` maps -> `bound_functions`
- `semantic_function_registry` variables/fields -> `function_registry`

### Bytecode metadata carriers

- `semantic_async_metadata` -> `async_metadata`
- `semantic_fusion_metadata` -> `fusion_metadata`
- `semantic_instruction_windows` -> `instruction_windows`
- `semantic_candidate_groups` / `semantic_candidates` -> `candidate_groups` / `candidates`
- local `semantic_index` variables with `HirIndex` type -> `hir_index`

### Tests and fixtures

- Remaining `semantic_*` test function names -> domain names (drop `semantic_` prefix).
- `spawn_semantic_lifecycle.rs` -> `spawn_function_lifecycle.rs`.
- String literals like `"RunMat:SemanticFunctionUnavailable"` should move to `"RunMat:FunctionUnavailable"` if compatibility constraints permit.

## Execution Plan (Updated)

### Completed

- Commit 1: internal/private rename sweep
- Commit 2: public type/API rename sweep
- Commit 3: tests/docs/support sweep

### Next

- Commit 4: rename callable identity + bound function field names.
- Commit 5: rename async/fusion metadata field names.
- Commit 6: rename remaining test symbols/file names and decide error identifier compatibility policy.

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
