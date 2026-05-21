# Completion Audit

Date: 2026-05-20

## Objective Checklist

1. Active execution/analysis paths are semantic HIR -> MIR -> analysis -> VM/runtime.
2. No production legacy compiler/runtime path dependence.
3. MATLAB core semantics are represented by compiler/runtime products.
4. Project composition and entrypoints are manifest-driven.
5. Nominal class/builtin metadata is unified.
6. Accel/fusion planning is semantic-fact-driven.
7. Validation cadence is green.

## Prompt-to-Artifact Mapping

1. Semantic pipeline path
- Artifact: `docs-tmp/DELIVERABLE_AUDIT.md` section `### 1` (`met`).
- Evidence command: `rg -n "compile_legacy|LegacyUserFunction|runmat_vm::execute|HirProgram|\\bVarId\\b" crates`
- Latest result: no production hits under `crates/`.

2. Legacy path removal
- Artifact: `docs-tmp/DELIVERABLE_AUDIT.md` section `### 2` (`met`).
- Evidence command: same legacy grep above.
- Latest result: no production hits under `crates/`.

3. MATLAB semantics as products
- Artifact: `docs-tmp/DELIVERABLE_AUDIT.md` section `### 3` (`partial`).
- Evidence files: `docs-tmp/TARGET_MODEL.md`, `docs-tmp/NEXT_STEPS.md`, `docs-tmp/DELIVERABLE_AUDIT.md`.
- Latest result: still partial; non-builtin semantic gap inventory remains open (recent progress includes compile-stage selector-plan identifier ratchets, explicit compile identifiers for unsupported MIR operators (unary+binary)/unknown MIR builtin ids/MIR aggregate-shape invariant violations/brace-index component+context invariants/read-index context invariants/multi-assign output-count invariants/unsupported fallback-policy invariants/method-call receiver+callee invariants, MIR literal/constant/function-handle validation identifiers (`RunMat:MirNumberLiteralInvalid`, `RunMat:MirConstantUnknown`, `RunMat:MirFunctionHandleNameMissing`), explicit `MirIndexPlan::Slice`/`MirIndexPlan::Scalar` normalization guards for misplaced range/end selector operands, runtime slice-assignment RHS identifier normalization (`RunMat:ShapeMismatch`, `RunMat:InvalidSliceAssignmentRhs`), runtime slice-read result-shape identifier normalization (`RunMat:ShapeMismatch`), runtime `IndexSliceExpr` and indexing GPU-helper/linear-gather fallback failure identifier normalization (`RunMat:ShapeMismatch`, `RunMat:AccelerationOperationFailed`), VM cell-ops shape-failure identifier normalization (`RunMat:ShapeMismatch`) plus cell selector aliasing semantics ratchets (`C(2:end-1)` read stability across subsequent paren assignment and `B = C; C{2} = ...` copy-on-write preservation), and `IndexSliceExpr`/`StoreSliceExpr` plus non-expr `IndexSlice`/`StoreSlice` dispatch identifier-preservation ratchets with direct unit contracts; HIR+core isolated-capture, class-self-inheritance, and class-member duplication/conflict identifier contracts with early aggregate-shape semantic invariant checks are also in place, but residual aggregate/selector design gaps remain).
- Incremental update: VM compile selector-plan gating now rejects only end-dependent range operands for `MirIndexPlan::Slice` and allows non-end-dependent range-valued locals on non-expr slice lowering, preserving mixed logical-mask + range semantics (`mixed_logical_mask_and_range_across_3d`) while keeping `RunMat:MirSliceIndexPlanInvalid` for true end-relative selector expressions.
- Incremental update: VM compile boundaries now enforce explicit delete-assignment invariants with stable identifiers: non-empty delete RHS rejection (`RunMat:MirDeleteAssignmentRhsInvalid`, `primary_compile_rejects_nonempty_delete_rhs_with_identifier`), delete mutation/assign target mismatch rejection (`RunMat:MirDeleteAssignmentPlaceMismatch`, `primary_compile_rejects_delete_place_mismatch_with_identifier`), non-index delete-target rejection (`RunMat:MirDeleteAssignmentTargetInvalid`, `primary_compile_rejects_delete_on_nonindexed_target_with_identifier`), non-paren delete-index rejection (`RunMat:MirDeleteAssignmentIndexKindInvalid`, `primary_compile_rejects_delete_on_brace_index_target_with_identifier`), delete-context mismatch rejection (`RunMat:MirDeleteAssignmentContextInvalid`, `primary_compile_rejects_delete_with_nondeletion_index_context_with_identifier`), non-delete deletion-context rejection (`RunMat:MirDeletionContextWithoutDeleteInvalid`, `primary_compile_rejects_deletion_context_without_delete_with_identifier`), and delete-creation-policy rejection (`RunMat:MirDeleteAssignmentCreationPolicyInvalid`, `primary_compile_rejects_delete_with_nonexisting_creation_policy_with_identifier`), keeping delete intent compiler-product driven instead of runtime-inferred or silently dropped.
- Incremental update: indexed member store-back semantics now have explicit MIR-shape ratchet coverage (`indexed_member_assignment_lowers_to_index_place_over_member_base`) pinning `s.a(2)=...` to `MirPlace::Index(MirPlace::Member(...), ...)` with `IndexedAssign` mutation policy, and semantic products no longer carry `IndexKind::Dot`.
- Incremental update: semantic runtime deletion behavior now has direct contract coverage for both positive and rejection paths: cell paren deletion executes via semantic store-back (`cell_paren_delete_executes_with_semantic_store_back`), matrix linear deletion on non-vectors reports `RunMat:UnsupportedDeletion`, and string slice deletion reports `RunMat:UnsupportedSliceDeletion`.
- Incremental update: source-level brace assignment with colon selectors now has explicit compile-path identifier contracts at both VM and core boundaries (`primary_compile_rejects_cell_assignment_colon_selector_from_source_with_identifier`, `compile_input_reports_cell_assignment_colon_selector_identifier`), pinning the semantic compiler rejection to `RunMat:MirCellIndexPlanInvalid`.
- Incremental update: VM source-level `isfield` string-array coverage replaced placeholder semantics with a real matrix contract (`struct_isfield_string_array_names`), asserting `names = ["a" "b"; "x" "a"]` yields a `LogicalArray` with `shape [2,2]` and column-major data `[1,0,0,1]`.
- Incremental update: runtime callable-descriptor semantic name-resolution policy now matches VM fallback-name policy categories (`DynamicName`/`Imported`/`Method` under runtime-name policy; well-formed `ExternalName` under external-boundary policy), with runtime contract coverage asserting semantic resolver dispatch for `Method` and `Imported` identities (`method_identity_runtime_name_resolution_policy_uses_semantic_resolver`, `imported_identity_runtime_name_resolution_policy_uses_semantic_resolver`).
- Incremental update: runtime semantic callback request ABI removed dead `SemanticCallableKind`/`kind` placeholder fields from `SemanticCallableRequest`, keeping only active policy inputs (identity/fallback/args/requested outputs), with updated VM/runtime callsites and preserved external-boundary resolver contract coverage (`external_name_descriptor_external_boundary_can_use_semantic_resolver` now pins well-formed qualified external identity behavior).
- Incremental update: VM object/member dispatch callable identities are now typed as `CallableIdentity::Method` instead of `DynamicName` across shared object-call entrypoints and closure/member-dispatch internals, with classref external-resolution contracts preserved under `Method` identity (`classref_external_method_uses_external_boundary_semantic_resolution`, `classref_external_method_without_resolver_remains_unresolved`).
- Incremental update: VM semantic expanded-call instruction-shape coverage now includes explicit multi-output semantic-function lowering contracts (`semantic_expand_multi_output_uses_typed_instruction`), pinning `pair(C{:})` to `CallSemanticFunctionExpandMultiOutput` with `out_count == 2` and preserving execution semantics (`s = 15`).
- Incremental update: unresolved dynamic-call instruction-shape coverage now includes explicit multi-output fixed and expanded variants (`unresolved_function_multi_output_uses_typed_instruction_and_errors`, `unresolved_function_expand_multi_output_uses_typed_instruction_and_errors`), pinning `out_count == 2` lowering contracts and preserving stable runtime failure identifier `RunMat:UndefinedFunction`.
- Incremental update: VM descriptor contracts now directly ratchet `CallableIdentity::Method` semantic-resolution behavior (`method_identity_runtime_name_resolution_can_use_semantic_resolver`, `method_identity_runtime_name_resolution_without_resolver_errors`), preserving semantic resolver dispatch when present and stable undefined-function behavior when absent.
- Incremental update: VM descriptor contracts now also ratchet `CallableIdentity::Imported` semantic-resolution behavior (`imported_identity_runtime_name_resolution_can_use_semantic_resolver`), pinning imported-identity dispatch to semantic resolver/invoker when available while keeping unresolved imported identities off builtin-name fallback (`imported_identity_never_falls_back_to_builtin_name_resolution`).
- Incremental update: closure `feval` descriptor construction now ratchets embedded semantic identity precedence (`feval_closure_with_embedded_semantic_prefers_embedded_identity`), pinning `Closure.semantic_function` dispatch to semantic function id execution even when registry name mappings exist for the same closure name.
- Incremental update: qualified function-handle descriptor execution now has explicit external-boundary semantic-resolver coverage (`feval_function_handle_external_boundary_can_use_semantic_resolver`), extending `pkg.remote_inc` handle behavior from classification-only contracts to execution-path semantic identity dispatch.
- Incremental update: `@`-qualified handle descriptor execution now has explicit external-boundary semantic-resolver coverage (`feval_at_handle_external_boundary_can_use_semantic_resolver`), extending `feval("@pkg.remote_inc", ...)` behavior from classification-only contracts to execution-path semantic identity dispatch.
- Incremental update: remaining `feval` handle-variant descriptor branches now have explicit execution ratchets: `Value::ExternalFunctionHandle` semantic resolver dispatch (`feval_external_function_handle_can_use_semantic_resolver`) and `Value::SemanticFunctionHandle` embedded-id precedence over resolver mapping (`feval_semantic_function_handle_prefers_embedded_function_id`).
- Incremental update: semantic `feval` multi-output lowering now has explicit instruction-shape ratchets for both fixed and expanded-argument forms (`semantic_feval_multi_assign_uses_typed_instruction`, `semantic_feval_expand_multi_assign_uses_typed_instruction`), pinning `CallFevalMulti`/`CallFevalExpandMultiOutput` contracts at `out_count == 2` and preserving expanded-path execution semantics.
- Incremental update: unresolved qualified external-handle `feval` multi-output paths now have explicit typed-opcode + error-surface ratchets (`unresolved_qualified_external_handle_multi_output_feval_uses_typed_instruction`, `unresolved_qualified_external_handle_expand_multi_output_feval_uses_typed_instruction`), preserving external-handle classification and stable `RunMat:UndefinedFunction` failures.
- Incremental update: callable-descriptor `feval` closure handling now has direct registry-resolution coverage (`feval_closure_without_embedded_semantic_uses_registry_name_resolution`), pinning the non-embedded-identity closure path to semantic registry lookup before runtime fallback.
- Incremental update: brace-content cell assignment now supports MATLAB-style subscript growth with empty fillers (`0x0 double`) and `end+1` subscript growth in store context, ratcheted by `primary_compile_supports_cell_brace_subscript_growth_with_empty_fillers` and `primary_compile_supports_cell_brace_end_plus_one_subscript_growth` while preserving matrix linear `end+1` rejection (`RunMat:UnsupportedCellGrowth`).
- Incremental update: linear brace cell-growth semantics now support non-contiguous vector expansion (`C{5}=...`, `C{end+3}=...`) with empty fillers, and empty-shape linear growth now normalizes `5x0`/`0x5` inputs to `1xN` row-vector expansion, ratcheted by `primary_compile_supports_cell_brace_linear_gap_growth_for_vectors`, `primary_compile_supports_cell_brace_linear_end_plus_k_growth_for_vectors`, and `primary_compile_linear_cell_growth_from_5_by_0_normalizes_to_row_vector`.
- Incremental update: empty-shape row-normalization coverage now explicitly ratchets both orientations (`5x0` and `0x5`) via `primary_compile_linear_cell_growth_from_0_by_5_normalizes_to_row_vector` alongside the existing `5x0` contract.
- Incremental update: selector numeric-integrality semantics now reject fractional numeric indices instead of truncating them in runtime selector materialization and scalar `end` expression index resolution: selector/expr-plan ratchets (`selector_from_value_dim_rejects_fractional_numeric_indices`, `linear_indices_reject_fractional_tensor_indices`) plus source-level scalar/cell end-expression ratchets (`scalar_end_div_indexing_rejects_fractional_result`, `primary_compile_rejects_fractional_cell_end_expression_read_index`) now pin `RunMat:UnsupportedIndexType` for non-integer numeric selector results.

4. Manifest-driven composition and entrypoints
- Artifact: `docs-tmp/DELIVERABLE_AUDIT.md` section `### 4` (`met`).
- Evidence command: `rg -n "runmat.toml|entrypoint|manifest|sources|dependencies" crates/runmat-config crates/runmat-core crates/runmat-cli`
- Latest result: resolver/discovery wiring present across config/core/CLI.

5. Unified nominal class/builtin metadata
- Artifact: `docs-tmp/DELIVERABLE_AUDIT.md` section `### 5` (`met`).
- Evidence command: `rg -n "CallableIdentity|CallableFallbackPolicy|ClassMetadata|nominal" crates/runmat-hir crates/runmat-builtins crates/runmat-vm crates/runmat-runtime`
- Latest result: shared callable identity + fallback policy surfaces are present and used cross-layer.

6. Semantic-fact-driven accel/fusion
- Artifact: `docs-tmp/DELIVERABLE_AUDIT.md` section `### 6` (`met` for current scope).
- Evidence command: `rg -n "AnalysisStore|fusion|FusionPlan|Accel" crates/runmat-mir crates/runmat-vm crates/runmat-core`
- Latest result: semantic analysis/fusion metadata + runtime-owned fusion graph path are wired and covered.
- Latest closure tests:
  - `cargo test -p runmat-vm --test spawn_semantic_lifecycle -- --nocapture`
  - `cargo test -p runmat-core --test fusion_regressions -- --nocapture`
  - `cargo test -p runmat-vm runtime_accel_graph_ignores_stale_compile_graph_metadata -- --nocapture`
  - `cargo test -p runmat-vm --lib runtime_state_ignores_stale_compile_graph_metadata -- --nocapture`
  - `cargo test -p runmat-vm --test fusion_gpu fusion_graph_helper_ignores_stale_compile_graph_metadata -- --nocapture`

7. Validation cadence
- Required gates:
  - `cargo fmt --all --check`
  - `cargo test -p runmat-core --test semicolon_suppression -- --nocapture`
  - `cargo check --workspace`
  - `git diff --check`
- Latest result: all green on 2026-05-20.

## Missing / Incomplete Requirements

- Objective item 3 remains incomplete (`partial`): `docs-tmp/DELIVERABLE_AUDIT.md` and `docs-tmp/NEXT_STEPS.md` still track open non-builtin semantic-product gaps.

## Completion Decision

Objective is **not achieved** yet.
