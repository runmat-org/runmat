# Progress

## Current Focus

Broad consumer migration and compatibility-surface cleanup, while keeping semantic pipeline validation green.

- (pending commit) Set builtins option-parse identifier ratchet
  - Added stable option-parse identifiers for set builtins:
    - `RunMat:unique:ConflictingOrderOptions`
    - `RunMat:unique:ConflictingOccurrenceOptions`
    - `RunMat:unique:UnknownOption`
    - `RunMat:union:ConflictingOrderOptions`
    - `RunMat:union:UnknownOption`
    - `RunMat:intersect:ConflictingOrderOptions`
    - `RunMat:intersect:UnknownOption`
    - `RunMat:setdiff:ConflictingOrderOptions`
    - `RunMat:setdiff:UnknownOption`
    - `RunMat:ismember:UnknownOption`
  - Tightened set-builtin option tests to assert `RuntimeError.identifier()` instead of message substring checks for:
    - unknown-option rejection (`unique`/`union`/`intersect`/`setdiff`/`ismember`)
    - conflicting order-option rejection (`unique`/`union`/`intersect`/`setdiff`)
    - conflicting occurrence-option rejection (`unique`)
  - Validation:
    - `cargo test -p runmat-runtime rejects_unknown_option -- --nocapture`
    - `cargo test -p runmat-runtime conflicting_order -- --nocapture`
    - `cargo test -p runmat-runtime conflicting_occurrence -- --nocapture`
    - `cargo test -p runmat-runtime legacy_option -- --nocapture`
    - `cargo fmt --all`

- (pending commit) Set builtins legacy-option identifier ratchet
  - Replaced message-only legacy-option rejections in set builtins with stable identifiers:
    - `RunMat:unique:LegacyOptionUnsupported`
    - `RunMat:union:LegacyOptionUnsupported`
    - `RunMat:intersect:LegacyOptionUnsupported`
    - `RunMat:setdiff:LegacyOptionUnsupported`
    - `RunMat:ismember:LegacyOptionUnsupported`
  - Updated runtime unit coverage in:
    - [unique.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/sorting_sets/unique.rs) (`unique_rejects_legacy_option`)
    - [union.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/sorting_sets/union.rs) (`union_rejects_legacy_option`)
    - [intersect.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/sorting_sets/intersect.rs) (`intersect_rejects_legacy_option`)
    - [setdiff.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/sorting_sets/setdiff.rs) (`setdiff_rejects_legacy_option`)
    - [ismember.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/sorting_sets/ismember.rs) (`options_reject_legacy`)
  - Validation:
    - `cargo test -p runmat-runtime legacy_option -- --nocapture`
    - `cargo test -p runmat-runtime options_reject_legacy -- --nocapture`
    - `cargo fmt --all`

## Latest Committed Slices (2026-05-19)

- `d6489fbc` + follow-up (working tree): Plan 3/4 async direct-call lazy future-descriptor lane
  - `MirRvalue::Future` now lowers to lazy semantic future descriptors (`CreateSemanticFuture*`) in [core.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/compiler/core.rs) and [instr.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/instr.rs), rather than immediate semantic function calls.
  - VM dispatch now materializes those descriptors at async boundaries in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs): `spawn` eagerly resolves a future descriptor before wrapping task payload, while `await` resolves deferred descriptors and preserves existing spawned-task handle validation.
  - Async runtime metadata now reflects this behavior as `LazyFutureDescriptorLane` in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs) and compile metadata assertions in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - Core interaction ratchet in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs):
    - `async_call_without_await_or_spawn_stays_lazy_until_await`
  - Contract: direct async calls are now lazy until `await`/`spawn`; `spawn` eager interaction behavior remains explicit and green.
  - Validation:
    - `cargo test -p runmat-core --test async_stdin -- --nocapture`
    - `cargo test -p runmat-vm --test spawn_semantic_lifecycle -- --nocapture`
    - `cargo check --workspace`

- (pending commit) Plan 3/4 deferred-future spawn boundary ratchet
  - Added core async interaction coverage in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs):
    - `deferred_future_triggers_interaction_when_spawned`
  - Contract: semantic future descriptors remain lazy at creation (`fut = asks()` has zero stdin events), and interaction is realized when the deferred value crosses a spawn boundary (`spawn(fut)`).
  - Validation:
    - `cargo test -p runmat-core --test async_stdin deferred_future_triggers_interaction_when_spawned -- --nocapture`
    - `cargo test -p runmat-core --test async_stdin -- --nocapture`
    - `cargo test -p runmat-vm --test spawn_semantic_lifecycle -- --nocapture`
    - `cargo test -p runmat-core --test semicolon_suppression -- --nocapture`
    - `cargo fmt --all --check`
    - `cargo check --workspace`
    - `git diff --check`

- (pending commit) Plan 3/4 async expansion-call future-descriptor lowering ratchet
  - Added VM compile/interpret coverage in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs):
    - `primary_compile_lowers_async_expansion_call_to_future_expand_instruction`
  - Contract: async direct calls with comma-list expansion (`args{:}`) lower to lazy future-descriptor instruction form (`CreateSemanticFutureExpandMultiOutput`) rather than eager semantic call execution.
  - Validation:
    - `cargo test -p runmat-vm primary_compile_lowers_async_expansion_call_to_future_expand_instruction -- --nocapture`

- (pending commit) Plan 7 fusion planner metadata now records runtime graph source
  - Fusion planner metadata in [types.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/types.rs) now carries `accel_graph_source` alongside `accel_graph_state`.
  - Core fusion snapshot producers in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs) and [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs) now populate that field from VM runtime graph selection.
  - VM bytecode runtime graph API in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs) now exposes explicit graph source classification via `runtime_accel_graph_for_fusion_with_source`.
  - Core fusion regressions in [fusion_regressions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/fusion_regressions.rs) now assert `accel_graph_source == "runtime_materialized_from_instructions"` for compile/runtime snapshots, tightening evidence that active fusion entrypoints are runtime-graph-driven.
  - Validation:
    - `cargo test -p runmat-vm runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing -- --nocapture`
    - `cargo test -p runmat-vm runtime_accel_graph_ignores_stale_compile_graph_metadata -- --nocapture`
    - `cargo test -p runmat-core --test fusion_regressions -- --nocapture`
    - `cargo fmt --all --check`
    - `cargo check --workspace`
    - `git diff --check`

- (pending commit) Plan 7 runtime-state stale compile-graph exclusion ratchet
  - Added VM interpreter-state regression in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs):
    - `runtime_state_ignores_stale_compile_graph_metadata`
  - Contract: the active runtime fusion execution state (`InterpreterState::new`) must retain a graph materialized from current bytecode instructions, not stale compile graph metadata.
  - Validation:
    - `cargo test -p runmat-vm runtime_state_ignores_stale_compile_graph_metadata -- --nocapture`
    - `cargo test -p runmat-core --test fusion_regressions -- --nocapture`
    - `cargo test -p runmat-vm --test spawn_semantic_lifecycle -- --nocapture`
    - `cargo fmt --all --check`
    - `cargo check --workspace`
    - `git diff --check`

- (pending commit) Plan 6/7 slice-assignment unsupported-base identifier ratchet
  - VM interpreter slice-assignment dispatch in [indexing.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/indexing.rs) now emits stable `RunMat:SliceNonTensor` identifiers for unsupported base types in both `StoreSlice` and `StoreSliceExpr` paths instead of message-only runtime errors.
  - Added VM semantic regression coverage in [basics.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/basics.rs):
    - `string_slice_assignment_on_scalar_string_reports_slice_non_tensor`
  - Validation:
    - `cargo test -p runmat-vm string_slice_assignment_on_scalar_string_reports_slice_non_tensor -- --nocapture`

- (pending commit) Plan 7 runtime fusion graph materialization no longer reuses compile graph metadata
  - Runtime fusion graph selection in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs) now always materializes from active bytecode instructions (`build_accel_graph`) when semantic runtime groups are present, instead of preferring `bytecode.accel_graph` when that compile-time artifact exists.
  - Added regression coverage:
    - `runtime_accel_graph_ignores_stale_compile_graph_metadata`
  - Contract: runtime fusion planning uses runtime-owned graph materialization, and stale compile graph artifacts cannot override active instruction-derived graph shape.
  - Validation:
    - `cargo test -p runmat-vm runtime_accel_graph_ignores_stale_compile_graph_metadata -- --nocapture`

- (pending commit) Plan 7 compile fusion ratchet for multi-window node-assignment boundary
  - Added VM compile regression coverage in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs):
    - `primary_compile_keeps_multi_window_groups_node_empty_before_runtime_reconciliation`
  - Contract: compile-time semantic fusion groups remain node-empty even across multiple semantic instruction windows split by non-accelerable operations; accel node reconciliation is runtime-owned (`prepare_fusion_plan`), not compile-owned.
  - Validation:
    - `cargo test -p runmat-vm primary_compile_keeps_multi_window_groups_node_empty_before_runtime_reconciliation -- --nocapture`

- (pending commit) Plan 7 core error-namespace ratchet now requires identifier prefix
  - Tightened namespace compatibility coverage in [error_namespace_compat.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/error_namespace_compat.rs) to require namespace prefixing on the error identifier itself, removing the transitional message-fragment fallback (`identifier_ok || message_ok`).
  - This enforces the proper contract for namespace migration work: identifier behavior is authoritative, while display message text is non-authoritative.
  - Validation:
    - `cargo test -p runmat-core --test error_namespace_compat -- --nocapture`

- (pending commit) Plan 7 data-runtime identifier ratchet for manifest/transaction conflicts
  - Data runtime error paths in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/data/mod.rs) now emit stable identifiers for two high-value control paths instead of relying on message-fragment assertions:
    - `RunMat:data:ManifestConflict` for manifest sequence precondition conflicts.
    - `RunMat:data:TransactionNotFound` for missing transaction registry lookups.
  - Data runtime unit coverage now asserts these identifiers directly in:
    - `ensure_manifest_sequence_rejects_conflict`
    - `transaction_registry_roundtrip`
  - Validation:
    - `cargo test -p runmat-runtime ensure_manifest_sequence_rejects_conflict -- --nocapture`
    - `cargo test -p runmat-runtime transaction_registry_roundtrip -- --nocapture`

- (pending commit) Plan 7 runtime scalar-fusion bypass for semantic literal/scalar paths
  - VM fusion dispatch in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/fusion.rs) now bypasses elementwise fusion execution when all runtime inputs are scalar-shaped values, forcing these groups down the normal VM/runtime scalar path.
  - Accelerate elementwise plan support in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs) now rejects scalar-shaped elementwise groups up front (`scalar_shape_known_one`), preventing scalar-only GPU kernel materialization at plan time.
  - This closes the open runtime failure where scalar literal-negation fusion produced `GpuTensor([1,1])` values that leaked into matrix literal construction and triggered `cannot convert GpuTensor ... to f64`.
  - Validation:
    - `cargo test -p runmat-vm --test fusion_gpu fused_safe_followup_builtins_remain_resident -- --nocapture`
    - `cargo test -p runmat-vm --test fusion_gpu direct_execution_of_safe_followup_group_returns_gpu_tensor -- --nocapture`
    - `cargo test -p runmat-vm --test fusion_gpu explained_variance_matches_cpu -- --nocapture`

- (pending commit) Plan 7 formatter unsupported-specifier identifier ratchet
  - Shared runtime formatter in [format.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/common/format.rs) now emits stable identifier `RunMat:format:UnsupportedSpecifier` when encountering unsupported `%` conversions (for example `%q`), instead of message-only runtime errors.
  - Core session integration coverage in [printf_semantics.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/printf_semantics.rs) now asserts that stable identifier at the session boundary for `fprintf` format failures in both `SessionExecutionResult.error` and `RunError::Runtime` paths, replacing display-text substring checks.
  - Added runtime unit ratchet in [sprintf.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/strings/core/sprintf.rs):
    - `sprintf_unsupported_specifier_reports_stable_identifier`
  - Validation: `cargo test -p runmat-runtime sprintf_unsupported_specifier_reports_stable_identifier -- --nocapture`, `cargo test -p runmat-core --test printf_semantics fprintf_format_error_propagates_to_session_boundary -- --nocapture`, `cargo fmt --all --check`.

- (pending commit) Plan 7 runtime-first fusion graph access migration in `fusion_gpu` coverage
  - Added shared runtime-first graph helper in [fusion_gpu.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/fusion_gpu.rs): `graph_for_fusion_test`.
  - Updated touched fusion tests to consume runtime-owned graph selection first (with compile-graph fallback only inside the helper), replacing direct `bytecode.accel_graph` call-site reads.
  - Tightened runtime graph API behavior in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs): when semantic runtime groups exist, `runtime_accel_graph_for_fusion` now returns compile graph metadata if present, otherwise materializes from active bytecode.
  - Validation:
    - `cargo test -p runmat-vm --test fusion_gpu direct_execution_of_safe_followup_group_returns_gpu_tensor -- --nocapture`
    - `cargo test -p runmat-vm --test fusion_gpu explained_variance_matches_cpu -- --nocapture`
    - `cargo test -p runmat-vm --test matrix_division -- --nocapture`
    - `cargo test -p runmat-vm primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes -- --nocapture`
    - `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group -- --nocapture`
    - `cargo test -p runmat-core --test fusion_regressions -- --nocapture`
  - Follow-up status:
    - `fused_safe_followup_builtins_remain_resident` regression is now green on branch after scalar-only fusion bypass and scalar-shaped plan-support guards.

- (pending commit) Plan 7 test-surface decoupling from compile accel-graph artifacts
  - VM compile-unit fusion tests in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now assert runtime graph materialization/reconciliation (`runtime_accel_graph_for_fusion`, `prepare_fusion_plan`) instead of asserting compile-populated `bytecode.accel_graph` presence.
  - Matrix-division semantic coverage in [matrix_division.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/matrix_division.rs) now asserts observable division behavior contracts (operator parity vs `mrdivide`/`mldivide` and scalar `/` vs `./`) rather than accel-graph node-shape internals.
  - Validation: `cargo test -p runmat-vm --test matrix_division -- --nocapture`, `cargo test -p runmat-vm primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes -- --nocapture`, `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group -- --nocapture`.

- (pending commit) Plan 7 removed compile-graph fallback from runtime fusion planning/snapshots
  - Core fusion snapshot paths now use only runtime-owned graph materialization (`runtime_accel_graph_for_fusion`) in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs) and [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), instead of falling back to `bytecode.accel_graph`.
  - VM interpreter fusion-plan setup in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) now similarly uses only runtime-owned graph selection for plan preparation/annotation.
  - This removes remaining production fallback reads of compile-provided accel graph metadata in active fusion planning surfaces.
  - Validation: `cargo test -p runmat-core --test fusion_regressions -- --nocapture`, `cargo test -p runmat-vm runtime_materialized_graph_is_retained_for_fusion_execution -- --nocapture`, `rg -n "or\\(.*bytecode\\.accel_graph|bytecode\\.accel_graph\\.as_ref\\(|prepared\\.bytecode\\.accel_graph\\.as_ref\\(" crates/runmat-core/src/session crates/runmat-vm/src/interpreter/state.rs` (no matches).

- (pending commit) Plan 7 runtime-materialized accel graph now reaches fused execution
  - VM interpreter state now retains the exact accel graph used for runtime fusion-plan preparation in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) (`fusion_accel_graph`), including on-demand runtime graph materialization when compile graph artifacts are missing.
  - Fused execution now uses that retained runtime graph in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) instead of hard-wiring execution to `bytecode.accel_graph`, closing a runtime path where prepared plans could exist but execution skipped fusion due to a missing compile graph.
  - Runtime graph materialization in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs) now runs whenever semantic runtime groups exist (even if compile graph metadata is present), so runtime execution/snapshot planning consistently uses a graph derived from active bytecode instructions rather than relying on compile-time graph presence.
  - Added state-level regression coverage:
    - `runtime_materialized_graph_is_retained_for_fusion_execution`
  - Added bytecode-level regression coverage:
    - `runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_present`
  - Validation: `cargo test -p runmat-vm runtime_materialized_graph_is_retained_for_fusion_execution -- --nocapture`, `cargo test -p runmat-vm runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_ -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions -- --nocapture`.

- (pending commit) Plan 3/7 VM static-property missing-name identifier ratchet
  - Tightened VM semantic function coverage in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs):
    - `unqualified_static_property_without_imports_errors` now asserts stable identifier `RunMat:UndefinedVariable` instead of message-fragment matching (`undefined`/`not found` text).
  - Validation: `cargo test -p runmat-vm unqualified_static_property_without_imports_errors -- --nocapture`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 6/7 aggregate-edge cell member RHS shape identifier ratchet
  - Tightened cell aggregate member-assignment error contracts in [cells.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/ops/cells.rs):
    - `assign_cell_member` cell-RHS shape mismatch now emits stable identifier `RunMat:CellMemberRhsShapeMismatch` instead of message-only string error conversion.
  - Added direct unit coverage:
    - `assign_cell_member_rejects_shape_mismatch_cell_rhs`
  - Validation: `cargo test -p runmat-vm assign_cell_member_rejects_shape_mismatch_cell_rhs -- --nocapture`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 6/7 object selector-plan identifier normalization ratchet
  - Tightened object paren-expr selector validation/runtime contracts in [shared.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/call/shared.rs):
    - unsupported object numeric selector types now emit stable identifier `RunMat:ObjectSelectorTypeUnsupported` (via `mex("ObjectSelectorTypeUnsupported", ...)`) instead of ad hoc string errors.
    - added explicit out-of-bounds range-dimension coverage (`RunMat:InvalidRangeSelectorDim`).
  - Added/updated selector-plan tests:
    - `object_paren_expr_selector_values_reject_out_of_bounds_range_dim`
    - `object_paren_expr_selector_values_reject_unsupported_numeric_selector_type`
  - Validation: `cargo test -p runmat-vm object_paren_expr_selector_values_reject_out_of_bounds_range_dim -- --nocapture`, `cargo test -p runmat-vm object_paren_expr_selector_values_reject_unsupported_numeric_selector_type -- --nocapture`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 runtime fusion sanitization no-longer-trusts stale compile node IDs
  - Strengthened runtime fusion-plan sanitization in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs):
    - pre-mapped compile node IDs are now retained only when they are both kind-compatible and span-overlap/touch compatible with the semantic group span.
    - stale mapped IDs now drop into existing runtime nearby-node recovery path instead of being trusted solely by node kind.
  - Added regression coverage:
    - `prepare_fusion_plan_replaces_stale_mapped_nodes_using_runtime_span_recovery`
  - Validation: `cargo test -p runmat-accelerate prepare_fusion_plan_replaces_stale_mapped_nodes_using_runtime_span_recovery -- --nocapture`, `cargo test -p runmat-accelerate prepare_fusion_plan_recovers_empty_group_nodes_from_runtime_graph -- --nocapture`, `cargo test -p runmat-accelerate prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_graph_is_too_far -- --nocapture`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 preserve semantic fusion windows under partial accel-node mapping drop
  - `runmat-vm` compile fusion-group derivation in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now preserves all semantic instruction windows as executable-group scaffolding, even when only a subset map to accel nodes.
  - Added compile helper path:
    - `derive_semantic_fusion_groups_preserving_unmapped_windows`
  - Mapped windows still retain accel node lists; unmapped windows now remain as empty-node groups (instead of being dropped unless all windows failed).
  - Added regression coverage:
    - `semantic_windows_preserve_unmapped_windows_alongside_mapped_groups`
  - Validation: `cargo test -p runmat-vm semantic_windows_preserve_unmapped_windows_alongside_mapped_groups -- --nocapture`, `cargo test -p runmat-vm semantic_windows_ -- --nocapture`, `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops -- --nocapture`, `cargo test -p runmat-core --test async_stdin`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 lexer error-token boundary ratchet for formatter compatibility
  - Tightened token-format compatibility coverage in [repl.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/repl.rs):
    - `unterminated_string_is_error_token` now asserts the first emitted token is `Error` (`split_whitespace().next() == Some("Error")`) instead of broad substring matching.
  - Validation: `cargo test -p runmat-core --test repl`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 workspace replace-import stale-binding contract ratchet
  - Tightened stale-variable verification in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs):
    - `workspace_state_roundtrip_replace_only` now asserts the explicit materialization error contract (`"Variable 'z' not found in workspace"`) instead of broad `is_err()` acceptance after replace-only import.
  - Validation: `cargo test -p runmat-core workspace_state_roundtrip_replace_only`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 core async spawn eager-interaction contract ratchet
  - Added interaction-backed runtime-model coverage in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs):
    - `spawn_of_async_function_triggers_pause_handler_before_await`
    - `parallel_spawn_inputs_follow_spawn_order_not_await_order`
    - `spawn_error_stops_later_spawn_from_running`
  - This test executes:
    - `async function y = wait_for_key(); pause; y = 1; end;`
    - `t = spawn(wait_for_key()); marker = 7;`
  - and asserts the keypress handler is invoked during `spawn` (before any explicit `await`) with one stdin event and stable state readback (`marker == 7`), providing direct runtime evidence for the current eager async value-lane execution model.
  - Parallel spawn/await interaction ordering is now also ratcheted by prompt order: even when awaiting `t2` before `t1`, prompt events occur in spawn order (`first: ` then `second: `), not await order.
  - Serial failure boundary is now also ratcheted: if first spawned async input interaction fails, later spawns do not run and only the first prompt is observed, with stable runtime identifier `RunMat:input:InteractionFailed`.
  - Validation: `cargo test -p runmat-core --test async_stdin spawn_of_async_function_triggers_pause_handler_before_await -- --nocapture`, `cargo test -p runmat-core --test async_stdin parallel_spawn_inputs_follow_spawn_order_not_await_order -- --nocapture`, `cargo test -p runmat-core --test async_stdin spawn_error_stops_later_spawn_from_running -- --nocapture`, `cargo test -p runmat-core --test async_stdin`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 3/7 import ambiguity/duplicate identifier-contract ratchet
  - Added stable semantic lowering identifiers for import conflict paths in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs):
    - `RunMat:ImportAmbiguous` for ambiguous call/handle/import resolution
    - `RunMat:ImportDuplicate` for duplicate import statements
  - Tightened VM import ambiguity coverage in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs):
    - ambiguity tests now assert `RunMat:ImportAmbiguous` instead of message-substring proxies
    - duplicate-import classstar test now asserts `RunMat:ImportDuplicate`
  - Added direct HIR-layer identifier-contract coverage in [lowering_extras.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/tests/lowering_extras.rs):
    - `import_normalization_and_ambiguity` now asserts `err.identifier == Some("RunMat:ImportAmbiguous")`
  - Added core compile-path identifier-contract coverage in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs):
    - `compile_input_reports_import_ambiguity_identifier`
    - `compile_input_reports_duplicate_import_identifier`
  - Validation: `cargo test -p runmat-vm --test functions import_ambiguity_`, `cargo test -p runmat-vm --test functions import_wildcard_vs_classstar_ambiguity_for_static_method`, `cargo test -p runmat-hir --test lowering_extras import_normalization_and_ambiguity -- --nocapture`, `cargo test -p runmat-core compile_input_reports_import_`, `cargo test -p runmat-core compile_input_reports_duplicate_import_identifier`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 core execution-attempt stats contract ratchet
  - Tightened execution-attempt accounting in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs):
    - `test_error_recovery_and_continued_execution` now asserts exact `stats.total_executions == 3` (valid run, invalid parse, valid recovery run), replacing the previous loose `2..=3` range.
  - This locks the core stats surface to deterministic execute-entry accounting, including parser-stage failures.
  - Validation: `cargo test -p runmat-core --test integration`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 core integration parse-stage contract ratchet
  - Tightened broad error handling in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs):
    - `test_error_recovery_and_continued_execution` now asserts `Err(RunError::Syntax(_))` for the intentionally incomplete matrix literal path, instead of a generic `is_err()` check.
  - This keeps recovery-path coverage pinned to parser-stage contract boundaries rather than any error surface.
  - Validation: `cargo test -p runmat-core --test integration`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 core syntax-failure stage-contract ratchet
  - Tightened broad parse/syntax failure checks in [engine.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/engine.rs):
    - `test_parse_error_handling`
    - `test_invalid_syntax_handling`
  - Both tests now assert `Err(RunError::Syntax(_))` explicitly instead of generic `is_err()`, preserving parser-stage failure contracts through the core execution API.
  - Validation: `cargo test -p runmat-core --test engine`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 core async handle-consumption identifier-contract ratchet
  - Replaced a weak/incorrect async side-effect assumption in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs) with a direct runtime contract for spawned-task handles:
    - `test_spawn_handle_is_consumed_after_await`
  - New coverage now asserts:
    - first `await` on a spawned handle succeeds and preserves computed value (`42`)
    - second `await` on the same handle fails with stable runtime identifier `RunMat:AwaitOperandInvalid`
  - This keeps async integration behavior on semantic identifier/value assertions instead of display or incidental side-effect proxies.
  - Validation: `cargo test -p runmat-core --test integration`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 explicit async runtime-model metadata contract
  - Added explicit semantic async runtime-model metadata in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs):
    - `SemanticAsyncRuntimeModel` (current value: `LazyFutureDescriptorLane`)
    - `SemanticAsyncMetadata.runtime_model`
  - VM compile now records this runtime model in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and interpreter startup diagnostics now surface the model in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs).
  - Updated async metadata ratchets in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs):
    - `primary_compile_records_semantic_spawn_site_metadata`
    - `primary_compile_records_semantic_await_site_metadata`
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_spawn_site_metadata`, `cargo test -p runmat-vm primary_compile_records_semantic_await_site_metadata`, `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo test -p runmat-core --test async_stdin`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 keep semantic fusion groups when accel-node mapping drops out
  - `runmat-vm` compile path in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now falls back to semantic-window-derived fusion groups (empty node lists + semantic span/kind metadata) when semantic instruction windows exist but accel-node mapping produces zero groups.
  - This preserves semantic executable-group scaffolding through bytecode products so runtime fusion-plan preparation can recover/drop groups against the live accel graph, instead of dropping semantic fusion-group artifacts early at compile time.
  - Added unit ratchet:
    - `semantic_windows_fallback_to_empty_node_groups_when_mapping_drops_all_nodes`
  - Validation: `cargo test -p runmat-vm semantic_windows_fallback_to_empty_node_groups_when_mapping_drops_all_nodes`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm semantic_windows_reject_overly_wide_covering_node_spans`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 3/7 command-form `pause` semantic parsing + interaction contract ratchet
  - Added command-form verb support for bare `pause` in [command.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-parser/src/parser/command.rs), aligning no-arg `pause` parsing with existing command-form controls (`clear`, `clc`, `close`, etc.).
  - Added parser coverage in [command_syntax.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-parser/tests/command_syntax.rs):
    - `pause_without_arg_is_command_form`
  - Core async-stdin interaction coverage now runs `pause` in command form and keeps identifier-contract assertion for handler failures in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs):
    - `pause_uses_keypress_handler`
    - `pending_handler_returns_error` (`RunMat:interaction:AsyncHandlerError`)
  - Validation: `cargo test -p runmat-parser --test command_syntax pause_without_arg_is_command_form`, `cargo test -p runmat-core --test async_stdin`, `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 async interaction handler failure identifier-contract ratchet
  - Tightened core stdin interaction coverage in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs):
    - `pending_handler_returns_error` now asserts stable runtime identifier `RunMat:interaction:AsyncHandlerError` instead of accepting any error surface.
    - `pause` interaction flows in async stdin tests now execute via explicit builtin call form `pause()` to keep the ratchet scoped to runtime interaction semantics.
  - Validation: `cargo test -p runmat-core --test async_stdin`, `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic async multi-task await lifecycle ratchet
  - Extended semantic invoker async lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) with multi-outstanding spawn-task flows:
    - `semantic_async_spawn_parallel_await_keeps_retained_handle_and_releases_dropped_handle`
    - `semantic_async_spawn_parallel_await_releases_both_unaliased_handles`
  - These ratchets validate independent await/cleanup behavior across two concurrent spawn handles in one semantic async function frame, including out-of-order await (`t2` then `t1`) and per-handle residency/provider-release outcomes.
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle semantic_async_spawn_parallel_await_keeps_retained_handle_and_releases_dropped_handle`, `cargo test -p runmat-vm --test spawn_semantic_lifecycle semantic_async_spawn_parallel_await_releases_both_unaliased_handles`, `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 tighten semicolon error-path assertions to identifier contracts
  - Replaced weak “any error” checks in [semicolon_suppression.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/semicolon_suppression.rs) (`test_errors_always_shown`) with stable undefined-variable identifier assertions (`RunMat:UndefinedVariable`) for both semicolon and non-semicolon paths.
  - This removes a remaining display/error-surface proxy in core semicolon behavior coverage and keeps the failure contract semantic.
  - Validation: `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 runtime fusion-group sanitization for semantic window mapping drift
  - `runmat-accelerate` fusion-plan preparation in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs) now sanitizes compile-provided semantic fusion groups against the runtime accel graph before executable planning:
    - filters stale/mismatched node IDs
    - enforces kind-compatible node membership
    - recovers empty groups from nearby compatible runtime nodes (bounded overlap/touch tolerance)
    - drops groups that remain unresolved after sanitization
  - Added runtime-plan ratchet coverage:
    - `prepare_fusion_plan_recovers_empty_group_nodes_from_runtime_graph`
    - `prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_graph_is_too_far`
  - Validation: `cargo test -p runmat-accelerate prepare_fusion_plan_recovers_empty_group_nodes_from_runtime_graph`, `cargo test -p runmat-accelerate prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_graph_is_too_far`, `cargo test -p runmat-accelerate prepare_fusion_plan_requires_semantic_candidate_groups`, `cargo test -p runmat-accelerate prepare_fusion_plan_allows_semantic_gated_groups`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-invoker async nested-unrequested varargout release ratchet
  - Extended semantic invoker async lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) with nested-unrequested output cleanup:
    - `semantic_async_spawn_varargout_nested_unrequested_handle_releases`
  - This ratchets unaliased async helper/callee cleanup when the dropped GPU handle is nested in an unrequested `varargout` entry (`{0, {x}}` shape).
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-invoker async multi-output/varargout release evidence ratchet
  - Extended semantic invoker async lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) with unrequested-output helper shapes that carry GPU handles:
    - `semantic_async_spawn_multi_output_helper_unrequested_handle_releases`
    - `semantic_async_spawn_varargout_helper_unrequested_handle_releases`
  - This ratchets unaliased async helper/callee cleanup for multi-output and `varargout` helper forms when only a single output is requested by the caller.
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-window disjoint-gap tolerance ratchet for accel-node mapping
  - `runmat-vm` semantic window -> accel-node mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now includes a bounded disjoint-gap fallback (<=1 instruction) when strict overlap matching yields no nodes.
  - This reduces dropouts from tiny graph/window span jitter while preserving rejection when disjoint gaps exceed tolerance.
  - Added mapping regression coverage:
    - `semantic_windows_map_accel_nodes_with_small_disjoint_gap`
    - `semantic_windows_reject_accel_nodes_with_large_disjoint_gap`
  - Validation: `cargo test -p runmat-vm semantic_windows_map_accel_nodes_with_small_disjoint_gap`, `cargo test -p runmat-vm semantic_windows_reject_accel_nodes_with_large_disjoint_gap`, `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-invoker async cell-helper release evidence ratchet
  - Extended semantic invoker async lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) with cell-nested helper payload release behavior:
    - `semantic_async_spawn_await_cell_helper_releases_unaliased_provider_handle`
  - This ratchets unaliased async helper/callee cleanup across another nested payload shape (`{x}` cell payload) through semantic invoker execution.
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-invoker async nested-helper release evidence ratchet
  - Extended semantic invoker async lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) with nested helper payload release behavior:
    - `semantic_async_spawn_await_struct_helper_releases_unaliased_provider_handle`
  - This ratchets unaliased async helper/callee cleanup across structured payload shapes (`struct('payload', x)`) through semantic invoker execution, reducing the remaining async spawned-workload cleanup evidence gap.
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-invoker async helper/callee release cleanup ratchet
  - `runmat-vm` semantic function invocation now clears non-output GPU-handle residency/storage from semantic function result slots after output extraction in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs), while preserving live handles reachable through returned outputs and runtime global/persistent roots.
  - Extended semantic lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) with an async helper/callee unaliased release ratchet executed through the semantic invoker path:
    - `semantic_async_spawn_await_helper_overwrite_releases_unaliased_provider_handle`
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-window overlap-tolerance ratchet for accel-node mapping
  - `runmat-vm` semantic window -> accel-node mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now accepts small partial-overlap boundary drift (<=1 instruction on both boundaries) when strict contained/covering span checks do not match.
  - This reduces fragility to minor accel-graph span jitter while preserving strict rejection of broad overlap coupling.
  - Added mapping regression coverage:
    - `semantic_windows_map_accel_nodes_with_small_boundary_shift_overlap`
    - `semantic_windows_reject_partial_overlap_with_large_boundary_shift`
  - Validation: `cargo test -p runmat-vm semantic_windows_`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic async spawn/await lifecycle coverage extension
  - Extended provider-backed semantic lifecycle coverage in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs) to include compiled semantic `async` function flows that execute explicit `spawn` + `await` bytecode boundaries:
    - `semantic_async_spawn_await_overwrite_unaliased_executes_with_scalar_output`
    - `semantic_async_spawn_await_overwrite_preserves_provider_handle_when_alias_retained`
  - Async alias-retained flow now ratchets provider residency/storage preservation through semantic function bytecode execution; unaliased async flow currently ratchets execution/output contract while deeper async release semantics remain tracked as an open Plan 7 gap.
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic spawned-workload lifecycle evidence ratchet
  - Added provider-backed VM integration coverage that executes compiled semantic functions (not synthetic stack setup) for `spawn(...)` overwrite/drop flows in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs):
    - `semantic_spawn_overwrite_releases_unaliased_provider_handle`
    - `semantic_spawn_overwrite_preserves_provider_handle_when_alias_retained`
  - These tests compile semantic function bodies through HIR->MIR->VM function bytecode, inject known GPU-handle inputs, and assert real provider residency/storage outcomes after spawn-handle lifecycle transitions.
  - Validation: `cargo test -p runmat-vm --test spawn_semantic_lifecycle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-window-driven fusion compile gating
  - `runmat-vm` compile-time fusion setup in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now gates accel-graph realization directly on semantic instruction windows (`semantic_instruction_windows.is_empty()`) rather than a second bytecode accel-capability heuristic pass.
  - This removes duplicated bytecode capability gating from the runtime compile path and makes executable fusion-group admission depend on semantic-window artifacts first, with accel graph used for node realization/stack-layout annotation after that gate.
  - Validation: `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group`, `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 3 private-property access identifier contract ratchet
  - Added stable `RunMat:PropertyPrivateAccess` diagnostics across both VM object resolution and runtime object-field builtins:
    - [resolve.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/object/resolve.rs)
    - [getfield.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/getfield.rs)
    - [setfield.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/setfield.rs)
  - Updated semantic VM coverage in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs) to assert identifier contracts for private property get/set failures:
    - `classes_static_and_inheritance`
    - `classes_property_access_attributes`
  - Validation: `cargo test -p runmat-vm --test functions classes_static_and_inheritance -- --exact`, `cargo test -p runmat-vm --test functions classes_property_access_attributes -- --exact`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 3 semantic class-attribute identifier contract ratchet
  - Added stable semantic-lowering identifiers for class attribute conflict diagnostics in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs):
    - `RunMat:ClassPropertyAttributeConflict` for incompatible property attribute combinations (for example `Constant` + `Dependent`).
    - `RunMat:ClassMethodAttributeConflict` for incompatible method attribute combinations (for example `Abstract` + `Sealed`).
  - Updated VM semantic compile coverage in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs) to assert those identifiers directly (`class_property_attribute_conflicts_error`, `class_method_attribute_conflicts_error`) instead of message-substring checks.
  - Validation: `cargo test -p runmat-vm --test functions class_property_attribute_conflicts_error -- --exact`, `cargo test -p runmat-vm --test functions class_method_attribute_conflicts_error -- --exact`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 identifier-contract hardening for indexing and object overload failures
  - Replaced message-fallback assertions with strict identifier assertions in VM semantic coverage:
    - `index_step_zero_mex` now asserts `RunMat:IndexStepZero`
    - `unsupported_cell_index_type_mex` now asserts `RunMat:CellIndexType`
    - `oop_negative_missing_subsref_mex` now asserts `RunMat:MissingSubsref`
    - `oop_negative_missing_subsasgn_mex` now asserts `RunMat:MissingSubsasgn`
  - Added concrete identifier emission where gaps were discovered:
    - `colon` zero-increment failures now carry `RunMat:IndexStepZero` in [colon.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/creation/colon.rs).
    - object-index dispatch for missing class overloads now emits `RunMat:MissingSubsref` / `RunMat:MissingSubsasgn` via runtime dispatch normalization in [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/lib.rs), and VM indexing dispatch now guards object indexing descriptor paths consistently in [indexing.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/indexing.rs).
  - Validation: `cargo test -p runmat-vm --test control_flow`, `cargo test -p runmat-vm --test indexing_properties`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 completion-boundary residency keep-set ratchet for local aliases
  - Updated VM interpreter completion cleanup in `run_interpreter_inner` to preserve live handles referenced by `context.locals` when clearing stack-dropped values.
  - Added/ratcheted runner coverage for spawned payload alias liveness through locals at completion/pop boundaries:
    - `spawn_pop_preserves_provider_handle_when_payload_still_live_in_locals`
    - `spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live_in_locals`
    - `spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live_in_locals`
  - Validation: `cargo test -p runmat-vm --lib spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live_in_locals`, `cargo test -p runmat-vm --lib spawn_pop_preserves`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 await-flow ratchet for local-slot spawn-handle aliases
  - Added runner-level await-flow coverage for spawn-handle alias liveness when aliases are carried through local slots:
    - `await_succeeds_after_overwriting_one_local_spawn_handle_alias`
    - `await_succeeds_after_overwriting_var_alias_when_local_spawn_handle_alias_live`
  - This extends stale-ID/alias-liveness execution evidence beyond var-only alias flows to local-slot alias replacement flows.
  - Validation: `cargo test -p runmat-vm await_succeeds_after_overwriting_one_local_spawn_handle_alias`, `cargo test -p runmat-vm await_succeeds_after_overwriting_var_alias_when_local_spawn_handle_alias_live`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 overwrite liveness ratchet for direct-handle aliases in locals
  - Added provider-backed VM runner coverage for overwrite cleanup where direct `GpuTensor` aliases remain live via `locals`:
    - `store_var_overwrite_preserves_provider_handle_when_shared_in_local`
    - `store_local_overwrite_preserves_provider_handle_when_shared_in_other_local`
  - This extends overwrite-path shared-liveness evidence to local-alias branches for direct handles (not only var aliases).
  - Validation: `cargo test -p runmat-vm store_var_overwrite_preserves_provider_handle_when_shared_in_local`, `cargo test -p runmat-vm store_local_overwrite_preserves_provider_handle_when_shared_in_other_local`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 overwrite liveness ratchet for nested handle-object aliases in locals
  - Added provider-backed VM runner coverage for overwrite cleanup where nested `HandleObject` aliases remain live via `locals`:
    - `store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_local`
    - `store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_local`
  - This extends overwrite-path shared-liveness evidence to local-alias branches beyond var-alias-only cases.
  - Validation: `cargo test -p runmat-vm store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_local`, `cargo test -p runmat-vm store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_local`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 local-slot replacement ID-liveness ratchet for nested handle-object values
  - Added VM dispatch coverage for local-slot overwrite retirement with `excluded_local` semantics when spawn-task handles are nested under `Value::HandleObject` targets:
    - `replaced_nested_spawn_task_handle_in_local_slot_retires_with_excluded_local`
    - `replaced_nested_spawn_task_handle_in_local_slot_keeps_id_when_other_local_alias_live`
  - This ratchets `StoreLocal`-path ID retirement correctness for nested task handles across unaliased and alias-live local replacement states.
  - Validation: `cargo test -p runmat-vm replaced_nested_spawn_task_handle_in_local_slot_retires_with_excluded_local`, `cargo test -p runmat-vm replaced_nested_spawn_task_handle_in_local_slot_keeps_id_when_other_local_alias_live`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 `StoreLocal` overwrite release-path ratchet for direct handles
  - Added provider-backed VM runner coverage for direct local-slot overwrite cleanup when no live alias remains:
    - `store_local_overwrite_releases_provider_handle_when_unaliased`
  - This closes the direct `StoreLocal` release-side counterpart to existing preserve-side liveness coverage.
  - Validation: `cargo test -p runmat-vm store_local_overwrite_releases_provider_handle_when_unaliased`, `cargo test -p runmat-vm store_local_overwrite_preserves_provider_handle_when_shared_in_var`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 spawn-task ID replacement-liveness ratchet for nested handle-object values
  - Added VM dispatch coverage for overwrite replacement retirement when spawn-task handles are nested under `Value::HandleObject` targets:
    - `replaced_nested_spawn_task_handle_in_handle_object_retires_task_id_when_unaliased`
    - `replaced_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live`
  - This ratchets that replacement-path ID retirement honors nested handle-object traversal and alias-liveness preservation.
  - Validation: `cargo test -p runmat-vm replaced_nested_spawn_task_handle_in_handle_object_retires_task_id_when_unaliased`, `cargo test -p runmat-vm replaced_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 `StoreLocal` lifecycle ratchet for nested handle-object overwrite paths
  - Added provider-backed VM runner coverage for local-slot overwrite cleanup where dropped/shared handles are nested under `Value::HandleObject` values:
    - `store_local_overwrite_releases_nested_handle_object_provider_handle_when_unaliased`
    - `store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_var`
  - This extends `Instr::StoreLocal` shared-liveness evidence beyond direct tensor values to handle-object target traversal.
  - Validation: `cargo test -p runmat-vm store_local_overwrite_releases_nested_handle_object_provider_handle_when_unaliased`, `cargo test -p runmat-vm store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_var`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 `StoreVar` lifecycle ratchet for nested handle-object overwrite paths
  - Added provider-backed VM runner coverage for var-slot overwrite cleanup where dropped/shared handles are nested under `Value::HandleObject` values:
    - `store_var_overwrite_releases_nested_handle_object_provider_handle_when_unaliased`
    - `store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_var`
  - This extends `Instr::StoreVar` shared-liveness evidence beyond direct tensor values to handle-object target traversal.
  - Validation: `cargo test -p runmat-vm store_var_overwrite_releases_nested_handle_object_provider_handle_when_unaliased`, `cargo test -p runmat-vm store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_var`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 `ExitScope` lifecycle ratchet for nested handle-object locals
  - Added provider-backed VM runner coverage for local-scope drop cleanup where GPU handles are nested under `Value::HandleObject` local values:
    - `exit_scope_releases_nested_handle_object_local_provider_handle`
    - `exit_scope_preserves_nested_handle_object_provider_handle_when_still_live_in_vars`
  - This extends `Instr::ExitScope` shared-liveness evidence beyond direct local tensor handles to handle-object target traversal.
  - Validation: `cargo test -p runmat-vm exit_scope_releases_nested_handle_object_local_provider_handle`, `cargo test -p runmat-vm exit_scope_preserves_nested_handle_object_provider_handle_when_still_live_in_vars`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 provider-release lifecycle ratchet for handle-object spawned payloads
  - Added provider-backed VM runner coverage for spawned payloads where GPU handles are nested under `Value::HandleObject` targets:
    - `spawn_await_completion_releases_nested_handle_object_target_provider_handle`
    - `spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live`
    - `spawn_pop_releases_nested_handle_object_target_provider_handle`
    - `spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live`
  - This extends spawn completion-path (`Await` and `Pop`) residency/provider-release evidence beyond direct struct/object/cell payloads to handle-object target traversal.
  - Validation: `cargo test -p runmat-vm spawn_await_completion_releases_nested_handle_object_target_provider_handle`, `cargo test -p runmat-vm spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live`, `cargo test -p runmat-vm spawn_pop_releases_nested_handle_object_target_provider_handle`, `cargo test -p runmat-vm spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 handle-object traversal hardening for spawn policy and task-ID liveness
  - VM spawn GPU-handle concurrency policy traversal now recurses through `Value::HandleObject` targets (cycle-safe) so nested provider handles are enforced at spawn boundaries.
  - VM spawn-task ID retirement/liveness now collects task IDs recursively across nested runtime shapes (including `HandleObject` targets) instead of only top-level task structs.
  - Added VM dispatch coverage:
    - `spawn_policy_rejects_gpu_handles_nested_in_handle_object_target`
    - `dropped_nested_spawn_task_handle_in_handle_object_retires_task_id`
    - `dropped_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live`
  - Validation: `cargo test -p runmat-vm spawn_policy_rejects_gpu_handles_nested_in_handle_object_target`, `cargo test -p runmat-vm dropped_nested_spawn_task_handle_in_handle_object_retires_task_id`, `cargo test -p runmat-vm dropped_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Validation ratchet tighten try/catch message-binding assertion shape
  - Core semantic try/catch binding coverage now asserts the exact bound message value shape for `y = err.message` (`'boom'`) instead of substring matching.
  - Updated test: `try_catch_binding_uses_semantic_vm` in `crates/runmat-core/src/tests.rs`.
  - Validation: `cargo test -p runmat-core try_catch_binding_uses_semantic_vm`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Validation ratchet replace VM indexing error display proxies with identifier assertions
  - Tightened VM basics error coverage to assert stable semantic error identifiers instead of display-message substring proxies:
    - `fft_end_arithmetic_out_of_bounds_raises_error` now asserts `RunMat:IndexOutOfBounds` / `RunMat:SubscriptOutOfBounds`.
    - `scalar_slice_with_nonnumeric_selector_errors` now asserts `RunMat:UnsupportedIndexType` / `RunMat:SliceNonTensor`.
  - Validation: `cargo test -p runmat-vm fft_end_arithmetic_out_of_bounds_raises_error`, `cargo test -p runmat-vm scalar_slice_with_nonnumeric_selector_errors`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 handle-object target recursion for residency/provider-release lifecycle
  - VM residency traversal now recurses through `Value::HandleObject` targets for both:
    - residency/provider clear on drop (`clear_value` path)
    - shared-liveness exclusion keep-set collection (`clear_value_excluding` path)
  - Added cycle-safe target traversal guards (visited target pointer set) to prevent recursion loops when handle-object targets are cyclic.
  - Added VM residency coverage:
    - `clear_value_releases_gpu_handles_nested_in_handle_object_target`
    - `clear_value_excluding_preserves_handles_referenced_in_handle_object_target`
  - Validation: `cargo test -p runmat-vm clear_value_releases_gpu_handles_nested_in_handle_object_target`, `cargo test -p runmat-vm clear_value_excluding_preserves_handles_referenced_in_handle_object_target`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 nested struct payload provider-release lifecycle ratchet
  - Added VM runner coverage for spawned payloads that carry provider handles through nested struct runtime value shapes:
    - `spawn_await_completion_releases_nested_struct_provider_handle`
    - `spawn_await_completion_releases_nested_object_property_provider_handle`
    - `spawn_await_completion_preserves_nested_object_property_handle_when_alias_live`
    - `spawn_await_completion_releases_nested_cell_provider_handle`
    - `spawn_await_completion_preserves_nested_cell_handle_when_alias_live`
  - This extends spawn/await lifecycle cleanup evidence beyond closure/output-list payloads to struct-nested and object-property-nested handle payloads.
  - Validation: `cargo test -p runmat-vm spawn_await_completion_releases_nested_struct_provider_handle`, `cargo test -p runmat-vm spawn_await_completion_releases_nested_object_property_provider_handle`, `cargo test -p runmat-vm spawn_await_completion_preserves_nested_object_property_handle_when_alias_live`, `cargo test -p runmat-vm spawn_await_completion_releases_nested_cell_provider_handle`, `cargo test -p runmat-vm spawn_await_completion_preserves_nested_cell_handle_when_alias_live`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 deliverable audit closeout to `met`
  - Completed production consumer audit for nominal class/builtin metadata usage across runtime+VM callsites.
  - Verified class metadata consumers route through inheritance-aware lookup boundaries and cycle-safe traversal paths (`lookup_method`/`lookup_property` plus cycle-guarded parent traversal in `isa`, `fieldnames`, and constructor hierarchy walk).
  - Updated deliverable audit status for item 5 from `partial` to `met`.
  - Added regression watchpoint grep to prevent direct-class-only method dispatch reintroduction:
    - `rg -n "cls\\.methods\\.get\\(|class_def\\.methods\\.get\\(" crates/runmat-runtime/src crates/runmat-vm/src`

- (pending commit) Plan 7 semantic-window mapping tolerates accel-node span widening
  - VM semantic fusion-group node mapping now accepts accel nodes that fully cover a semantic instruction-window span only when widening is bounded (±1 instruction), in addition to strict contained-by-window nodes.
  - This keeps semantic-window realization robust to small accel-graph span widening while preventing broad-node absorption that can reintroduce non-semantic overlap coupling.
  - Added compile-level regression coverage:
    - `semantic_windows_map_accel_nodes_that_cover_window_span`
    - `semantic_windows_reject_overly_wide_covering_node_spans`
  - Validation: `cargo test -p runmat-vm semantic_windows_map_accel_nodes_that_cover_window_span`, `cargo test -p runmat-vm semantic_windows_reject_overly_wide_covering_node_spans`, `cargo test -p runmat-vm semantic_candidates_with_partial_overlap_do_not_build_fusion_groups`, `cargo test -p runmat-vm semantic_windows_map_accel_nodes_without_semantic_tags`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 constructor fallback metadata normalization in runtime dispatcher
  - Runtime builtin-dispatch constructor fallback now uses inheritance-aware class-method lookup for same-name constructor metadata and enforces static/public constructor dispatch policy.
  - Constructor fallback now default-constructs object instances when same-name constructor metadata is private or non-static instead of invoking those methods.
  - Added runtime dispatcher coverage:
    - `constructor_fallback_uses_inherited_static_constructor_metadata`
    - `constructor_fallback_skips_private_or_non_static_constructor_methods`
  - Validation: `cargo test -p runmat-runtime constructor_fallback_uses_inherited_static_constructor_metadata`, `cargo test -p runmat-runtime constructor_fallback_skips_private_or_non_static_constructor_methods`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 end-to-end inherited object protocol dispatch ratchets
  - Added VM object resolution coverage asserting missing member read/write paths dispatch through inherited protocol methods, not only direct-class protocol metadata:
    - `load_member_uses_inherited_subsref_for_missing_property`
    - `store_member_uses_inherited_subsasgn_for_missing_property`
  - Both tests register parent/child class metadata and verify child object member access routes into inherited `subsref`/`subsasgn` runtime handlers with concrete semantic outcomes.
  - Validation: `cargo test -p runmat-vm load_member_uses_inherited_subsref_for_missing_property`, `cargo test -p runmat-vm store_member_uses_inherited_subsasgn_for_missing_property`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 constructor metadata traversal hardening for cyclic class inheritance
  - Runtime object construction (`new_object`) now guards class-parent traversal against metadata cycles when collecting inherited default property initialization chain.
  - Added runtime regression coverage:
    - `new_object_builtin_handles_class_parent_cycles`
  - The test registers a cyclic parent relationship across synthetic classes and asserts constructor initialization terminates deterministically while applying defaults from both classes.
  - Validation: `cargo test -p runmat-runtime new_object_builtin_handles_class_parent_cycles`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 inherited `subsref/subsasgn` metadata gating for VM object member dispatch
  - VM object member dispatch metadata checks now use inheritance-aware class-method lookup for protocol members (`subsref`, `subsasgn`) instead of direct-class-only method maps.
  - Added VM shared call helper coverage:
    - `class_defines_member_subsref_includes_inherited_method_metadata`
    - `class_defines_member_subsasgn_includes_inherited_method_metadata`
  - This ratchets object member fallback dispatch gating to unified nominal class metadata inheritance semantics.
  - Validation: `cargo test -p runmat-vm class_defines_member_subsref_includes_inherited_method_metadata`, `cargo test -p runmat-vm class_defines_member_subsasgn_includes_inherited_method_metadata`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 VM object static-member inheritance consumer ratchets
  - Added VM object resolution coverage that asserts inherited class metadata controls static member read/write resolution through child class refs:
    - `load_static_member_resolves_inherited_static_property_value`
    - `store_member_updates_inherited_static_property_owner_slot`
    - `load_static_member_resolves_inherited_static_method`
  - These tests verify inherited static properties/methods resolve through parent metadata ownership and that static property writes through child class refs update the owner class slot.
  - Validation: `cargo test -p runmat-vm load_static_member_resolves_inherited_static_property_value`, `cargo test -p runmat-vm store_member_updates_inherited_static_property_owner_slot`, `cargo test -p runmat-vm load_static_member_resolves_inherited_static_method`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 metadata traversal hardening for `isa` inheritance
  - `isa` class inheritance traversal now guards against parent-cycle metadata loops during nominal class ancestry checks.
  - Added runtime regression coverage:
    - `isa_inheritance_walk_handles_parent_cycles`
  - The test registers a cyclic parent relationship across synthetic classes and asserts `isa` returns deterministically (`false` for missing target, `true` for reachable parent class) without loop behavior.
  - Validation: `cargo test -p runmat-runtime isa_inheritance_walk_handles_parent_cycles`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 runtime fieldnames inheritance-aware class metadata lookup
  - `fieldnames` class-property discovery now traverses class metadata parent chains (with cycle guard) instead of reading only the immediate class definition.
  - Added runtime coverage:
    - `fieldnames_object_includes_inherited_class_properties`
  - This ratchets runtime object introspection behavior to shared nominal class metadata inheritance semantics.
  - Validation: `cargo test -p runmat-runtime fieldnames_object_includes_inherited_class_properties`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 runtime handle-object fieldnames inheritance consumer ratchet
  - Added runtime coverage ensuring handle-object introspection includes inherited class metadata properties:
    - `fieldnames_handle_object_includes_inherited_class_properties`
  - The test registers parent/child class metadata and asserts `fieldnames(handle)` emits child class properties, inherited parent properties, and target payload fields in sorted output order.
  - Validation: `cargo test -p runmat-runtime fieldnames_handle_object_includes_inherited_class_properties`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 class-registry lookup cycle hardening for inherited metadata consumers
  - `runmat-builtins` class metadata lookup now guards against parent-cycle loops in both:
    - `lookup_property`
    - `lookup_method`
  - Added class-registry coverage:
    - `method_lookup_handles_parent_cycle`
    - `property_lookup_handles_parent_cycle`
    - `property_lookup_uses_parent_class_metadata_chain`
  - This hardens inherited metadata traversal behavior used by runtime consumers (`getfield`/`setfield` property resolution and `exist(..., 'method')` lookup) against cyclic class-parent metadata.
  - Validation: `cargo test -p runmat-builtins class_registry_tests`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 runtime property access ratchets for inherited class metadata
  - Added runtime object-property coverage ensuring inherited class metadata is consumed by property access builtins:
    - `getfield_inherited_dependent_property_uses_parent_metadata`
    - `setfield_rejects_inherited_static_property_assignment`
  - The `getfield` ratchet verifies dependent-property fallback (`<name>_backing`) resolves through parent class metadata.
  - The `setfield` ratchet verifies static-property assignment rejection still applies when the static property is declared on an ancestor class.
  - Validation: `cargo test -p runmat-runtime getfield_inherited_dependent_property_uses_parent_metadata`, `cargo test -p runmat-runtime setfield_rejects_inherited_static_property_assignment`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 nested spawned-payload provider-release lifecycle ratchets
  - Added VM runner coverage for spawned payloads that carry provider handles through nested runtime value shapes:
    - `spawn_pop_releases_nested_closure_captured_provider_handle`
    - `spawn_await_completion_releases_nested_output_list_provider_handle`
  - This ratchets that spawn lifecycle cleanup (`Pop` and `Await` completion) releases provider-backed handles for nested `Closure` captures and `OutputList` payloads, not only direct scalar `GpuTensor` payloads.
  - Validation: `cargo test -p runmat-vm spawn_pop_releases_nested_closure_captured_provider_handle`, `cargo test -p runmat-vm spawn_await_completion_releases_nested_output_list_provider_handle`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 `ExitScope` provider-handle lifecycle correctness for shared liveness
  - VM dispatch `Instr::ExitScope` now clears dropped local values with live-value exclusion (`stack` + `vars` + remaining `locals`) instead of unconditional residency clear.
  - This prevents premature provider release when a GPU handle remains live outside the exiting local scope.
  - Added VM runner coverage:
    - `exit_scope_releases_local_only_provider_handle`
    - `exit_scope_preserves_provider_handle_when_still_live_in_vars`
    - `await_rejects_spawn_task_handle_after_scope_exit_retires_id`
  - Validation: `cargo test -p runmat-vm exit_scope_releases_local_only_provider_handle`, `cargo test -p runmat-vm exit_scope_preserves_provider_handle_when_still_live_in_vars`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 overwrite-path provider-handle shared-liveness preservation
  - VM dispatch overwrite cleanup for `Instr::StoreVar` and `Instr::StoreLocal` now uses live-value exclusion across stack/vars/locals instead of current-vs-incoming-only exclusion.
  - This prevents releasing provider storage for a handle being overwritten in one slot when the same handle remains live in another slot.
  - Added VM runner coverage:
    - `store_var_overwrite_preserves_provider_handle_when_shared_in_other_var`
    - `store_local_overwrite_preserves_provider_handle_when_shared_in_var`
  - Validation: `cargo test -p runmat-vm store_var_overwrite_preserves_provider_handle_when_shared_in_other_var`, `cargo test -p runmat-vm store_local_overwrite_preserves_provider_handle_when_shared_in_var`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 spawn-task ID retirement shared-liveness preservation
  - VM dispatch task-ID retirement for dropped/replaced spawn handles now checks live aliases across stack/vars/locals before retiring IDs.
  - This prevents stale `await` failures when one alias is overwritten/dropped while another alias still carries the same spawn task handle.
  - Added coverage:
    - `dropped_spawn_task_handle_keeps_id_when_alias_still_live` (dispatch-level)
    - `await_succeeds_after_overwriting_one_spawn_handle_alias` (runner-level)
  - Validation: `cargo test -p runmat-vm dropped_spawn_task_handle_keeps_id_when_alias_still_live`, `cargo test -p runmat-vm await_succeeds_after_overwriting_one_spawn_handle_alias`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 runtime consumer ratchet for class-metadata inheritance lookup
  - Added runtime `exist` builtin coverage that asserts method existence queries consume registered class metadata through inheritance lookup:
    - `exist_method_uses_registered_class_metadata_including_inheritance`
  - The test registers synthetic parent/child classes via `runmat_builtins::register_class` and verifies:
    - direct method lookup resolves (`Parent.parentOnly`)
    - inherited lookup resolves (`Child.parentOnly`)
    - missing method reports not found
  - Validation: `cargo test -p runmat-runtime exist_method_uses_registered_class_metadata_including_inheritance`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 6 class-registry metadata ratchets for nominal static/inheritance lookup
  - Added `runmat-builtins` class-registry tests that ratchet language-facing metadata behavior for runtime/primitive nominal classes:
    - `primitive_classes_expose_static_zeros_method_metadata`
    - `method_lookup_uses_parent_class_metadata_chain`
  - Coverage now asserts primitive class static method metadata (`double`/`single`/`logical` -> `zeros`) and parent-chain method resolution through class metadata lookup.
  - Validation: `cargo test -p runmat-builtins primitive_classes_expose_static_zeros_method_metadata`, `cargo test -p runmat-builtins method_lookup_uses_parent_class_metadata_chain`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 5 evidence closeout for manifest-driven resolver wiring
  - Re-ran cross-consumer resolver/symbol-discovery tests across config/core/CLI/LSP to verify shared `runmat-config` ownership is active end-to-end:
    - `cargo test -p runmat-config --test project_manifest resolve_project_source_input_from_`
    - `cargo test -p runmat-core source_input_path_`
    - `cargo test -p runmat-core compile_input_resolves_wildcard_import_`
    - `cargo test -p runmat --lib resolve_script_input_`
    - `cargo test -p runmat --lib resolve_benchmark_input_`
    - `cargo test -p runmat-lsp source_context_symbol_discovery_reads_manifest_project_symbols`
  - Updated deliverable audit status for manifest-driven composition/entrypoint wiring from `partial` to `met`, leaving only regression watchpoint coverage.

- (pending commit) Plan 7 semantic-window mapping fallback for untagged accel nodes
  - VM semantic fusion-group mapping now keeps span-matched accel nodes eligible when accel-graph semantic tags are absent, instead of dropping semantic windows solely due missing node tags.
  - Kind mismatch filtering for explicitly-tagged reduction/matmul nodes remains enforced.
  - Added compile-level ratchet coverage:
    - `semantic_windows_map_accel_nodes_without_semantic_tags`
  - Validation: `cargo test -p runmat-vm semantic_windows_map_accel_nodes_without_semantic_tags`, `cargo test -p runmat-vm semantic_elementwise_window_excludes_reduction_nodes`, `cargo test -p runmat-vm semantic_reduction_window_accepts_reduction_nodes`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_transpose_nodes`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Core engine empty/whitespace execution contract ratchet
  - Tightened `runmat-core` engine coverage to require deterministic semantic behavior for empty and whitespace-only inputs.
  - `test_empty_input_handling` and `test_whitespace_only_input` now assert successful execution and absence of runtime diagnostics, replacing proxy assertions that only required a non-empty error display string.
  - Validation: `cargo test -p runmat-core test_empty_input_handling`, `cargo test -p runmat-core test_whitespace_only_input`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Command-form semantic preservation while tightening undefined-variable assertions
  - Parser command-form gating now treats `StringifyWords` verbs as zero-arg-capable and includes explicit `clc` command classification, restoring semantic command parsing for `clear;`, `close;`, and `clc;`.
  - Added parser coverage in `command_syntax.rs`:
    - `clear_without_arg_is_command_form`
    - `clc_without_arg_is_command_form`
  - `runmat-core` command-control coverage now asserts semantic undefined-variable identifier contracts (`RunMat:UndefinedVariable`) instead of message substring matching.
  - Validation: `cargo test -p runmat-parser --test command_syntax`, `cargo test -p runmat-core --test command_controls`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Semantic identifier contract for undefined-variable lowering
  - Added explicit HIR lowering identifier `RunMat:UndefinedVariable` for unresolved identifier reads in `lower_expr_semantic_requested`.
  - Updated VM semantic test helper to preserve lowering identifiers by converting `SemanticError` through `runmat_vm::CompileError` instead of debug-string wrapping.
  - Tightened VM control-flow coverage `undefined_variable_raises_mex` to assert `err.identifier() == Some("RunMat:UndefinedVariable")`.
  - Added HIR ratchet coverage `undefined_variable_errors_have_stable_identifier`.
  - Validation: `cargo test -p runmat-hir undefined_variable_errors_have_stable_identifier`, `cargo test -p runmat-vm undefined_variable_raises_mex`.

- (pending commit) Validation ratchet replace display-proxy control-flow checks with semantic assertions
  - Tightened control-flow coverage in `runmat-core` tests to assert deterministic execution/value contracts instead of accepting any non-empty error display string.
  - Updated integration coverage in `test_control_flow_execution` to require successful execution for `if`/`if-else`/`for`/`while-break` and explicit readback values (`x = 10`, `y = 30`, `z = 6`).
  - Updated engine coverage in `test_execution_with_control_flow` to require successful execution plus explicit readback values (`x = 5`, `y = 3`).
  - Validation: `cargo test -p runmat-core test_control_flow_execution`, `cargo test -p runmat-core test_execution_with_control_flow`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 retire dropped spawn-task IDs on VM value-drop boundaries
  - VM dispatch now retires registered spawn-task IDs when spawned task-handle values are dropped or replaced across value-drop boundaries:
    - `Instr::Pop`
    - `Instr::ExitScope`
    - overwrite paths in `Instr::StoreVar` / `Instr::StoreLocal`
  - Replacement retirement is ID-aware and preserves live IDs on self-reassignment (`t = t`) while retiring IDs when replaced by a different/non-task value.
  - Added dispatch regression coverage:
    - `dropped_spawn_task_handle_retires_task_id`
    - `spawn_task_id_extraction_ignores_non_task_structs`
    - `replaced_spawn_task_id_is_retired_when_incoming_differs`
    - `replacing_with_same_spawn_task_keeps_id_registered`
  - Added runner regression coverage:
    - `await_succeeds_after_spawn_handle_self_reassignment`
  - Validation: `cargo test -p runmat-vm dropped_spawn_task_handle_retires_task_id`, `cargo test -p runmat-vm spawn_task_id_extraction_ignores_non_task_structs`, `cargo test -p runmat-vm replaced_spawn_task_id_is_retired_when_incoming_differs`, `cargo test -p runmat-vm replacing_with_same_spawn_task_keeps_id_registered`, `cargo test -p runmat-vm spawn_`, `cargo test -p runmat-vm await_`, `cargo fmt --all --check`.

- (pending commit) Plan 7 enforce semantic window-kind compatibility in fusion-node mapping
  - VM semantic fusion-window -> accel-node mapping now filters by semantic window kind compatibility (not just generic accel-tag presence):
    - elementwise windows exclude reduction/matmul-tagged nodes
    - reduction windows exclude matmul-tagged nodes
    - matmul windows keep current permissive semantic-signal acceptance
  - Added compile-level ratchet coverage:
    - `semantic_elementwise_window_excludes_reduction_nodes`
    - `semantic_reduction_window_accepts_reduction_nodes`
  - Validation: `cargo test -p runmat-vm semantic_elementwise_window_excludes_reduction_nodes`, `cargo test -p runmat-vm semantic_reduction_window_accepts_reduction_nodes`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_transpose_nodes`, `cargo fmt --all --check`.

- (pending commit) Plan 7 semantic-tag-driven fusion-node mapping for semantic windows
  - VM semantic fusion-window -> accel-node mapping now selects candidate nodes using accel semantic tags (`Unary`/`Elementwise`/`Reduction`/`MatMul`/`Transpose`) instead of a hard-coded node-category allowlist.
  - This removes a brittle category coupling that previously excluded transpose-tagged accel nodes from semantic-window fusion groups.
  - Added compile regression coverage `semantic_candidates_build_fusion_groups_from_transpose_nodes` to ratchet transpose node inclusion under semantic window mapping.
  - Validation: `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_transpose_nodes`, `cargo test -p runmat-vm semantic_window_kind_is_not_overridden_by_graph_category`, `cargo fmt --all --check`.

- (pending commit) Validation cadence keep workspace check green
  - `cargo check --workspace` was failing on `runmat-lsp` due `-D warnings` dead-code rejection for `analyze_document_with_compat`.
  - Added a narrow non-test dead-code allowance guard on that helper (`#[cfg_attr(not(test), allow(dead_code))]`) in `crates/runmat-lsp/src/core/analysis.rs`.
  - Validation: `cargo check --workspace`, `cargo fmt --all --check`.

- (pending commit) Plan 7 ratchet await task-handle runtime contract through core integration
  - Added `runmat-core` integration coverage `test_await_rejects_non_spawn_task_operand_at_runtime`, asserting `await(1)` returns runtime identifier `RunMat:AwaitOperandInvalid` in the session execution envelope.
  - Validation: `cargo test -p runmat-core --test integration test_await_rejects_non_spawn_task_operand_at_runtime`, `cargo fmt --all --check`.

- (pending commit) Plan 7 introduce explicit spawned-task value lane at VM runtime boundary
  - VM dispatch `Instr::Spawn` now wraps payload values into an explicit spawned-task handle shape (struct-backed task record), and `Instr::Await` now validates/unwraps that handle instead of treating both opcodes as pure no-ops.
  - Added explicit await-operand contract diagnostics (`RunMat:AwaitOperandInvalid`) for non-task operands and malformed task records.
  - Added provider-backed lifecycle coverage for spawned-task payload handles without await:
    - `spawn_pop_releases_stack_only_provider_handle`
    - `spawn_pop_preserves_provider_handle_when_payload_still_live_in_vars`
  - Added dispatch/unit coverage for task-handle wrap/unwrap shape and await rejection paths.
  - Validation: `cargo test -p runmat-vm spawn_`, `cargo test -p runmat-vm await_rejects_non_spawn_task_operand`, `cargo fmt --all --check`.

- (pending commit) Plan 7 surface semantic instruction-window artifacts in fusion snapshots
  - Core fusion snapshot construction now accepts VM semantic instruction-window artifacts and emits explicit `SemanticWindow` nodes/decisions alongside semantic candidate artifacts.
  - Compile-time fusion-plan preview and runtime fusion-plan emission now pass semantic instruction windows through from bytecode semantic fusion metadata.
  - Added snapshot-level assertions that semantic-window artifacts are present in both no-bytecode-group and bytecode-group snapshot paths.
  - Validation: `cargo test -p runmat-core fusion::snapshot::tests::`, `cargo test -p runmat-core --test fusion_regressions`, `cargo fmt --all --check`.

- (pending commit) Plan 7 propagate semantic instruction-window counts through fusion planner metadata
  - Core fusion planner metadata now includes semantic instruction-window counts sourced from VM bytecode semantic fusion metadata.
  - Both compile-time preview (`compile_fusion_plan`) and runtime emission (`execute_outcome` with fusion snapshots enabled) now propagate this count.
  - Fusion snapshot summary diagnostics now include semantic-window counts when bytecode groups are absent.
  - Added/updated coverage in `runmat-core` fusion snapshot and fusion regression tests to assert non-zero semantic-window counts for fusible scripts.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core fusion::snapshot::tests::`, `cargo fmt --all --check`.

- (pending commit) Plan 7 persist semantic instruction-window fusion metadata on bytecode
  - VM compile now records semantic fusion instruction windows (instruction span + semantic kind hint) as explicit bytecode metadata in `SemanticFusionMetadata`.
  - Fusion-group realization now consumes that precomputed semantic window metadata instead of re-deriving windows in the accel-graph realization path.
  - Added compile-level ratchet assertions that semantic instruction-window metadata is non-empty for fusible programs and that serialized window count matches window entries.
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm semantic_candidate_instruction_windows_split_on_non_accel_ops`, `cargo test -p runmat-vm semantic_window_kind_is_not_overridden_by_graph_category`.

- (pending commit) Plan 7 make accel pre-gate assertions control-first, not display-proxy-first
  - Split builtin pre-gate rejection coverage so primary negative contract is explicit control/assertion behavior:
    - `semantic_candidate_accel_capability_gate_rejects_control_assert_builtin`
  - Kept display/stream sink classification as secondary coverage:
    - `semantic_candidate_accel_capability_gate_rejects_sink_builtins`
  - Validation: `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_control_assert_builtin`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_sink_builtins`.

- (pending commit) Plan 7 clear dropped GPU handles on `Instr::Pop` with live-handle exclusion
  - VM `Instr::Pop` now applies residency/provider-handle cleanup for dropped values, while excluding handles still referenced by live stack/var/local values to avoid premature release.
  - Removed now-dead `ops::stack::pop` helper after dispatch migration.
  - Added provider-backed runner regressions:
    - `pop_releases_stack_only_provider_handle`
    - `pop_preserves_provider_handle_when_still_live_in_vars`
    - `spawn_await_completion_releases_stack_only_provider_handle`
    - `spawn_await_completion_preserves_provider_handle_when_still_live_in_vars`
  - Validation: `cargo test -p runmat-vm pop_releases_stack_only_provider_handle`, `cargo test -p runmat-vm pop_preserves_provider_handle_when_still_live_in_vars`, `cargo test -p runmat-vm spawn_await_completion_releases_stack_only_provider_handle`, `cargo test -p runmat-vm spawn_await_completion_preserves_provider_handle_when_still_live_in_vars`, `cargo test -p runmat-vm spawn_policy_`.

- (pending commit) Plan 7 remove graph-derived shape inference from semantic fusion groups
  - VM semantic fusion-group construction now assigns `ShapeInfo::Unknown` directly instead of inferring shape from accel-graph node outputs, reducing semantic group-planning dependency on graph-derived shape artifacts.
  - Added ratchet assertion in `semantic_candidates_build_fusion_groups_from_accel_graph_nodes` that semantic groups keep `ShapeInfo::Unknown`.
  - Validation: `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-core --test fusion_regressions`.

- (pending commit) Plan 7 release provider-backed GPU handles during residency clear
  - VM residency clear paths now perform best-effort provider `free` for dropped GPU handles (in addition to residency/metadata clearing), so overwrite/drop paths release provider storage rather than only clearing fusion residency marks.
  - `clear_value_excluding` still preserves shared handles through existing incoming-handle exclusion logic.
  - Added provider-backed residency regressions in `accel/residency.rs`:
    - `clear_value_releases_provider_storage_for_dropped_handle`
    - `clear_value_excluding_preserves_shared_handles` now also asserts provider storage release/preservation behavior.
  - Validation: `cargo test -p runmat-vm clear_value_`, `cargo test -p runmat-vm cancellation_clears_gpu_residency_for_live_values`, `cargo test -p runmat-vm completion_clears_stack_only_gpu_residency`, `cargo test -p runmat-vm spawn_policy_`.

- (pending commit) Plan 7 clear transient GPU residency marks on cancellation and completion
  - VM interpreter loop now clears residency markers for live stack/variable values before returning `ExecutionCancelled`, preventing cancellation exits from leaving stale fusion residency state for GPU-handle values.
  - VM interpreter completion now clears residency marks for stack-only (non-live-var) GPU-handle values while preserving residency for handles still present in live vars.
  - Added VM regressions in `interpreter/runner.rs`:
    - `cancellation_clears_gpu_residency_for_live_values`
    - `completion_clears_stack_only_gpu_residency`
  - Validation: `cargo test -p runmat-vm cancellation_clears_gpu_residency_for_live_values`, `cargo test -p runmat-vm completion_clears_stack_only_gpu_residency`, `cargo test -p runmat-vm spawn_policy_`.

- (pending commit) Plan 1 callback ABI unresolved external identity normalization for cellfun/arrayfun
  - Runtime `cellfun` and `arrayfun` external-callback unresolved paths now emit `RunMat:UndefinedFunction` instead of a tool-specific fallback identifier or identifier-less flow error.
  - Core session outcome coverage now asserts this identifier through `ExecutionOutcome` runtime diagnostics for unresolved external `cellfun`/`arrayfun` callback paths.
  - Added runtime regressions:
    - `cellfun_external_handle_errors_as_undefined_when_unresolved`
    - `arrayfun_external_handle_errors_as_undefined_when_unresolved`
  - Validation: `cargo test -p runmat-runtime cellfun_external_handle_errors_as_undefined_when_unresolved`, `cargo test -p runmat-runtime arrayfun_external_handle_errors_as_undefined_when_unresolved`, `cargo test -p runmat-runtime arrayfun_error_without_handler_propagates_identifier`, `cargo test -p runmat-runtime cellfun_external_handle_uses_semantic_resolver`, `cargo test -p runmat-core cellfun_unresolved_external_callback_reports_undefined_function_identifier`, `cargo test -p runmat-core arrayfun_unresolved_external_callback_reports_undefined_function_identifier`.

- (pending commit) Plan 7 classify fusion-group kind from semantic windows, not accel-node category overrides
  - VM compile semantic-fusion grouping now derives `FusionKind` directly from semantic instruction-window signal kind hints (`Elementwise`/`Reduction`/`Matmul`) instead of allowing accel-graph node-category scans to override classification.
  - This keeps group-kind planning on semantic instruction facts while retaining accel-graph use for node realization and shape extraction.
  - Added compile-unit regression `semantic_window_kind_is_not_overridden_by_graph_category`.
  - Validation: `cargo test -p runmat-vm semantic_window_kind_is_not_overridden_by_graph_category`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-core --test fusion_regressions`.

- (pending commit) Plan 7 keep async/spawn semantic policy tests on identifier contracts
  - Removed remaining HIR semantic policy test assertions that matched display-message fragments after identifier checks.
  - Async/spawn policy regressions in `runmat-hir` now ratchet behavior strictly on stable identifiers (`RunMat:AwaitContextInvalid`, `RunMat:SpawnExtensionDisabled`, `RunMat:AwaitExtensionDisabled`, `RunMat:SpawnLexicalCaptureUnsupported`) rather than diagnostic wording.
  - Validation: `cargo test -p runmat-hir await_requires_async_function_or_top_level_script`, `cargo test -p runmat-hir lowering_policy_can_disable_top_level_await`, `cargo test -p runmat-hir strict_mode_disables_runmat_extension_calls`, `cargo test -p runmat-hir spawn_rejects_anonymous_function_with_lexical_capture`, `cargo test -p runmat-core execute_request_honors_top_level_await_host_policy`, `cargo test -p runmat-core --test integration test_strict_mode_rejects_runmat_extensions`, `cargo test -p runmat-core --test integration test_request_host_policy_disables_top_level_await`.

- (pending commit) Plan 7 recursive GPU gather detection/materialization for closure/output-list values
  - `runmat-runtime` dispatcher GPU recursion now includes:
    - `Value::Closure` captures
    - `Value::OutputList` entries
  - This applies to both detection (`value_contains_gpu`) and gather materialization (`gather_if_needed_async`), closing a runtime retry/gather seam where nested GPU handles in these value lanes could previously evade gather/error boundaries.
  - Added runtime regressions:
    - `value_contains_gpu_detects_nested_closure_captures`
    - `value_contains_gpu_detects_output_list_entries`
    - `gather_if_needed_reports_provider_unavailable_for_nested_output_list_gpu`
    - `gather_if_needed_reports_provider_unavailable_for_closure_capture_gpu`
  - Validation: `cargo test -p runmat-runtime value_contains_gpu_detects`, `cargo test -p runmat-runtime gather_if_needed_reports_provider_unavailable`.

- (pending commit) Plan 5 wire eval-hook lowering to shared source-context symbol discovery
  - `runmat-core` interactive `input()` eval-hook lowering in `session/run.rs` now uses shared config-owned source-context known-symbol discovery (same ownership boundary as compile/CLI/LSP paths) via `discover_known_project_symbols_from_source_name`.
  - This removes a remaining active core execution path that previously lowered without shared manifest/source-index symbol context.
  - Added core run-session regression `discover_known_project_symbols_reads_manifest_source_context`.
  - Validation: `cargo test -p runmat-core discover_known_project_symbols_reads_manifest_source_context`, `cargo test -p runmat-core source_input_path_`.

- (pending commit) Plan 7 recursive residency clearing for nested GPU-handle values
  - VM acceleration residency clearing now traverses nested runtime values instead of only top-level `Value::GpuTensor`:
    - `Cell`
    - `Struct`
    - `Object`
    - `Closure` captures
    - `OutputList`
  - VM overwrite residency clearing now preserves shared nested handles by clearing only handles that are absent from the incoming value (`clear_value_excluding`), avoiding premature de-residency when values share GPU-handle storage during replacement.
  - Fusion materialized-store writeback now also uses handle-aware exclusion clearing for globals/locals in `runmat-vm/src/accel/fusion.rs`, removing remaining top-level-only residency-clear behavior on fusion write paths.
  - Added direct fusion writeback regression `fusion_writeback_preserves_shared_gpu_handles` to ratchet shared-handle preservation at the fusion store-materialization boundary.
  - This closes a lifecycle gap where nested GPU handles could remain residency-marked after value replacement/clear paths.
  - Added `runmat-vm` unit coverage:
    - `clear_value_releases_nested_gpu_handles_in_cells`
    - `clear_value_releases_nested_gpu_handles_in_closure_captures`
    - `clear_value_excluding_preserves_shared_handles`
  - Validation: `cargo test -p runmat-vm clear_value_releases_nested_gpu_handles`.

- (pending commit) Plan 7 widen spawn GPU-handle policy boundary coverage
  - Extended VM spawn-policy dispatch tests to cover additional active-path policy boundaries:
    - nested GPU-handle capture traversal via `Value::Cell` (`spawn_policy_rejects_nested_gpu_handles_in_cell_capture`)
    - closure-capture traversal via `Value::Closure` (`spawn_policy_rejects_gpu_handles_captured_by_closure_values`)
    - explicit missing-provider diagnostic path (`spawn_policy_reports_provider_unavailable_for_gpu_handles`)
  - Both new tests assert stable runtime identifiers (`RunMat:SpawnGpuHandleUnsupported`, `RunMat:SpawnProviderUnavailable`) to keep policy-failure behavior on identifier contracts, not display text.
  - Validation: `cargo test -p runmat-vm spawn_policy_`.

- (pending commit) Plan 5 unify source-context known-symbol discovery ownership
  - Added shared config-layer helper `discover_known_project_symbols_from_source_name(source_name, cwd)` in `runmat-config` to centralize fallback policy (`None` source, discovery errors, remote/path guard fallout -> empty symbol set).
  - Switched active consumers to this shared helper instead of local duplicate wrappers:
    - `runmat-core` compile path (`session/compile.rs`)
    - CLI bytecode emission (`commands/bytecode.rs`)
    - LSP document analysis (`core/analysis.rs`)
  - Added config-layer regression `discover_known_project_symbols_from_source_name_returns_symbols_or_empty` and fixed core wildcard-import source-context fixtures to create explicit `main.m` sources for path-guard-compatible discovery coverage.
  - Validation: `cargo test -p runmat-config discover_known_project_symbols_from_source_name_returns_symbols_or_empty`, `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_project_source_index`, `cargo test -p runmat --lib commands::bytecode::tests::discover_known_project_symbols_reads_manifest_source_context`, `cargo test -p runmat-lsp source_context_symbol_discovery_reads_manifest_project_symbols`.

- (pending commit) Plan 7 assert async policy failures by semantic identifier
  - HIR lowering now emits explicit semantic error identifiers for async policy boundaries:
    - `RunMat:AwaitExtensionDisabled`
    - `RunMat:AwaitContextInvalid`
    - `RunMat:SpawnExtensionDisabled`
    - `RunMat:SpawnLexicalCaptureUnsupported`
  - Core and HIR policy regressions now assert identifier contracts directly instead of display-text fragments for strict-mode `spawn` and top-level-await host-policy rejection paths.
  - Validation: `cargo test -p runmat-hir await_requires_async_function_or_top_level_script`, `cargo test -p runmat-hir lowering_policy_can_disable_top_level_await`, `cargo test -p runmat-hir strict_mode_disables_runmat_extension_calls`, `cargo test -p runmat-hir spawn_rejects_anonymous_function_with_lexical_capture`, `cargo test -p runmat-core execute_request_honors_top_level_await_host_policy`, `cargo test -p runmat-core --test integration test_strict_mode_rejects_runmat_extensions`, `cargo test -p runmat-core --test integration test_request_host_policy_disables_top_level_await`.

- (pending commit) Plan 7 explicit provider handle spawn-concurrency policy boundary
  - `runmat-accelerate-api` now exposes provider-declared spawn-handle concurrency policy (`SpawnHandleConcurrency`) with conservative default `Reject`.
  - Production providers now explicitly declare synchronized handle-sharing semantics across spawn boundaries:
    - `InProcessProvider` -> `SynchronizedMutation`
    - `WgpuProvider` -> `SynchronizedMutation`
  - VM `Instr::Spawn` dispatch now enforces this boundary for `GpuTensor` handles captured in spawned values (including nested cell/struct/object/closure captures), returning explicit runtime diagnostics when provider policy rejects sharing or when no provider is available for a handle.
  - Added VM dispatch regressions:
    - `spawn_policy_rejects_gpu_handles_when_provider_disallows_sharing`
    - `spawn_policy_allows_gpu_handles_when_provider_declares_immutable_share`
  - Validation: `cargo test -p runmat-vm spawn_policy_`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`.

- (pending commit) Plan 5 wire manifest/source-index symbol discovery into LSP analysis
  - `runmat-lsp` document analysis now threads source-path context into lowering (`analyze_document_with_compat_and_source`) and reuses shared config-layer discovery (`discover_project_symbols_from_source_name`) before semantic lowering.
  - Native backend reanalysis and wasm document open/change paths now derive file source paths from document URIs and pass them through this shared symbol-discovery boundary.
  - Added LSP regression `source_context_symbol_discovery_reads_manifest_project_symbols` to ratchet project-symbol discovery on source-context analysis paths.
  - Validation: `cargo test -p runmat-lsp source_context_symbol_discovery_reads_manifest_project_symbols`, `cargo test -p runmat-lsp`.

- (pending commit) Plan 7 derive fusion group boundaries from semantic instruction windows
  - `runmat-vm` fusion-group construction now derives executable fusion-group instruction windows directly from semantic candidate spans plus accel-capable bytecode instructions, instead of selecting group boundaries from accel-graph node overlap first.
  - VM compile still builds accel graph artifacts for node realization and stack-layout annotation, but executable group span boundaries are now semantic/bytecode-fact-driven first.
  - Added compile-unit coverage `semantic_candidate_instruction_windows_split_on_non_accel_ops` to ratchet window splitting at non-accelerable bytecode boundaries.
  - Validation: `cargo test -p runmat-vm semantic_candidate_instruction_windows_split_on_non_accel_ops`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-core --test fusion_regressions`.

- `f0f21c5c` `RM-378: require containment in fusion-group span filter`
  - Tightened `fusion_group_within_semantic_candidate_spans` to require full instruction-span containment within a single semantic candidate span (instead of permissive overlap), aligning fusion-group semantic span filtering with the stricter containment contract used in pre-gate and node filtering.
  - Removed now-dead overlap helper after containment unification.
  - Validation: `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`, `cargo test -p runmat-vm semantic_candidates_with_partial_overlap_do_not_build_fusion_groups`.

- `758c4516` `RM-378: refine colon-name symbol guard for local paths`
  - Refined colon-style source-name guard behavior across core compile and CLI bytecode symbol discovery: colon names are blocked from project-symbol discovery only when they do not resolve to an existing local path.
  - This preserves remote/virtual source isolation while avoiding portability regressions for legitimate local colon-bearing paths.
  - Validation: `cargo test -p runmat-core compile_input_does_not_leak_local_project_symbols_for_colon_remote_name`, `cargo test -p runmat --lib commands::bytecode::tests::`.

- `f2cdc87e` `RM-378: reject colon-style remote names for symbol discovery`
  - Hardened both core compile and CLI bytecode source-context symbol discovery against colon-style remote/virtual source names (for example `remote:main.m`) so local manifest/source-index symbols are not injected into remote contexts.
  - Added regressions:
    - `compile_input_does_not_leak_local_project_symbols_for_colon_remote_name` (`runmat-core`)
    - `discover_known_project_symbols_rejects_colon_remote_name` (`runmat-cli` bytecode command)
  - Validation: `cargo test -p runmat-core compile_input_does_not_leak_local_project_symbols_for_remote_source_names`, `cargo test -p runmat-core compile_input_does_not_leak_local_project_symbols_for_colon_remote_name`, `cargo test -p runmat --lib commands::bytecode::tests::`.

- `63539f02` `RM-378: block remote source symbol bleed in core compile`
  - Hardened `runmat-core` compile-time project symbol discovery to reject path-like source names that do not map to existing local paths, preventing remote/virtual source-name contexts from inheriting local composition symbols.
  - Added integration regression: `compile_input_does_not_leak_local_project_symbols_for_remote_source_names`.
  - Validation: `cargo test -p runmat-core compile_input_does_not_leak_local_project_symbols_for_remote_source_names`, `cargo test -p runmat --lib commands::bytecode::tests::`, `cargo test -p runmat-core --test semicolon_suppression`.

- `e2f9c345` `RM-378: guard bytecode symbol discovery on local source`
  - Tightened CLI bytecode source-context symbol discovery to require an existing local source path before manifest/source-index symbol lookup, avoiding accidental local-CWD symbol injection for virtual/nonexistent source names.
  - Added `discover_known_project_symbols_requires_existing_local_source_path` coverage and hardened existing source-context fixture expectations.
  - Validation: `cargo test -p runmat --lib commands::bytecode::tests::`, `cargo test -p runmat --lib commands::script::tests::`, `cargo test -p runmat --lib commands::benchmark::tests::`, `cargo test -p runmat-core --test semicolon_suppression`.

- `f54e8e79` `RM-378: add bytecode source-context symbol tests`
  - Added CLI bytecode-command tests that ratchet manifest/source-context symbol discovery and wildcard-import lowering visibility for emitted-bytecode mode:
    - `discover_known_project_symbols_reads_manifest_source_context`
    - `emit_bytecode_uses_source_context_project_symbols`
  - Validation: `cargo test -p runmat --lib commands::bytecode::tests::`, `cargo test -p runmat --lib commands::script::tests::`, `cargo test -p runmat --lib commands::benchmark::tests::`, `cargo test -p runmat-core --test semicolon_suppression`.

- `a44db54a` `RM-378: tighten fusion-node span containment filter`
  - Tightened semantic fusion-group derivation filter so accel-graph nodes qualify only when their mapped instruction spans are fully contained by semantic candidate spans (no partial boundary overlap acceptance).
  - Added regression `semantic_candidates_with_partial_overlap_do_not_build_fusion_groups`.
  - Validation: `cargo test -p runmat-vm semantic_candidates_with_partial_overlap_do_not_build_fusion_groups`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_partial_span_overlap`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`.

- `40b00c47` `RM-378: thread project symbols into bytecode emit path`
  - CLI bytecode emission now threads source-context project symbols into lowering: `emit_bytecode` accepts optional source name and feeds `discover_project_symbols_from_source_name(...)` into `LoweringContext::with_known_project_symbols`.
  - Script `--emit-bytecode` path now passes the resolved script source name, so wildcard import/name resolution in emitted-bytecode mode uses the same manifest/source-index symbol context as normal script execution.
  - Validation: `cargo test -p runmat --lib commands::script::tests::`, `cargo test -p runmat --lib commands::benchmark::tests::`, `cargo test -p runmat-core --test semicolon_suppression`.

- `af87d84b` `RM-378: cover benchmark module entrypoint error path`
  - Extended benchmark target-resolution coverage with explicit unresolved module/function entrypoint diagnostics, mirroring script-path error-contract expectations.
  - Added `module_function_entrypoint_errors_when_module_file_missing` under benchmark command tests.
  - Validation: `cargo test -p runmat --lib commands::benchmark::tests::`.

- `2bfe761f` `RM-378: resolve benchmark targets via manifest entrypoints`
  - `runmat-cli` benchmark command now resolves input targets through shared config-layer source-input resolution (`resolve_project_source_input_from`) before file read, matching script/core manifest-driven path semantics.
  - This extends Plan 5 entrypoint-resolution wiring from `script` into another active CLI execution path (`benchmark`) so named entrypoints and module/function targets resolve consistently.
  - Added benchmark command unit coverage for:
    - named entrypoint -> path target resolution
    - relative path `.m` inference
    - module/function entrypoint -> source-root file resolution
  - Validation: `cargo test -p runmat --lib commands::benchmark::tests::`, `cargo test -p runmat-core --test semicolon_suppression`.

- `7fe6791d` `RM-378: require candidate span containment in accel gate`
  - Tightened VM compile semantic pre-gate matching from permissive span overlap to full instruction-span containment within semantic candidate source spans for accel-capable instruction qualification.
  - Added coverage: `semantic_candidate_accel_capability_gate_rejects_partial_span_overlap`, asserting boundary-only/partial overlap no longer qualifies a candidate for accel-graph construction.
  - Validation: `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_partial_span_overlap`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_non_accel_builtins`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_binary_ops`.

- `0b65a6b4` `RM-378: tighten atan2 expansion bytecode contract`
  - Strengthened VM basics multi-output argument expansion ratchet for `atan2(g())` by asserting:
    - explicit `CallBuiltinExpandMultiOutput("atan2", specs, 1)` shape with `ArgSpec { is_expand: true, expand_all: true, num_indices: 0 }`
    - absence of fixed-arity fallback shape `CallBuiltinMulti("atan2", 2, 1)` in that path.
  - This tightens remaining VM basics multi-output argument-shape coverage to a concrete semantic bytecode contract.
  - Validation: `cargo test -p runmat-vm atan2_multi_output_argument_path_unpacks_before_call`.

- `6cb30a56` `RM-378: assert chol multi-output bytecode shape`
  - Tightened VM basics `chol` multi-assign coverage by asserting semantic bytecode lowers `[R, p] = chol(A)` to explicit multi-output builtin call shape (`Instr::CallBuiltinMulti("chol", 1, 2)`) before runtime value assertions.
  - This reduces remaining multi-output argument/output-shape smoke coverage by enforcing a concrete semantic-lowering contract.
  - Validation: `cargo test -p runmat-vm chol_multiassign_reports_failure`.

- `5db3726c` `RM-378: ratchet object end-range payload in VM basics`
  - Tightened `runmat-vm` basics coverage for object end-range assignment by asserting semantic bytecode carries the rich `end` arithmetic payload shape (`end*1 - 1/2`) in `Instr::StoreSliceExpr` (`EndExpr::Sub(Mul(End, Const), Div(Const, Const))`) before runtime execution assertions.
  - This moves object range-end protocol coverage from execution-only smoke behavior to an explicit semantic bytecode contract check.
  - Validation: `cargo test -p runmat-vm object_range_end_assignment_accepts_rich_end_expression_payload`.

- `26450a6c` `RM-378: widen non-accel builtin gate coverage`
  - Expanded `runmat-vm` accel-capability pre-gate negative coverage from a single sentinel builtin to multiple explicit non-accelerable builtins:
    - `assert` (control/assertion)
    - `disp` (display sink)
    - `fprintf` (streaming sink)
  - Renamed gate test to `semantic_candidate_accel_capability_gate_rejects_non_accel_builtins` and now ratchets each builtin explicitly.
  - Validation: `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_non_accel_builtins`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_reduction_builtin`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_binary_ops`.

- `af19ee9d` `RM-378: use assert for sink-builtin accel gate test`
  - Updated `runmat-vm` pre-gate rejection coverage to use control/assertion builtin semantics (`assert`) instead of display-side proxy semantics (`disp`) for the non-accelerable builtin overlap path.
  - This keeps the Plan 7 accel-capability gate negative ratchet aligned to proper assertion/control behavior at the builtin boundary.
  - Validation: `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_non_accel_builtins`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_reduction_builtin`.

- `40610f2a` `RM-378: cover builtin accel-capability gating`
  - Added `runmat-vm` builtin-path ratchets for semantic-candidate accel-capability pre-gate classification:
    - `semantic_candidate_accel_capability_gate_accepts_reduction_builtin` (`sum`)
    - `semantic_candidate_accel_capability_gate_rejects_non_accel_builtins` (`assert`, `disp`, `fprintf`)
  - This hardens Plan 7 pre-gate behavior across builtin-call bytecode paths, not only primitive arithmetic/logical opcodes.
  - Validation: `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_reduction_builtin`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_non_accel_builtins`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_binary_ops`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `f08b9e0d` `RM-378: cover logical-only candidate pre-gate`
  - Added compile-level `runmat-vm` regression:
    - `primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops`
  - The test asserts semantic candidate groups can exist while accel graph and executable fusion groups remain omitted when candidate overlap maps only to non-accelerable logical bytecode operations.
  - Validation: `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops`, `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `06789be1` `RM-378: gate accel graph on accel-capable bytecode`
  - VM compile now adds a semantic-candidate pre-gate that requires candidate-overlapping accel-capable bytecode instructions before building accel graph artifacts.
  - When semantic candidate spans map only to non-accelerable instructions (for example logical ops), VM compile now skips accel-graph construction early and leaves executable fusion groups empty.
  - Added `runmat-vm` unit ratchets:
    - `semantic_candidate_accel_capability_gate_rejects_logical_ops`
    - `semantic_candidate_accel_capability_gate_accepts_binary_ops`
  - Validation: `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_rejects_logical_ops`, `cargo test -p runmat-vm semantic_candidate_accel_capability_gate_accepts_binary_ops`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `ec74ba1e` `RM-378: harden source-input resolver tests`
  - Expanded `runmat-config` integration coverage for `resolve_project_source_input_from` contracts:
    - pass-through behavior for non-entrypoint simple names
    - explicit named-entrypoint resolution error surfacing (`ResolveProjectSourceInputError::EntrypointResolve`) for invalid module/function targets.
  - This hardens the shared path-resolution boundary now consumed by both CLI and core path-source execution.
  - Validation: `cargo test -p runmat-config --test project_manifest resolve_project_source_input_from_returns_plain_candidate_when_name_is_not_entrypoint`, `cargo test -p runmat-config --test project_manifest resolve_project_source_input_from_reports_named_entrypoint_resolution_errors`, `cargo test -p runmat-core source_input_path_errors_for_invalid_named_entrypoint_target`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `1410839c` `RM-378: dedupe discovered composition loading`
  - Refactored `runmat-config` to share one internal discovered-composition loader (`discover_project_composition_from`) across:
    - `resolve_named_entrypoint_from`
    - `discover_project_symbols_from`
  - This removes duplicated manifest-discovery, composition-graph-load, and root-package verification logic in those APIs while preserving behavior and diagnostics.
  - Validation: `cargo test -p runmat-config --test project_manifest discover_project_symbols_includes_dependency_alias_qualified_names`, `cargo test -p runmat-config --test project_manifest resolve_named_entrypoint_from_discovers_and_resolves`, `cargo test -p runmat-config --test project_manifest resolve_named_entrypoint_from_reports_resolution_errors`, `cargo test -p runmat --lib resolve_script_input_`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `45570878` `RM-378: centralize source-input path resolution`
  - Added shared config-layer source-input path resolver:
    - `runmat_config::resolve_project_source_input_from(cwd, source_input)`
  - The shared resolver now centralizes:
    - direct file/path acceptance
    - optional `.m` inference for extensionless path-like inputs
    - single-segment named entrypoint fallback through composition discovery.
  - Migrated both consumers to the shared resolver:
    - CLI `resolve_script_input` in `runmat-cli`
    - core ABI path-source resolution (`SourceInput::Path`) in `runmat-core`.
  - Added config integration coverage:
    - `resolve_project_source_input_from_infers_m_extension`
    - `resolve_project_source_input_from_resolves_named_entrypoint`
  - Validation: `cargo test -p runmat-config --test project_manifest resolve_project_source_input_from_infers_m_extension`, `cargo test -p runmat-config --test project_manifest resolve_project_source_input_from_resolves_named_entrypoint`, `cargo test -p runmat --lib resolve_script_input_`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `26b67af5` `RM-378: centralize symbol discovery start path`
  - Added shared config-layer source-name discovery helper: `runmat_config::discover_project_symbols_from_source_name(source_name, cwd)`.
  - This centralizes source-name -> project-discovery start-path derivation in `runmat-config` and removes the remaining duplicate heuristic logic from `runmat-core` compile.
  - Added config integration coverage:
    - `discover_project_symbols_from_source_name_uses_cwd_for_plain_name`
  - Validation: `cargo test -p runmat-config --test project_manifest discover_project_symbols_from_source_name_uses_cwd_for_plain_name`, `cargo test -p runmat-config --test project_manifest discover_project_symbols_includes_dependency_alias_qualified_names`, `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_dependency_alias`, `cargo test -p runmat-core compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `dfcc7f65` `RM-378: share composition symbol discovery API`
  - Added shared config-layer composition symbol discovery API: `runmat_config::discover_project_symbols_from`.
  - The shared API now centralizes:
    - project manifest discovery
    - composition graph loading
    - root dependency alias mapping
    - symbol set construction (`raw`, `package-qualified`, and `dependency-alias-qualified` names).
  - `runmat-core` compile path now consumes this shared API for wildcard-import known-project symbol discovery instead of owning a duplicate composition/source-index traversal loop.
  - Added config integration coverage:
    - `discover_project_symbols_includes_dependency_alias_qualified_names`
  - Validation: `cargo test -p runmat-config --test project_manifest discover_project_symbols_includes_dependency_alias_qualified_names`, `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_dependency_alias`, `cargo test -p runmat-core compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `git diff --check`.

- `341e801c` `RM-378: tighten alias wildcard handle assertion`
  - Tightened dependency-alias wildcard function-handle regression to assert exact alias-qualified lowering identity (`CreateExternalFunctionHandle("statsdep.summarize")`) instead of a substring match.
  - This keeps Plan 5 wildcard-import/alias function-handle evidence on a strict semantic identity contract.
  - Validation: `cargo test -p runmat-core compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`, `cargo test -p runmat-core --test semicolon_suppression`.

- `9642000f` `RM-378: cover alias wildcard function handles`
  - Added `runmat-core` integration coverage for dependency-alias wildcard imports feeding function-handle lowering:
    - `compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`
  - The regression asserts `import statsdep.*; f = @summarize;` compiles with an explicit external function-handle bytecode creation path (`Instr::CreateExternalFunctionHandle`) for the imported dependency symbol.
  - Validation: `cargo test -p runmat-core compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`, `cargo test -p runmat-core --test semicolon_suppression`.

- `7ed84fa4` `RM-378: drop accel graph when no executable groups`
  - VM bytecode compile now drops accel-graph artifacts when semantic candidate groups exist but no executable semantic-mapped fusion groups survive filtering.
  - This reduces residual bytecode-graph coupling in transitional semantic-candidate/no-executable-group cases by avoiding retention of unused graph artifacts.
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm semantic_candidates_without_overlap_do_not_build_fusion_groups`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `9fd5f027` `RM-378: resolve wildcard imports via dependency aliases`
  - Project composition symbol discovery in `runmat-core` now includes dependency alias-qualified symbols from root `runmat.toml` dependency mapping (`alias -> package`), in addition to package-qualified and raw source-index symbols.
  - This enables wildcard imports like `import statsdep.*; y = summarize(1);` to resolve as package-function calls through manifest dependency aliases.
  - Added integration regression coverage:
    - `compile_input_resolves_wildcard_import_from_dependency_alias`
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_project_source_index`, `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_dependency_alias`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat --lib resolve_script_input_`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `f3dd3c4b` `RM-378: thread project source symbols into import resolution`
  - `runmat-core` compile path now discovers project composition/source-index symbols from the active source context and passes them into HIR lowering.
  - `runmat-hir` wildcard import resolution now accepts project source-index candidates (not only builtins) for both:
    - direct call target resolution
    - function-handle target resolution
  - Added integration regression coverage:
    - `compile_input_resolves_wildcard_import_from_project_source_index`
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_project_source_index`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat --lib resolve_script_input_`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `eaadfab5` `RM-378: infer .m for unresolved source paths`
  - Core `SourceInput::Path` resolution now infers `.m` for unresolved file-style path inputs (e.g. `src/main` -> `src/main.m`) before entrypoint-name fallback.
  - CLI script path resolution now applies the same `.m` inference before manifest entrypoint-name fallback.
  - Added regression coverage:
    - `runmat-core`: `source_input_path_infers_m_extension_for_relative_path`
    - `runmat-cli`: `resolve_script_input_infers_m_extension_for_relative_path`
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat --lib resolve_script_input_`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `d4afa838` `RM-378: remove legacy VarId note reference`
  - Removed the remaining `VarId` mention under `crates/` notes to keep the legacy-path evidence grep unambiguous for active production surfaces.
  - Validation evidence check: `rg -n "\\bVarId\\b|compile_legacy|LegacyUserFunction|runmat_vm::execute|\\bHirProgram\\b" crates` now returns no matches.

- `bf711150` `RM-378: carry semantic await-site metadata`
  - Bytecode semantic async metadata now carries explicit MIR await-site inventory (`mir_await_site_count`, `mir_await_sites`) alongside spawn-site metadata.
  - VM compile now derives await sites from MIR `Await` terminators scoped to the active entrypoint target, mirroring spawn-site scoping rules.
  - Added regression coverage:
    - `primary_compile_records_semantic_await_site_metadata`
    - `primary_compile_scopes_await_site_metadata_to_entrypoint_target`
    - strengthened `primary_compile_records_semantic_spawn_site_metadata` with await-site absence assertions for spawn-only programs
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-vm primary_compile_records_semantic_spawn_site_metadata`, `cargo test -p runmat-vm primary_compile_records_semantic_await_site_metadata`, `cargo test -p runmat-vm primary_compile_scopes_await_site_metadata_to_entrypoint_target`, `cargo test -p runmat-vm primary_compile_emits_explicit_spawn_instruction`, `cargo test -p runmat-vm primary_compile_interprets_async_call_and_await_via_semantic_value_lane`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `2c12e33d` `RM-378: emit explicit await bytecode boundary`
  - MIR `Await` terminators now lower to an explicit `Instr::Await` bytecode opcode instead of implicitly relying on operand-lane behavior.
  - VM dispatch/runner now handle `Instr::Await` explicitly, and Turbine marks `Instr::Await` as interpreter-only.
  - Expanded regression coverage in `primary_compile_emits_explicit_spawn_instruction` to assert both explicit `Spawn` and `Await` opcode emission.
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-vm primary_compile_emits_explicit_spawn_instruction`, `cargo test -p runmat-vm primary_compile_interprets_async_call_and_await_via_semantic_value_lane`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm semantic_candidates_without_overlap_do_not_build_fusion_groups`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `7d1478f3` `RM-378: emit explicit spawn bytecode boundary`
  - MIR `Spawn` now lowers to an explicit `Instr::Spawn` bytecode opcode instead of silently aliasing the future operand lane.
  - VM dispatch/runner now handle `Instr::Spawn` explicitly; Turbine marks it interpreter-only.
  - Added regression coverage: `primary_compile_emits_explicit_spawn_instruction`.
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-vm primary_compile_emits_explicit_spawn_instruction`, `cargo test -p runmat-vm primary_compile_interprets_async_call_and_await_via_semantic_value_lane`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm semantic_candidates_without_overlap_do_not_build_fusion_groups`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `b44e43bf` `RM-378: remove bytecode fusion fallback from semantic candidates`
  - VM compile no longer falls back to `detect_fusion_groups()` when semantic candidate-derived fusion groups are empty.
  - This tightens executable fusion-group construction to semantic candidate evidence only.
  - Added regression coverage: `semantic_candidates_without_overlap_do_not_build_fusion_groups`.
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm semantic_candidates_without_overlap_do_not_build_fusion_groups`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `e3ee0fb6` `RM-378: derive fusion groups from semantic candidates`
  - VM compile now derives executable fusion groups from semantic candidate source spans mapped onto accel-graph nodes (`derive_semantic_fusion_groups_from_candidates`) before semantic span retention.
  - Added regression coverage: `semantic_candidates_build_fusion_groups_from_accel_graph_nodes`.
  - Validation: `cargo fmt --all --check`, `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

## Plan Status Snapshot

- Plan 0: semantic HIR type model in place.
- Plan 1: semantic lowering in place.
- Plan 2: MIR + analysis layer in place.
- Plan 3: downstream consumer migration in progress.
- Plan 4: MATLAB core semantic coverage in progress.
- Plan 5: manifest/composition closeout audit still needed.
- Plan 6: nominal class/builtin metadata unification in progress.
- Plan 7: semantic-fact-driven accel/fusion closeout audit still needed.

## Recent Landed Slices

- (pending commit) Plan 7 require single-semantic-candidate span coverage per executable bytecode fusion group
  - `runmat-vm` semantic-span fusion-group filter now requires a bytecode fusion group's full instruction-span range to be covered by one semantic candidate span, rather than allowing coverage by unioning multiple candidate spans.
  - This removes residual cross-candidate bytecode grouping leakage and tightens executable fusion group retention to single semantic candidate regions.
  - Added `runmat-vm` regression coverage:
    - `fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`
  - Validation: `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 strict semantic-span coverage filter for executable bytecode fusion groups
  - `runmat-vm` bytecode fusion-group filtering now requires full instruction-span coverage by semantic candidate source spans (not just any-overlap).
  - This further constrains executable bytecode fusion groups to semantic candidate regions and reduces residual bytecode-graph-driven latitude in candidate retention.
  - Added `runmat-vm` regression coverage:
    - `fusion_group_semantic_span_filter_requires_full_group_coverage`
  - Validation: `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 surface semantic candidate source spans in fusion snapshot diagnostics
  - `runmat-core` fusion snapshot semantic-candidate node labels and decision reasons now include semantic candidate source-span ranges.
  - This makes semantic candidate evidence traceable in planner artifacts to concrete source regions used for executable-group semantic alignment.
  - Updated fusion snapshot unit coverage to assert source-span propagation in semantic-candidate labels/reasons.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --lib fusion::snapshot::tests::semantic_candidate_groups_emit_nodes_with_bytecode_groups`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic source-span alignment for executable bytecode fusion groups
  - `runmat-vm` semantic fusion candidate groups now carry merged source spans for each MIR candidate run.
  - VM compile now filters detected bytecode fusion groups against semantic candidate source-span overlap (via instruction-source spans), so executable fusion groups are retained only when they intersect semantic candidate regions.
  - This tightens executable fusion candidate construction toward semantic evidence rather than bytecode graph shape alone.
  - Added `runmat-vm` regression coverage:
    - `fusion_group_semantic_overlap_uses_source_spans`
    - strengthened `primary_compile_records_semantic_fusion_metadata` with source-span assertions
  - Validation: `cargo test -p runmat-vm fusion_group_semantic_overlap_uses_source_spans`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-vm primary_compile_scopes_semantic_fusion_metadata_to_entrypoint_target`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 scope fusion planner MIR local-fact counts to active entrypoint
  - `runmat-core` fusion snapshot planner metadata now counts MIR local facts only for the active entrypoint target function in both:
    - preview path (`compile_fusion_plan`)
    - runtime emission path (`set_emit_fusion_plan(true)` / execute outcome)
  - This prevents non-entrypoint helper-function local facts from inflating active-entrypoint planner metadata.
  - Added `runmat-core` regression coverage:
    - `compile_fusion_plan_scopes_local_fact_count_to_entrypoint`
  - Validation: `cargo test -p runmat-core compile_fusion_plan_scopes_local_fact_count_to_entrypoint`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo test -p runmat-vm primary_compile_scopes_semantic_fusion_metadata_to_entrypoint_target`, `cargo test -p runmat-vm primary_compile_scopes_spawn_site_metadata_to_entrypoint_target`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 scope semantic spawn-site metadata to compiled entrypoint target
  - `runmat-vm` bytecode compile now derives semantic async/spawn metadata from the selected entrypoint target MIR body instead of aggregating across every MIR body.
  - This prevents helper-function spawn sites from being attributed to active entrypoint bytecode artifacts and keeps async boundary metadata aligned to the active execution path.
  - Added `runmat-vm` regression coverage:
    - `primary_compile_scopes_spawn_site_metadata_to_entrypoint_target`
  - Validation: `cargo test -p runmat-vm primary_compile_scopes_spawn_site_metadata_to_entrypoint_target`, `cargo test -p runmat-vm primary_compile_records_semantic_spawn_site_metadata`, `cargo test -p runmat-vm --test basics`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 scope semantic fusion metadata to compiled entrypoint target
  - `runmat-vm` bytecode compile now derives semantic fusion signal/candidate metadata from the selected entrypoint target MIR body instead of aggregating across every MIR body in the assembly.
  - This prevents non-entrypoint helper functions from falsely opening fusion gating for the active executable bytecode path.
  - Added `runmat-vm` regression coverage:
    - `primary_compile_scopes_semantic_fusion_metadata_to_entrypoint_target`
  - Validation: `cargo test -p runmat-vm primary_compile_scopes_semantic_fusion_metadata_to_entrypoint_target`, `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-vm --test basics`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 candidate-group-gated accel graph construction in VM compile
  - `runmat-vm` bytecode compile now omits accel graph construction unless semantic fusion candidate groups are present (`mir_fusion_candidate_group_count > 0`).
  - This removes the remaining `signals>0 but no semantic candidate groups` path that still built accel graph artifacts, tightening compile-time fusion artifact creation to semantic candidate evidence.
  - Added `runmat-vm` regression coverage:
    - `primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group`
  - Validation: `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group`, `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-vm --test basics`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic-gated fusion snapshot decisions for bytecode-group mismatch
  - `runmat-core` fusion snapshot decisions for bytecode fusion groups are now gated by semantic candidate-group evidence from planner metadata.
  - When bytecode fusion groups are present but `mir_fusion_candidate_group_count == 0`, snapshot decisions are now marked `fused: false` with explicit semantic-gating reason (`semantic-candidate-groups=0`) instead of reporting unconditional fused status.
  - Added `runmat-core` regression coverage:
    - `bytecode_groups_without_semantic_candidates_are_marked_non_fused`
  - Validation: `cargo test -p runmat-core bytecode_groups_without_semantic_candidates_are_marked_non_fused`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic gating at fusion plan preparation boundary
  - `runmat-accelerate::prepare_fusion_plan` now requires semantic candidate-group evidence (`semantic_candidate_group_count > 0`) before constructing executable fusion plans from bytecode fusion groups.
  - This hardens the runtime fusion activation boundary against bytecode-only planning artifacts and keeps semantic candidate products as the gating source-of-truth for executable fusion-plan preparation.
  - Added `runmat-accelerate` unit ratchets:
    - `prepare_fusion_plan_requires_semantic_candidate_groups`
    - `prepare_fusion_plan_allows_semantic_gated_groups`
  - Validation: `cargo test -p runmat-accelerate prepare_fusion_plan_requires_semantic_candidate_groups`, `cargo test -p runmat-accelerate prepare_fusion_plan_allows_semantic_gated_groups`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic gating for bytecode fusion-group detection
  - `runmat-vm` bytecode compile now derives semantic fusion metadata before accel-graph group detection and gates bytecode `detect_fusion_groups()` behind semantic candidate presence.
  - When `mir_fusion_candidate_group_count == 0`, executable bytecode fusion groups are now forced empty, making semantic MIR products the prerequisite source for bytecode fusion candidate construction.
  - Added native-accel regression coverage asserting semantic-gated behavior (`no semantic candidate groups => no bytecode fusion groups`).
  - Validation: `cargo test -p runmat-vm primary_compile_semantically_gates_bytecode_fusion_groups`, `cargo test -p runmat-vm primary_compile_records_semantic_spawn_site_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 explicit semantic spawn-site metadata boundary
  - `runmat-vm::Bytecode` now carries `semantic_async_metadata` with MIR-derived spawn-site inventory:
    - `mir_spawn_site_count`
    - `mir_spawn_sites` containing `function`, `block`, and `stmt_index`
  - VM bytecode compile now derives spawn-site metadata directly from MIR statement forms (`MirRvalue::Spawn`) using deterministic function-id traversal.
  - Interpreter state setup now emits explicit runtime debug signal when compiled bytecode carries spawn sites, replacing silent transitional behavior with explicit semantic boundary telemetry.
  - Added `runmat-vm` compile regression coverage asserting spawn-site metadata presence and consistency.
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_spawn_site_metadata`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 add MIR-localized semantic fusion candidate metadata
  - `runmat-vm` semantic fusion candidate artifacts now carry MIR location metadata per candidate run:
    - `function` (`FunctionId`)
    - `block` (`BasicBlockId`)
    - `stmt_start` / `stmt_end` run bounds
  - Candidate derivation now iterates MIR bodies in deterministic function-id order and records concrete statement-run spans for each semantic fusion candidate group.
  - `runmat-core` fusion snapshot semantic-candidate node labels now include MIR location metadata, making semantic candidate evidence traceable to specific MIR regions for follow-on planner migration.
  - Added regression checks that candidate groups carry non-empty statement spans and updated snapshot tests to assert mixed bytecode + semantic candidate artifact emission with MIR-localized metadata.
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core semantic_candidate_groups_emit_nodes_with_bytecode_groups`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 thread semantic candidate evidence into fusion plan preparation boundary
  - Extended `runmat-accelerate::prepare_fusion_plan` to accept semantic candidate-group count from VM bytecode semantic metadata.
  - Interpreter state setup now passes `bytecode.semantic_fusion_metadata.mir_fusion_candidate_group_count` into fusion-plan preparation, so the execution setup boundary has explicit semantic-candidate awareness even when executable bytecode groups are empty.
  - Added transition debug telemetry when semantic candidates exist but executable bytecode fusion groups are absent, making semantic-vs-bytecode planning gaps explicit during runtime plan preparation.
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic candidate artifacts always surfaced in fusion snapshots
  - `runmat-core` fusion snapshot generation now appends semantic candidate nodes/decisions even when bytecode accel fusion groups are present, instead of limiting semantic candidate artifacts to the bytecode-empty fallback path.
  - Snapshot decisions now explicitly annotate both semantic signal strength and bytecode-group presence, improving semantic-vs-bytecode planner coverage visibility while runtime fusion execution remains unchanged.
  - Added core unit regression coverage for mixed snapshots (bytecode group + semantic candidate group both present).
  - Validation: `cargo test -p runmat-core semantic_candidate_groups_emit_nodes_with_bytecode_groups`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic fusion candidate-group artifacts and snapshot nodes
  - `runmat-vm` semantic fusion metadata now carries explicit candidate-group artifacts (`mir_fusion_candidate_groups`) in addition to aggregate counts.
  - Candidate groups are derived from contiguous MIR semantic fusion-signal runs during bytecode compile and recorded on the bytecode artifact boundary.
  - `runmat-core` fusion snapshot generation now consumes semantic candidate groups and emits explicit semantic-candidate nodes/decisions when bytecode accel fusion groups are empty, rather than only emitting count summaries.
  - This moves fusion-candidate construction visibility further toward semantic MIR products and reduces dependence on bytecode accel-group reconstruction for planner diagnostics.
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core semantic_candidate_groups_emit_nodes_without_bytecode_groups`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 reuse prepared MIR artifacts for fusion planning metadata
  - `RunMatSession::compile_input` now carries the lowered MIR assembly in `PreparedExecution`, making MIR an explicit prepared artifact instead of a compile-only temporary.
  - Fusion-plan preview (`compile_fusion_plan`) and runtime fusion snapshot emission now reuse prepared MIR directly for analysis fact counts rather than lowering MIR again from HIR.
  - This tightens semantic ownership boundaries in core session flow and removes duplicate MIR lowering work on fusion metadata paths.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 semantic fusion summary decoupled from accel-graph presence
  - `runmat-core` fusion snapshot generation no longer requires an accel graph artifact to emit semantic-candidate summaries.
  - When bytecode fusion groups are empty but semantic MIR candidate evidence exists, summary decisions now include explicit accel-graph presence state (`present`/`missing`) in planner reasoning.
  - Added unit regression coverage in core fusion snapshot plumbing for semantic summary emission with a missing accel graph.
  - Validation: `cargo test -p runmat-core semantic_candidate_summary_emits_without_accel_graph`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 bytecode-owned semantic fusion metadata
  - `runmat-vm::Bytecode` now carries explicit semantic fusion metadata (`mir_fusion_signal_count`, `mir_fusion_candidate_group_count`) as part of the compiled artifact boundary under native acceleration.
  - VM bytecode compile now derives those semantic counts directly from MIR and stores them on the bytecode product alongside accel graph/groups.
  - Core fusion snapshot planner metadata now consumes the bytecode-owned semantic fusion metadata in both preview (`compile_fusion_plan`) and runtime fusion snapshot emission paths, instead of recomputing MIR fusion signal/candidate counts in core fusion snapshot plumbing.
  - Added `runmat-vm` compile regression coverage asserting non-zero semantic fusion metadata for representative fusible arithmetic chains.
  - Validation: `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic-candidate snapshot emission when bytecode groups are empty
  - Fusion snapshot generation now emits a semantic candidate summary snapshot when bytecode fusion groups are empty but MIR semantic fusion signals/candidate groups are present.
  - This prevents semantic planning evidence from being hidden behind bytecode-group presence and keeps planner telemetry visible in transitional no-group cases.
  - Added `runmat-core` regression coverage for a scalar semantic-op chain that produces semantic candidate metadata even when no bytecode fusion group is detected.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 MIR fusion candidate-group metadata bridge
  - Extended fusion planner metadata with `mir_fusion_candidate_group_count`.
  - Added MIR-side fusion candidate-group estimator in core fusion snapshot plumbing:
    - counts contiguous MIR statement runs with fusion-relevant semantic operations
    - uses a minimum run length of 2 operations as candidate-group threshold
  - Preview and runtime fusion snapshots now emit both:
    - `mir_fusion_signal_count`
    - `mir_fusion_candidate_group_count`
  - Updated fusion regression tests to assert non-zero candidate-group counts for representative fusible scripts.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 semantic fusion-signal metadata bridge
  - `FusionPlannerMetadata` now carries `mir_fusion_signal_count` in addition to MIR local fact/diagnostic counts.
  - Added `semantic_fusion_signal_count(&MirAssembly)` bridge in core fusion snapshot plumbing that counts MIR-level fusion-relevant semantic operations (unary/binary ops and builtin calls with elementwise/reduction/linear-algebra/shape-transform semantic kinds).
  - `compile_fusion_plan` preview and runtime fusion snapshot emission now record the MIR fusion signal count in planner metadata.
  - Updated `runmat-core` fusion regression coverage to assert non-zero MIR fusion signal counts for representative fusible scripts.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 shared discovered-entrypoint resolver API across config/CLI/core
  - Added `runmat-config::resolve_named_entrypoint_from(start, entrypoint_name)` to centralize:
    - manifest discovery
    - composition graph loading
    - root package selection
    - named entrypoint resolution
  - Added typed discovery/resolve diagnostics:
    - `DiscoverProjectEntrypointError`
    - `DiscoveredProjectEntrypoint`
  - Migrated both consumers to the shared API:
    - CLI script path resolution in `runmat-cli`
    - core request path source resolution in `runmat-core`
  - Added config tests covering discovered resolver success and explicit resolution errors.
  - Validation: `cargo test -p runmat-config`, `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 core ABI path-input manifest entrypoint resolution
  - `runmat-core` `ExecutionRequest` path-source loading now resolves simple named path inputs through discovered project composition/entrypoint metadata before file read.
  - `source_input_text(SourceInput::Path(..))` now:
    - discovers `runmat.toml`
    - builds project composition graph
    - resolves named entrypoints through `resolve_project_entrypoint`
    - reads resolved source file path for execution
  - Added `runmat-core` unit coverage for:
    - named manifest entrypoint path resolution in core request path handling
    - explicit error surfacing for invalid module/function entrypoint targets
  - Validation: `cargo test -p runmat-core source_input_path_`, `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 CLI composition-graph integration for entrypoint resolution
  - `runmat-cli` script target resolution now builds a composition graph from discovered `runmat.toml` (`build_project_composition_graph`) and resolves named entrypoints from the root package manifest/root source context.
  - This moves active CLI entrypoint selection off direct single-manifest loading and onto the new composition artifact boundary.
  - Validation: `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 composition graph foundation in `runmat-config`
  - Added `build_project_composition_graph(root_manifest_path)` with typed composition artifacts:
    - `ProjectCompositionGraph`
    - `ProjectCompositionPackage`
    - `ProjectCompositionError`
  - Composition loading now covers:
    - root manifest load
    - local path dependency manifest discovery/load
    - per-package source index construction for root + dependencies
    - explicit diagnostics for missing dependency manifest, duplicate package names, and dependency cycles
  - Added `runmat-config` integration coverage for:
    - successful root + local dependency graph construction
    - missing dependency-manifest diagnostics
    - duplicate package-name diagnostics
  - Validation: `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 module/function entrypoint resolution via source index
  - `runmat-config::resolve_project_entrypoint` module/function targets now resolve against `build_project_source_index` outputs instead of using a dotted-module file-path heuristic.
  - Module/function target resolution now supports MATLAB layout-derived qualified names, including class-folder methods (`+pkg/@ClassName/method.m`) represented as `module = "pkg.ClassName"` + `function = "method"`.
  - Added resolver coverage for:
    - class-folder module/function target resolution
    - source-index failure surfacing from entrypoint resolution
  - Validation: `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 resolver ownership consolidation
  - Added `runmat-config::resolve_project_entrypoint(project_root, manifest, entrypoint_name)` as the shared entrypoint resolution boundary, with typed resolved target metadata:
    - `ResolvedProjectEntrypoint`
    - `ResolvedEntrypointTarget::{Path, ModuleFunction}`
    - `ProjectEntrypointResolveError`
  - `runmat-cli` script target resolution now delegates named manifest entrypoint selection to the shared config resolver instead of duplicating path/module resolution logic in CLI command code.
  - Added `runmat-config` integration tests covering resolved path targets, resolved module/function targets, and missing-module explicit errors.
  - Validation: `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- `a5fac002` RM-378: trim builtin compatibility
  - Removed dead `BuiltinCompatibility::RunMatExtended` variant to keep policy surfaces behavior-backed.
  - Validation: `cargo test -p runmat-builtins`, `cargo fmt --all --check`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- `e2095950` RM-378: migrate gc stress tests
  - Replaced legacy stress-test usage of removed APIs (`HirProgram`, `runmat_vm::execute`) with semantic pipeline (`lowering.assembly -> runmat_mir::lower_assembly -> runmat_vm::compile -> runmat_vm::interpret`).
  - Added `runmat-mir` dev dependency for `runmat-gc`.
  - Validation: `cargo test -p runmat-gc --tests` green.

- (pending commit) Plan 5 foundation in `runmat-config`
  - Added dedicated project-manifest support for `runmat.toml` separate from runtime `.runmat.*` config:
    - `ProjectManifest` shape (`package`, `sources`, `dependencies`, `entrypoints`)
    - upward discovery (`discover_project_manifest_from`)
    - TOML parsing/loading (`parse_project_manifest_toml`, `load_project_manifest`)
    - validation rules for non-empty package/source roots, relative roots/paths, existing source/dependency paths, entrypoint target forms, duplicate entrypoint names, and optional `.m` inference for path entrypoints.
  - Added `runmat-config` tests covering parse/load/validation and manifest discovery walking.
  - Validation: `cargo test -p runmat-config`, `cargo fmt --all --check`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 source-index layout scanning
  - Added `build_project_source_index(project_root, manifest)` in `runmat-config` with typed index outputs for discovered files plus explicit MATLAB layout buckets (`package_dirs`, `class_dirs`, `private_dirs`).
  - Scanner now classifies `.m` files across source roots with qualified-name derivation that accounts for `+pkg`, `@ClassName`, and `private` directories.
  - Added tests covering mixed layout discovery (`main`, package functions, class methods, private helpers, nested utility modules) and missing-source-root error paths.
  - Validation: `cargo test -p runmat-config`, `cargo test -p runmat --lib`, `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 5 CLI integration step
  - `runmat-cli` script execution now resolves named entrypoints from discovered `runmat.toml` manifests when the provided run target is not an existing file.
  - Path entrypoints are resolved with optional `.m` inference; module/function entrypoints now resolve to module files under configured source roots (`module` dotted path -> `<root>/<module path>.m`) with explicit errors when unresolved.
  - Added `runmat-cli` unit coverage for path-target resolution, module/function-target resolution, and unresolved-module diagnostics.
  - Validation: `cargo test -p runmat --lib`, `cargo fmt --all --check`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 visibility bridge in fusion snapshots
  - `FusionPlanSnapshot` now carries planner metadata (`source`, `mir_local_fact_count`, `mir_diagnostic_count`).
  - `RunMatSession::compile_fusion_plan` and runtime execution path (`execute_internal` when fusion snapshot emission is enabled) now compute MIR + `AnalysisStore` and record analysis fact counts into fusion snapshot metadata, making semantic-analysis availability explicit in the fusion planning surface.
  - Fusion candidate construction itself is still bytecode-graph-driven; this slice improves observability and contract clarity while that migration remains open.
  - Validation: `cargo test -p runmat --lib`, `cargo test -p runmat-config`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 metadata regression coverage
  - Added `runmat-core` integration coverage asserting fusion planner metadata in both preview path (`compile_fusion_plan`) and runtime outcome path (`set_emit_fusion_plan(true)` + `execute_outcome`).
  - New tests live in `crates/runmat-core/tests/fusion_regressions.rs` and assert semantic planner source tags plus non-zero MIR local fact counts.
  - Validation: `cargo test -p runmat-core --test fusion_regressions`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

- (pending commit) VM test identifier-contract ratchet (display-string debt reduction)
  - Updated `crates/runmat-vm/tests/functions.rs` to assert stable runtime identifiers instead of message substrings in:
    - `expansion_on_non_cell_errors` -> `RunMat:ExpandError`
    - `mixed_range_end_assign_shape_mismatch_error` -> `RunMat:ShapeMismatch`
  - Rationale: these tests validate runtime behavior contracts, so identifier checks are the stable assertion surface; display text remains non-contract UX.
  - Validation: `cargo test -p runmat-vm expansion_on_non_cell_errors -- --nocapture`, `cargo test -p runmat-vm mixed_range_end_assign_shape_mismatch_error -- --nocapture`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 compile-time fusion mapping heuristic reduction
  - Removed compile-time semantic-window node mapping fallbacks for:
    - small disjoint graph/window gaps
    - small partial-overlap boundary drift
    - bounded covering-node span coupling
    in `crates/runmat-vm/src/bytecode/compile.rs`.
  - Compile mapping now accepts only contained-window spans; disjoint/partial-overlap/covering-span reconciliation remains runtime-owned via fusion plan sanitization in `runmat-accelerate`.
  - Updated compile tests to codify stricter compile behavior:
    - renamed `semantic_windows_map_accel_nodes_with_small_disjoint_gap`
    - now `semantic_windows_reject_disjoint_gap_at_compile_mapping_stage` asserting no compile-time mapping for disjoint spans.
    - renamed `semantic_windows_map_accel_nodes_with_small_boundary_shift_overlap`
    - now `semantic_windows_reject_partial_overlap_at_compile_mapping_stage` asserting no compile-time mapping for partial-overlap spans.
    - renamed `semantic_windows_map_accel_nodes_that_cover_window_span`
    - now `semantic_windows_reject_covering_node_span_at_compile_mapping_stage` asserting no compile-time mapping for covering spans.
    - added `semantic_windows_without_tags_reject_category_mismatch` to ensure missing accel tags do not bypass semantic kind/category compatibility.
  - Tightened missing-tag fallback in semantic window mapping: absent graph tags now require accel-category compatibility (`Elementwise|Transpose` for elementwise windows, `Reduction` for reduction windows, `MatMul` for matmul windows) instead of unconditional admission.
  - Compile path now emits fusion groups from semantic instruction windows without compile-time node assignment (`derive_semantic_fusion_groups_from_instruction_windows`), making runtime fusion plan preparation the sole node-reconciliation boundary.
  - This further reduces compile-time accel-graph coupling and advances Plan 7 toward semantic-window-owned planning artifacts.
  - Legacy compile-time node-mapping helper path (`derive_semantic_fusion_groups_from_candidates` and its accel-node matching helpers) is now explicitly test-only, removing this fallback surface from production builds.
  - Validation: `cargo test -p runmat-vm semantic_windows_reject_disjoint_gap_at_compile_mapping_stage -- --nocapture`, `cargo test -p runmat-vm semantic_windows_reject_partial_overlap_at_compile_mapping_stage -- --nocapture`, `cargo test -p runmat-vm semantic_windows_reject_covering_node_span_at_compile_mapping_stage -- --nocapture`, `cargo test -p runmat-vm semantic_windows_reject_overly_wide_covering_node_spans -- --nocapture`, `cargo test -p runmat-vm semantic_windows_ -- --nocapture`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 runtime fusion sanitization heuristic reduction
  - Tightened runtime group sanitization in `crates/runmat-accelerate/src/fusion.rs` in two steps:
    - removed overlap-or-touch (`<=1` disjoint gap) recovery
    - then required contained node spans (not merely overlap)
    - and restricted runtime graph scan backfill to compile-unmapped groups only (no stale mapped-node remap)
  - Runtime recovery now aligns with compile-time contained-span mapping, reducing residual span-jitter reconciliation heuristics.
  - Updated runtime sanitization tests:
    - `prepare_fusion_plan_recovers_empty_group_nodes_from_contained_runtime_span`
    - `prepare_fusion_plan_rejects_stale_mapped_nodes_without_runtime_remap`
    - `prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_node_covers_group_span`
    - plus existing negative far-span coverage remains.
  - Validation: `cargo test -p runmat-accelerate prepare_fusion_plan_ -- --nocapture`, `cargo test -p runmat-accelerate sanitize_runtime_groups -- --nocapture`, `cargo test -p runmat-vm semantic_windows_ -- --nocapture`, `cargo fmt --all --check`, `git diff --check`.

- (pending commit) Plan 7 compile/runtime fusion reconciliation boundary ratchet
  - Added `primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes` in `crates/runmat-vm/src/bytecode/compile.rs`.
  - The test locks the intended boundary:
    - compile emits semantic-window fusion scaffolding with `group.nodes` empty (no compile-time accel-node assignment),
    - runtime `prepare_fusion_plan(...)` reconciles executable groups by assigning node IDs from the accel graph.
  - This guards against regressions that re-introduce compile-time node reconciliation or bypass runtime plan preparation as the executable-node boundary.
  - Validation: `cargo test -p runmat-vm --features native-accel primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes -- --nocapture`.

- (pending commit) Plan 5 core entrypoint-resolution identifier-contract ratchet
  - Updated `crates/runmat-core/src/session/run.rs` test `source_input_path_errors_for_invalid_named_entrypoint_target` to assert the stable runtime identifier contract (`RunMat:EntrypointResolveFailed`) instead of message substring matching.
  - This keeps manifest-driven entrypoint failure coverage pinned to behavior contract surfaces rather than display text.
  - Validation: `cargo test -p runmat-core source_input_path_errors_for_invalid_named_entrypoint_target -- --nocapture`.

- (pending commit) Plan 5 manifest/composition test typed-error ratchet
  - Refactored display-string assertions in `crates/runmat-config/tests/project_manifest.rs` to typed error variant checks:
    - `ProjectManifestLoadError::{Validation,Parse}`
    - `ProjectSourceIndexError::InvalidSourceRoot`
    - `ProjectEntrypointResolveError::SourceIndex`
    - `DiscoverProjectEntrypointError::Resolve`
    - `ProjectCompositionError::{MissingDependencyManifest,DuplicatePackageName}`
  - This keeps manifest/composition contracts pinned to semantic error surfaces and stable fields (`dependency`, `package`, structured variants) instead of rendered error text.
  - Validation: `cargo test -p runmat-config --test project_manifest -- --nocapture`.

- (pending commit) Core semicolon matrix output typed-value ratchet
  - Updated `crates/runmat-core/tests/semicolon_suppression.rs` `test_matrix_semicolon_suppression` to assert a concrete tensor value (`shape == [1,3]`, `data == [1.0, 2.0, 3.0]`) instead of substring matching on rendered output.
  - This keeps semicolon-suppression behavior checks aligned to runtime value contracts rather than display formatting.
  - Validation: `cargo test -p runmat-core --test semicolon_suppression test_matrix_semicolon_suppression -- --nocapture`.

- (pending commit) Plan 5 core dependency-symbol composition ratchet
  - Added `discover_known_project_symbols_includes_dependency_alias_qualified_names` in `crates/runmat-core/src/session/run.rs`.
  - The test asserts core-known-symbol discovery (used by eval-hook parse/lower context) includes dependency package and dependency-alias qualified symbols from `runmat.toml` composition (`summarize`, `statslib.summarize`, `statsdep.summarize`).
  - This strengthens manifest/composition-graph evidence at the core session boundary, not only in `runmat-config` unit tests.
  - Validation: `cargo test -p runmat-core discover_known_project_symbols_includes_dependency_alias_qualified_names -- --nocapture`.

- (pending commit) Goal checklist legacy-surface evidence refresh
  - Re-ran goal criterion grep: `rg -n "compile_legacy|LegacyUserFunction|runmat_vm::execute|HirProgram|VarId" crates`.
  - Current result: no matches in `crates`, indicating legacy compiler/runtime symbol surfaces tracked by this gate are absent from production code paths.
  - Updated `docs-tmp/GOAL.md` criterion statuses accordingly (criterion 1 now marked substantially met pending full closeout audit; criterion 3 status refreshed with current composition-wiring evidence).

- (pending commit) Plan 7 fusion planner-source metadata exact-contract ratchet
  - Tightened planner source assertions in `crates/runmat-core/tests/fusion_regressions.rs` from substring matching to exact equality:
    - `compile_fusion_plan_exposes_semantic_planner_metadata` now asserts `semantic-mir-analysis`.
    - `runtime_fusion_snapshot_exposes_semantic_planner_metadata` now asserts `semantic-mir-analysis-runtime`.
  - Fusion planner metadata now carries explicit `accel_graph_state` (`present|missing|unknown`) separate from semantic source tags.
  - This hardens semantic-fact-driven fusion planner metadata checks against accidental source-tag drift hidden by partial matches.
  - Validation: `cargo test -p runmat-core --test fusion_regressions compile_fusion_plan_exposes_semantic_planner_metadata -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions runtime_fusion_snapshot_exposes_semantic_planner_metadata -- --nocapture`.

- (pending commit) Plan 7 core fusion snapshot input decoupling ratchet
  - Removed direct `accel_graph` parameter dependency from `build_fusion_snapshot` in `crates/runmat-core/src/fusion/snapshot.rs`; snapshot construction now consumes semantic fusion artifacts plus planner metadata (`accel_graph_state`) without needing an accel-graph object at this boundary.
  - Updated core compile/runtime snapshot call-sites and snapshot unit tests to the new signature.
  - This further reduces bytecode accel-graph coupling in core/session fusion diagnostics while preserving explicit accel-graph presence reporting through planner metadata.
  - Validation: `cargo test -p runmat-core --test fusion_regressions compile_fusion_plan_exposes_semantic_planner_metadata -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions runtime_fusion_snapshot_exposes_semantic_planner_metadata -- --nocapture`, `cargo test -p runmat-core semantic_candidate_summary_emits_without_accel_graph -- --nocapture`.

- (pending commit) Plan 7 runtime semantic-window scaffold fallback ratchet
  - Added shared runtime helper `Bytecode::runtime_fusion_groups` in `crates/runmat-vm/src/bytecode/program.rs`.
  - Runtime planning now prefers compile-populated `bytecode.fusion_groups`, but when those are empty and semantic candidate/window metadata exists, it derives fusion-group scaffolds directly from semantic instruction windows before calling `prepare_fusion_plan`.
  - Core fusion snapshot generation now consumes the same helper in `crates/runmat-core/src/session/compile.rs` and `crates/runmat-core/src/session/run.rs`, aligning diagnostics with runtime planning behavior.
  - This reduces runtime dependence on compile-populated fusion-group artifacts and keeps the runtime planning boundary semantic-window driven under missing-group conditions.
  - Added unit coverage:
    - `runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty`
    - `runtime_fusion_groups_prefer_existing_bytecode_groups`
  - Validation: `cargo test -p runmat-vm runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty -- --nocapture`, `cargo test -p runmat-vm runtime_fusion_groups_prefer_existing_bytecode_groups -- --nocapture`, `cargo check -p runmat-vm --features native-accel`.

- (pending commit) Plan 7 preserve accel graph under semantic fusion scaffolds
  - Updated VM compile path in `crates/runmat-vm/src/bytecode/compile.rs` to preserve `bytecode.accel_graph` whenever semantic candidate/window scaffolds exist, even when compile-time executable fusion groups are empty after filtering.
  - Rationale: runtime planning now has a semantic-window fallback path (`Bytecode::runtime_fusion_groups`), so dropping the graph at compile time prematurely can block runtime semantic-fact-driven reconciliation.
  - Validation: `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops -- --nocapture`, `cargo test -p runmat-vm primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions compile_fusion_plan_exposes_semantic_candidates_without_bytecode_groups -- --nocapture`, `cargo check -p runmat-vm --features native-accel`.

- (pending commit) Plan 7 runtime accel-graph on-demand materialization ratchet
  - Added runtime helper `runtime_accel_graph_for_fusion` in `crates/runmat-vm/src/interpreter/state.rs`.
  - When compile-provided `bytecode.accel_graph` is missing but semantic runtime groups exist, interpreter startup now materializes a runtime accel graph from bytecode instructions/var types before `prepare_fusion_plan`.
  - This further reduces hard dependency on compile-time accel-graph materialization while keeping semantic-gated runtime planning behavior intact.
  - Added unit coverage:
    - `runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing`
    - `runtime_accel_graph_is_not_materialized_when_runtime_groups_are_empty`
  - Validation: `cargo test -p runmat-vm runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing -- --nocapture`, `cargo test -p runmat-vm runtime_accel_graph_is_not_materialized_when_runtime_groups_are_empty -- --nocapture`, `cargo test -p runmat-vm runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty -- --nocapture`, `cargo check -p runmat-vm --features native-accel`.

- (pending commit) Plan 7 runtime stack-layout annotation for semantic fallback groups
  - Added `Bytecode::runtime_fusion_groups_for_graph` in `crates/runmat-vm/src/bytecode/program.rs`, which applies `annotate_fusion_groups_with_stack_layout` to runtime-selected fusion groups when a graph is available.
  - Interpreter runtime planning now uses that graph-aware helper in `crates/runmat-vm/src/interpreter/state.rs` before `prepare_fusion_plan`, so semantic-window fallback groups can carry stack-layout metadata rather than only compile-populated groups.
  - This keeps runtime fallback semantics aligned with stack-layout-sensitive fusion planning behavior and further reduces compile-group dependence.
  - Validation: `cargo test -p runmat-vm runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing -- --nocapture`, `cargo test -p runmat-vm runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty -- --nocapture`, `cargo test -p runmat-vm primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes -- --nocapture`, `cargo check -p runmat-vm --features native-accel`.

- (pending commit) Plan 7 shared runtime graph materialization for core/VM fusion surfaces
  - Moved runtime accel-graph-on-demand helper into shared bytecode API: `Bytecode::runtime_accel_graph_for_fusion` in `crates/runmat-vm/src/bytecode/program.rs`.
  - VM interpreter fusion planning and core fusion snapshot paths now both consume the shared materialization + graph-aware group helpers, aligning compile-preview/runtime diagnostics with actual runtime planning behavior when compile graph artifacts are missing.
  - Updated:
    - `crates/runmat-vm/src/interpreter/state.rs`
    - `crates/runmat-core/src/session/compile.rs`
    - `crates/runmat-core/src/session/run.rs`
  - Validation: `cargo test -p runmat-vm runtime_accel_graph_materializes_when_semantic_groups_exist_and_compile_graph_is_missing -- --nocapture`, `cargo test -p runmat-vm runtime_fusion_groups_fallback_to_semantic_windows_when_bytecode_groups_are_empty -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions compile_fusion_plan_exposes_semantic_planner_metadata -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions runtime_fusion_snapshot_exposes_semantic_planner_metadata -- --nocapture`, `cargo test -p runmat-core --test fusion_regressions compile_fusion_plan_exposes_semantic_candidates_without_bytecode_groups -- --nocapture`, `cargo check -p runmat-core`, `cargo check -p runmat-vm --features native-accel`.

## Next Resolution Items

- Keep legacy assertion/reference cleanup on maintenance watch for non-targeted surfaces; core/config/vm/cli targeted migration surfaces are now on typed/exact contracts.
- Plan 5 and Plan 7 evidence audit has been captured in `docs-tmp/DELIVERABLE_AUDIT.md`; follow-up is implementation closeout, not status ambiguity.
- Finish core/session project composition wiring so resolver ownership is composition-graph-driven end-to-end (Plan 5 closeout path); CLI manifest entrypoint resolution now delegates to `runmat-config`.
- Shift fusion candidate planning source-of-truth from bytecode accel graph to semantic/MIR/analysis products (Plan 7 closeout path).
