# Progress

## Current Focus

Broad consumer migration and compatibility-surface cleanup, while keeping semantic pipeline validation green.

## Latest Committed Slices (2026-05-19)

- (pending commit) Plan 7 recursive residency clearing for nested GPU-handle values
  - VM acceleration residency clearing now traverses nested runtime values instead of only top-level `Value::GpuTensor`:
    - `Cell`
    - `Struct`
    - `Object`
    - `Closure` captures
    - `OutputList`
  - This closes a lifecycle gap where nested GPU handles could remain residency-marked after value replacement/clear paths.
  - Added `runmat-vm` unit coverage:
    - `clear_value_releases_nested_gpu_handles_in_cells`
    - `clear_value_releases_nested_gpu_handles_in_closure_captures`
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

## Next Resolution Items

- Finish converting remaining legacy test/doc references that imply removed APIs where they block semantic-only confidence.
- Plan 5 and Plan 7 evidence audit has been captured in `docs-tmp/DELIVERABLE_AUDIT.md`; follow-up is implementation closeout, not status ambiguity.
- Finish core/session project composition wiring so resolver ownership is composition-graph-driven end-to-end (Plan 5 closeout path); CLI manifest entrypoint resolution now delegates to `runmat-config`.
- Shift fusion candidate planning source-of-truth from bytecode accel graph to semantic/MIR/analysis products (Plan 7 closeout path).
