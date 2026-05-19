# Progress

## Current Focus

Broad consumer migration and compatibility-surface cleanup, while keeping semantic pipeline validation green.

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
