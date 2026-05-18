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
