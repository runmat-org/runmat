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

- (pending commit) Plan 5 CLI integration step
  - `runmat-cli` script execution now resolves named entrypoints from discovered `runmat.toml` manifests when the provided run target is not an existing file.
  - Path entrypoints are resolved with optional `.m` inference; module/function entrypoints return an explicit "not yet executable from CLI run path" error instead of silently misrouting.
  - Added `runmat-cli` unit coverage for both path-target resolution and module/function-target rejection.
  - Validation: `cargo test -p runmat --lib`, `cargo fmt --all --check`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- (pending commit) Plan 7 visibility bridge in fusion snapshots
  - `FusionPlanSnapshot` now carries planner metadata (`source`, `mir_local_fact_count`, `mir_diagnostic_count`).
  - `RunMatSession::compile_fusion_plan` and runtime execution path (`execute_internal` when fusion snapshot emission is enabled) now compute MIR + `AnalysisStore` and record analysis fact counts into fusion snapshot metadata, making semantic-analysis availability explicit in the fusion planning surface.
  - Fusion candidate construction itself is still bytecode-graph-driven; this slice improves observability and contract clarity while that migration remains open.
  - Validation: `cargo test -p runmat --lib`, `cargo test -p runmat-config`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo fmt --all --check`, `cargo check --workspace`, `git diff --check`.

## Next Resolution Items

- Finish converting remaining legacy test/doc references that imply removed APIs where they block semantic-only confidence.
- Plan 5 and Plan 7 evidence audit has been captured in `docs-tmp/DELIVERABLE_AUDIT.md`; follow-up is implementation closeout, not status ambiguity.
- Wire project-manifest composition into CLI/core entrypoint selection and resolver graph (Plan 5 closeout path).
- Shift fusion candidate planning source-of-truth from bytecode accel graph to semantic/MIR/analysis products (Plan 7 closeout path).
