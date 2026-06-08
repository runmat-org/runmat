# Merge Plan: `origin/dev` into `geometry-analysis-core`

Date: 2026-06-08

## Goal

Merge the current `origin/dev` into `geometry-analysis-core` without losing the simulation/analysis effort built under `docs/simulation`, and without dragging older branch-era platform structure back over the current RunMat runtime, workspace, CI, docs, and release machinery.

The expected outcome is a merge commit where:

- the simulation system remains present and buildable,
- upstream `origin/dev` remains the source of truth for broad platform refactors,
- analysis/geometry/meshing additions are adapted to the current workspace,
- branch-only governance and EM readiness work is preserved,
- verification is staged enough to isolate failures quickly.

## Current State

- Current branch: `geometry-analysis-core`
- Current branch head: `6e1748881 Add EM phase assertion parity guard`
- Upstream target: `origin/dev`
- Upstream head at inspection time: `ccc01633a Merge pull request #381 from runmat-org/rm-514`
- Merge base: `95f2b6c0c8fbf7f474c85f9619c2bcd68be85ee7`
- Divergence at inspection time:
  - `origin/dev` is about `2600` commits ahead of merge base.
  - `geometry-analysis-core` is about `403` commits ahead of merge base.

Working tree is currently dirty with a small, coherent uncommitted change set:

- `docs/simulation/EM_TRACK.md`
- `docs/simulation/ROADMAP.md`
- `docs/simulation/STATUS.md`
- `docs/simulation/WORKLOG.md`
- `scripts/tests/test_release_readiness_nonlinear.py`

That uncommitted set adds an EM trend-assertion contract parity guard and matching simulation-doc updates. It should be committed or stashed before any merge attempt.

## Branch Intent To Preserve

The simulation branch is not just a documentation branch. It adds a programmatic simulation system with:

- new analysis/geometry/meshing crates:
  - `crates/runmat-analysis/core`
  - `crates/runmat-analysis/fea`
  - `crates/runmat-geometry/core`
  - `crates/runmat-geometry/io`
  - `crates/runmat-geometry/ops`
  - `crates/runmat-meshing/core`
- runtime operation surfaces:
  - `analysis.*`
  - `geometry.*`
- study workflow operations:
  - `analysis.validate_study/v1`
  - `analysis.plan_study/v1`
  - `analysis.run_study/v1`
  - study sweep validation/planning/run variants
- evidence and artifact storage:
  - run artifacts
  - study artifacts
  - prep artifacts
  - readiness outputs
- governance tooling:
  - nonlinear benchmark schema validation
  - release readiness
  - external reference benchmark generation and validation
  - threshold ratchet reports
  - promotion threshold calibration
  - prep calibration and recommendation flows
- canonical docs under `docs/simulation`.

Core architecture constraints from the branch:

- Runtime orchestrates; physics math lives in analysis/FEA crates.
- Contracts are versioned and evolve additively.
- Typed errors and reason codes are stable machine interfaces.
- Governance is release-blocking on protected branches.
- Documentation and implementation are co-updated.

## Upstream Themes To Respect

`origin/dev` has moved substantially since the branch point. It should be treated as source of truth for broad platform shape.

Relevant upstream changes include:

- release/version movement to `0.5.0`,
- workspace restructuring away from older `runmat-ignition` / `runmat-kernel` branch-era shape,
- current `runmat-hir`, `runmat-mir`, `runmat-vm`, and `runmat-server-client` workspace shape,
- runtime builtin descriptor migration and runtime facade cleanup,
- large LSP and static-analysis refactors,
- class support and extended function semantics,
- control/plotting/builtin additions,
- WASM/bindings workflow changes,
- CI and release workflow updates,
- docs and website restructuring, including WASM docs moving to `docs/wasm/index.md`.

Merge posture:

- Prefer upstream for broad platform structure.
- Re-apply simulation additions into that structure.
- Avoid resurrecting older branch-era script paths, docs files, crate versions, or CI conventions unless they are explicitly still needed.

## Merge Strategy

Use a merge commit, not a rebase.

Reasons:

- the branch has hundreds of commits and appears published,
- the branch history contains many checkpoint commits,
- a merge commit makes conflict resolution auditable,
- PR integration can squash later if a cleaner final history is desired.

Recommended command shape after committing/stashing the dirty changes:

```bash
git fetch origin dev:refs/remotes/origin/dev
git merge origin/dev
```

If the merge becomes noisy, stop and resolve by ownership boundary rather than file order:

1. workspace and dependency graph,
2. runtime module integration,
3. static-analysis/LSP semantic fixes,
4. CI and scripts,
5. docs,
6. generated lockfile and formatting,
7. verification.

## Known Conflict Set

A dry-run merge simulation reported conflicts in:

- `.github/workflows/ci.yml`
- `.github/workflows/publish-dev-bindings.yml`
- `.github/workflows/release.yml`
- `Cargo.toml`
- `Cargo.lock`
- `crates/runmat-lsp/src/core/analysis.rs`
- `crates/runmat-static-analysis/src/lints/shape.rs`
- `docs/wasm/TESTING.md`
- `scripts/release/bump-release.sh`
- `scripts/runtime/test-wasm-headless.sh`

Additional auto-merged but high-risk files include:

- `crates/runmat-runtime/Cargo.toml`
- `crates/runmat-runtime/src/lib.rs`
- `crates/runmat-static-analysis/tests/lints.rs`

## Resolution Policy By Area

### 1. Dirty working tree

Before merging:

- Commit the current EM trend-assertion parity guard and doc updates, or stash them.
- Prefer a small commit, for example:
  - `Add EM trend assertion parity guard`
- Confirm clean status before starting the merge:

```bash
git status --short
```

### 2. Root `Cargo.toml`

Start from upstream `origin/dev`.

Preserve upstream:

- `0.5.0` workspace dependency versions,
- current upstream workspace members,
- current upstream dependency set,
- removal of branch-era `runmat-ignition` / `runmat-kernel` if they are not present upstream,
- inclusion of `runmat-mir`, `runmat-vm`, and `runmat-server-client` where upstream has them.

Add back simulation crates as current-version workspace members:

```toml
"crates/runmat-geometry/core",
"crates/runmat-geometry/io",
"crates/runmat-geometry/ops",
"crates/runmat-analysis/core",
"crates/runmat-analysis/fea",
"crates/runmat-meshing/core",
```

Add back simulation workspace dependencies using the upstream version line, likely `=0.5.0`:

```toml
runmat-geometry-core = { version = "=0.5.0", path = "crates/runmat-geometry/core" }
runmat-geometry-io = { version = "=0.5.0", path = "crates/runmat-geometry/io" }
runmat-geometry-ops = { version = "=0.5.0", path = "crates/runmat-geometry/ops" }
runmat-analysis-core = { version = "=0.5.0", path = "crates/runmat-analysis/core" }
runmat-analysis-fea = { version = "=0.5.0", path = "crates/runmat-analysis/fea" }
runmat-meshing-core = { version = "=0.5.0", path = "crates/runmat-meshing/core" }
```

Then update each new simulation crate `Cargo.toml` from `0.2.8` to `0.5.0`.

### 3. `Cargo.lock`

Do not manually resolve lockfile hunks beyond clearing conflict markers if absolutely necessary.

Preferred approach:

1. Resolve `Cargo.toml` and crate manifests first.
2. Run a workspace metadata/check command to regenerate the lockfile.
3. Review lockfile changes for unexpected dependency churn.

Candidate commands:

```bash
cargo metadata --no-deps
cargo check -p runmat-analysis-core
```

If lockfile resolution is too tangled, use upstream `Cargo.lock` as the base and regenerate after all manifests are resolved.

### 4. `crates/runmat-runtime/src/lib.rs`

Start from upstream `origin/dev`.

Add only the simulation module exports needed by the branch:

```rust
pub mod analysis;
pub mod geometry;
pub mod operations;
```

Preserve upstream runtime facade decisions:

- current builtin descriptor APIs,
- current common helper module paths,
- current async/import plotting API names,
- current object/class/function handle support,
- upstream lint and cfg attributes.

Do not restore branch-era root modules such as `arrays`, `concatenation`, `elementwise`, `indexing`, or `matrix` if upstream has moved them under `builtins::common`.

### 5. `crates/runmat-runtime/Cargo.toml`

Start from upstream `origin/dev`.

Add only simulation dependencies:

```toml
runmat-geometry-core = { workspace = true }
runmat-geometry-io = { workspace = true }
runmat-geometry-ops = { workspace = true }
runmat-analysis-core = { workspace = true }
runmat-analysis-fea = { workspace = true }
runmat-meshing-core = { workspace = true }
```

Preserve upstream additions such as `image`, upstream dev dependencies, and feature changes.

### 6. Simulation crates

Preserve the new crates, but adapt them to upstream workspace versions and APIs.

Initial checks:

```bash
cargo test -p runmat-geometry-core
cargo test -p runmat-geometry-io
cargo test -p runmat-geometry-ops
cargo test -p runmat-meshing-core
cargo test -p runmat-analysis-core
cargo test -p runmat-analysis-fea
```

Likely adjustments:

- manifest version updates,
- imports affected by upstream runtime/builtins changes,
- clippy or formatting changes,
- possible serde or error-type compatibility issues.

Do not change analysis/FEA layering to work around compile errors. Runtime may call analysis; analysis crates should not depend back on runtime.

### 7. `crates/runmat-runtime/src/analysis/*`

Preserve branch modules, but compile them against upstream runtime APIs.

Watch for upstream changes in:

- `RuntimeError` construction,
- `Value` variants,
- builtin registration shape,
- plotting/import APIs,
- operation result conventions,
- async/runtime helper paths,
- descriptor-backed errors.

Keep operation versions stable:

- `analysis.create_model/v1`
- `analysis.run_linear_static/v1`
- `analysis.run_modal/v1`
- `analysis.run_acoustic/v1`
- `analysis.run_transient/v1`
- `analysis.run_thermal/v1`
- `analysis.run_nonlinear/v1`
- `analysis.run_electromagnetic/v1`
- `analysis.run_cfd/v1`
- `analysis.run_cht/v1`
- `analysis.run_fsi/v1`
- `analysis.validate_study/v1`
- `analysis.plan_study/v1`
- `analysis.run_study/v1`
- sweep operation variants.

Do not silently break payload fields. If an upstream API forces a payload break, add a new version instead.

### 8. `crates/runmat-runtime/src/geometry/*`

Preserve geometry operations and prep artifact lifecycle.

Keep operation versions stable:

- `geometry.inspect/v1`
- `geometry.load/v1`
- `geometry.compute_stats/v1`
- `geometry.list_regions/v1`
- `geometry.query_entities/v1`
- `geometry.capture_view/v1`
- `geometry.prep_for_analysis/v1`
- `geometry.prep_artifact_health/v1`

Verify prep artifact env knobs still match docs:

- `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT`
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS`
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY`
- `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS`

### 9. Static analysis shape lint

Do not blindly keep the branch's old hunk.

Branch intent:

- numeric range indexing should not emit `lint.shape.logical_index`,
- numeric vector/scalar indexing should not emit `lint.shape.logical_index`,
- matching logical indexing should remain accepted,
- mismatched logical indexing should still warn.

Upstream static analysis now uses MIR-based shape inference. The current upstream logic appears to check index element count without distinguishing logical mask indices from numeric indices.

Port this as a test-driven fix:

1. Keep/add tests equivalent to:
   - `shape_lint_allows_numeric_range_indexing`
   - `shape_lint_allows_numeric_vector_and_scalar_indexing`
   - `shape_lint_allows_matching_logical_indexing`
2. Add or preserve an LSP regression test that no `lint.shape.logical_index` diagnostic appears for numeric indexing.
3. Adjust MIR shape lint logic to detect logical-index intent, not just index shape.

Verification:

```bash
cargo test -p runmat-static-analysis
cargo test -p runmat-lsp diagnostics_do_not_report_logical_index_lint_for_numeric_indexing
cargo test -p runmat-lsp hover_reports_indexed_slice_shape_for_assigned_symbol
```

The exact test names may need adjustment after conflict resolution.

### 10. `crates/runmat-lsp/src/core/analysis.rs`

Start from upstream `origin/dev`.

Port only the branch's narrow LSP tests if still relevant:

- no logical-index lint for numeric indexing,
- hover for indexed slice shape.

Do not replace upstream's large LSP refactor with the branch version. Upstream has thousands of lines of current LSP behavior that should remain.

### 11. CI workflows

Start from upstream workflows.

Preserve upstream:

- current build/test job shape,
- current release flow,
- current WASM/bindings flow,
- upstream removal of `.github/workflows/publish-dev-bindings.yml`,
- upstream script paths unless deliberately changed in a separate follow-up.

Reintroduce simulation governance as a dedicated job in `.github/workflows/ci.yml`, likely named `nonlinear-conformance`.

Keep the branch's high-value governance steps:

- run nonlinear benchmark conformance,
- verify filesystem artifact replay stability,
- validate nonlinear benchmark report keys,
- test release-readiness script logic,
- validate prep calibration evidence,
- generate prep calibration recommendations,
- validate prep calibration promotion,
- analyze rolling trends,
- summarize benchmark report,
- summarize prep artifact SLO,
- generate and validate promotion threshold calibration,
- generate and validate external reference benchmark artifact,
- evaluate nonlinear release readiness,
- generate and validate threshold ratchet report,
- upload analysis artifacts on failure/protected branches.

Then adapt the job to upstream CI realities:

- current runner labels,
- current toolchain setup,
- current dependency install strategy,
- current artifact upload conventions,
- upstream branch names and protected-branch conditions.

Avoid mixing the branch's older WASM script relocation into this merge.

### 12. Scripts

Keep upstream's top-level WASM script layout:

- `scripts/test-wasm-headless.sh`
- `scripts/chrome-headless.sh`
- `scripts/resolve-chromedriver.sh`

Do not force the branch's `scripts/runtime/test-wasm-headless.sh` relocation during this merge. Upstream's script has newer chromedriver resolution and should win.

Preserve branch simulation scripts:

- `scripts/analysis/governance/*`
- `scripts/analysis/reporting/*`
- `scripts/analysis/prep_calibration/*`
- `scripts/analysis/thermo_artifacts/*`
- `scripts/analysis/reference_data/*`
- `scripts/tests/test_*analysis*`
- related `scripts/tests/test_release_readiness_nonlinear.py`

Resolve release script conflict by keeping upstream `scripts/cut-release.sh` unless there is a current product requirement for `scripts/release/bump-release.sh`. If release script reorganization is still desired, do it as a follow-up after the merge.

### 13. WASM docs

Accept upstream deletion of `docs/wasm/TESTING.md`.

If branch-only WASM testing details are still useful, port them into upstream `docs/wasm/index.md` in a separate docs follow-up. Do not resurrect `docs/wasm/TESTING.md` during the merge.

### 14. Simulation docs

Preserve `docs/simulation/*` and `docs/simulation/ARCHIVE/*`.

After merge resolution, update simulation docs only if the merge materially changes execution posture, for example:

- workspace version movement to `0.5.0`,
- CI job naming changes,
- script paths used by governance,
- verification command updates,
- any operation contract version changes.

Do not edit archive files unless specifically correcting broken links caused by the merge.

### 15. Target artifacts

The branch currently includes:

- `crates/runmat-runtime/target/runmat-analysis-artifacts/thermo-fields/*.json`

Review whether committed target artifacts are intentional. If they are canonical fixtures, consider moving them out of `target/` in a follow-up. During this merge, do not delete them unless tests or governance clearly no longer need them.

## Verification Plan

Run verification in layers so failures are attributable.

### Layer 0: Merge hygiene

```bash
git status --short
rg -n "<<<<<<<|=======|>>>>>>>" .
cargo fmt --check
```

### Layer 1: workspace metadata and manifests

```bash
cargo metadata --no-deps
cargo check -p runmat-analysis-core
cargo check -p runmat-analysis-fea
cargo check -p runmat-runtime
```

### Layer 2: new crate tests

```bash
cargo test -p runmat-geometry-core
cargo test -p runmat-geometry-io
cargo test -p runmat-geometry-ops
cargo test -p runmat-meshing-core
cargo test -p runmat-analysis-core
cargo test -p runmat-analysis-fea
```

### Layer 3: static-analysis and LSP conflict area

```bash
cargo test -p runmat-static-analysis
cargo test -p runmat-lsp
```

If full LSP tests are too broad initially, run focused tests first, then broaden.

### Layer 4: runtime simulation tests

```bash
cargo test -p runmat-runtime --lib analysis
cargo test -p runmat-runtime --test analysis
cargo test -p runmat-runtime --test operation_contracts
cargo test -p runmat-runtime --test geometry_prep_conformance
cargo test -p runmat-runtime --test prep_solve_conformance
```

Use `-- --test-threads=1` for tests that touch shared filesystem artifact roots or global runtime state.

### Layer 5: governance Python tests

```bash
python3 -m unittest \
  scripts.tests.test_release_readiness_nonlinear \
  scripts.tests.test_validate_analysis_report_nonlinear \
  scripts.tests.test_validate_external_reference_benchmark \
  scripts.tests.test_external_reference_baseline \
  scripts.tests.test_generate_external_reference_benchmark \
  scripts.tests.test_generate_threshold_ratchet_report \
  scripts.tests.test_validate_threshold_ratchet_report \
  scripts.tests.test_generate_promotion_threshold_calibration \
  scripts.tests.test_validate_promotion_threshold_calibration \
  scripts.tests.test_evaluate_prep_calibration_drift \
  scripts.tests.test_promote_prep_calibration_evidence
```

Also run direct script smoke checks where expected artifacts exist:

```bash
python3 scripts/analysis/governance/release_readiness_nonlinear.py
python3 scripts/analysis/governance/validate_analysis_report_nonlinear.py
python3 scripts/analysis/governance/generate_external_reference_benchmark.py
python3 scripts/analysis/governance/validate_external_reference_benchmark.py
```

### Layer 6: broad workspace gate

After focused gates pass:

```bash
cargo test --all-targets --all-features
```

If this is too slow locally, run the focused gates locally and rely on CI for the full matrix, but document that explicitly in the merge commit or PR notes.

## Expected Follow-Ups After Merge

These should not block the merge unless they become required for build/test success:

- Move canonical analysis artifacts out of `target/` if they are meant to be source-controlled fixtures.
- Decide whether script directory reorganization is still desired after upstream's script updates.
- Fold WASM test-harness notes into `docs/wasm/index.md` if still relevant.
- Revisit the nonlinear CI job runtime cost and runner labels after first CI run.
- Consider squashing checkpoint-heavy branch history at PR integration time.
- Audit new simulation crates for publication metadata consistency with upstream `0.5.0`.

## Decision Log

- Use merge commit, not rebase.
- Upstream wins broad platform structure.
- Simulation wins new analysis/geometry/meshing functionality and governance artifacts.
- Cargo versions should align to upstream `0.5.0`.
- Runtime module integration should be additive and minimal.
- WASM script/docs relocation from the branch should not be forced during this merge.
- Shape-lint behavior should be ported by semantic intent against upstream MIR analysis.

## Stop Conditions

Stop and reassess before committing the merge if any of these happen:

- resolving conflicts requires breaking an `analysis.*` or `geometry.*` operation payload without a version bump,
- simulation crates need a reverse dependency from analysis/FEA back into runtime,
- CI resolution requires deleting major governance steps rather than adapting them,
- static-analysis/LSP tests cannot preserve numeric-indexing behavior without broad inference changes,
- lockfile regeneration introduces unrelated dependency churn that cannot be explained by upstream plus simulation crates.

