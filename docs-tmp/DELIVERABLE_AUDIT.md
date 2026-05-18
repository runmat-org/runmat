# Deliverable Audit

Date: 2026-05-18

This audit maps the active objective to concrete repository evidence and marks each item as `met`, `partial`, or `open`.

## Objective Breakdown

1. Active execution/analysis paths are semantic HIR -> MIR -> analysis -> VM/runtime.
2. No production legacy path dependence.
3. MATLAB core semantics represented by compiler/runtime products.
4. Project composition and entrypoints are manifest-driven.
5. Nominal class/builtin metadata is unified.
6. Accel/fusion planning is semantic-fact-driven.
7. Validation cadence is green.

## Evidence Checklist

### 1) Semantic pipeline (`partial`)

- Evidence:
  - semantic compile path in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs)
  - MIR lowering API used before VM compile in [stress.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-gc/tests/stress.rs)
- Gap:
  - broad consumer migration across all crates remains in progress (see `PLAN.3.md` / `PROGRESS.md`).

### 2) No production legacy path dependence (`partial`)

- Evidence:
  - `rg -n "\\bHirProgram\\b|runmat_vm::execute\\b|compile_legacy\\b|LegacyUserFunction\\b" crates` has no hits in production crate code.
  - recent removal/migration commits in `NEXT_STEPS.md`.
- Gap:
  - confidence work remains in tests/docs and plan-closeout auditing.

### 3) MATLAB semantics as products (`partial`)

- Evidence:
  - semantic coverage ratchets tracked in [NEXT_STEPS.md](/Users/nallana/Source/runmat-acc-2/runmat/docs-tmp/NEXT_STEPS.md).
  - HIR/MIR semantic model in [TARGET_MODEL.md](/Users/nallana/Source/runmat-acc-2/runmat/docs-tmp/TARGET_MODEL.md).
- Gap:
  - designed gaps still open (async/future/spawn runtime model, aggregate edge behavior, remaining selector-plan normalization).

### 4) Manifest-driven composition/entrypoints (`partial`)

- Evidence:
  - dedicated `runmat.toml` manifest boundary added in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs):
    - typed sections (`package`, `sources`, `dependencies`, `entrypoints`)
    - discovery (`discover_project_manifest_from`)
    - parse/load (`parse_project_manifest_toml`, `load_project_manifest`)
    - validation of required shape + relative/existing paths + entrypoint target forms
  - tests in [project_manifest.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/tests/project_manifest.rs)
  - typed source-index scanning now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `build_project_source_index`, including explicit `+pkg`, `@ClassName`, and `private` discovery buckets.
  - shared config-layer entrypoint resolver now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `resolve_project_entrypoint`, with typed resolved target metadata and explicit resolution errors.
  - shared discovered-entrypoint resolver now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `resolve_named_entrypoint_from`, centralizing discovery + composition load + root-package selection + entrypoint resolution for consumers.
  - module/function entrypoint resolution now uses source-index qualified-name matching (`build_project_source_index`) instead of a direct dotted-path file heuristic, including support for class-folder targets (`+pkg/@ClassName/method.m`).
  - CLI run-path integration in [script.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/script.rs) now resolves manifest entrypoint names through the shared discovered-entrypoint resolver for both path targets and module/function targets.
  - config-layer composition graph loading remains in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `build_project_composition_graph`, including local path dependency manifest loading and per-package source indexes.
  - core request-path source loading in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs) now resolves simple named path inputs through the shared discovered-entrypoint resolver before source read.
- Gap:
  - core/session-level execution now resolves named entrypoint path inputs through composition metadata, but resolver/import consumers are not yet wired end-to-end to consume the composition graph as source-of-truth.
  - source-index discovery, shared entrypoint resolution, and composition-graph loading now exist at config-layer, but downstream resolver consumers have not yet switched to composition-graph/source-index ownership end-to-end.

### 5) Unified nominal class/builtin metadata (`partial`)

- Evidence:
  - shared callable identity/fallback policy in [hir.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/hir.rs).
  - builtin semantics surface in [semantics.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-builtins/src/semantics.rs).
- Gap:
  - full Plan 6 acceptance criteria not yet closed out across all consumers.

### 6) Semantic-fact-driven accel/fusion (`open`)

- Evidence:
  - analysis facts exist in [hir.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/hir.rs) and [store.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-mir/src/analysis/store.rs).
  - fusion snapshot planner metadata now records MIR analysis fact counts plus MIR semantic fusion signal counts and MIR fusion candidate-group counts in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), runtime emission path [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), and [types.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/types.rs), with signal/candidate extraction in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - fusion snapshot emission now includes a semantic candidate summary decision when bytecode fusion groups are empty but MIR semantic candidate evidence exists, so semantic planning signals remain visible in transitional no-group cases.
  - runtime/provider decision telemetry exists in [native_auto.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/native_auto.rs).
  - residency hooks exist in accelerate runtime.
- Blocking gap:
  - primary fusion candidate construction is still bytecode/accel-graph driven:
    - [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) builds accel graph from bytecode instructions and then calls `detect_fusion_groups`.
  - explicit spawned-task provider-handle lifetime semantics remain incomplete; compiler still has transitional spawn comment in [core.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/compiler/core.rs:1719).

### 7) Validation cadence (`met` for current slices)

- Latest executed gates:
  - `cargo test -p runmat-config`
  - `cargo fmt --all --check`
  - `cargo test -p runmat-core --test semicolon_suppression`
  - `cargo check --workspace`
  - `git diff --check`

## Current Conclusion

Objective is **not achieved**.

Highest-impact unresolved areas:

1. Plan 5 end-to-end composition wiring (CLI/core + resolver graph + source index).
2. Plan 7 shift from bytecode-first fusion planning to semantic/MIR/analysis-driven planning.
