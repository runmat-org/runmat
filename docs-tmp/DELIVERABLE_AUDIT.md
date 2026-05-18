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
- Gap:
  - CLI/core entrypoint selection by project target and composition-graph-driven resolution are not yet wired end-to-end.
  - source-index discovery for MATLAB layout (`+pkg`, `@ClassName`, `private`) not yet evidenced as Plan 5-complete.

### 5) Unified nominal class/builtin metadata (`partial`)

- Evidence:
  - shared callable identity/fallback policy in [hir.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/hir.rs).
  - builtin semantics surface in [semantics.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-builtins/src/semantics.rs).
- Gap:
  - full Plan 6 acceptance criteria not yet closed out across all consumers.

### 6) Semantic-fact-driven accel/fusion (`open`)

- Evidence:
  - analysis facts exist in [hir.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/hir.rs) and [store.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-mir/src/analysis/store.rs).
  - fusion snapshot planner metadata now records MIR analysis fact counts in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs) and [types.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/types.rs).
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
