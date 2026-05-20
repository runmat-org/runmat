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
- Latest result: still partial; non-builtin semantic gap inventory remains open (recent progress includes compile-stage selector-plan invariant identifier ratchets, but aggregate-edge design gaps remain).

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
