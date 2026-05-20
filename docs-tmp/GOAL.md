# Goal

Complete the semantic migration of RunMat across Plans 0-7 until:

- active execution/analysis paths are semantic HIR -> MIR -> analysis -> VM/runtime
- production code does not depend on legacy compiler/runtime paths
- MATLAB core semantics are represented as compiler/runtime products
- project composition and entrypoint behavior are manifest-driven
- nominal class and builtin metadata are unified
- accel/fusion planning is semantic-fact-driven
- workspace validation cadence stays green

Scope contract for this goal:
- In scope: compiler/runtime migration closure work across Plans 0-7.
- Out of scope: standalone builtin completeness/parity work.
- Exception: builtin edits are in scope only when required to unblock an in-scope compiler/runtime/fusion closure item.

## Success Criteria Checklist

1. Semantic-only active pipeline
- Evidence: no production `compile_legacy`/legacy execute shims in active compiler/runtime paths.
- Evidence command: `rg -n "compile_legacy|LegacyUserFunction|runmat_vm::execute|HirProgram|VarId" crates`
- Current status: evidence command currently returns no matches in `crates`; treat as substantially met pending full closeout audit.

2. MATLAB core semantics modeled by products
- Evidence: semantic HIR/MIR + VM lowering coverage for indexing, calls, workspace effects, outputs, and compatibility diagnostics on the in-scope compiler/runtime migration path.
- Evidence files: `docs-tmp/TARGET_MODEL.md`, `docs-tmp/ABI_DESIGN.md`, `docs-tmp/NEXT_STEPS.md`.
- Current status: in progress (remaining designed non-builtin migration gaps tracked in `NEXT_STEPS.md`; builtin parity items are tracked separately unless they block in-scope closure).

3. Manifest-driven composition and entrypoints
- Evidence: config/discovery and entrypoint selection wired through config crates + CLI/session integration.
- Evidence commands: `rg -n "runmat.toml|entrypoint|manifest|sources|dependencies" crates/runmat-config crates/runmat-core crates/runmat-cli`.
- Current status: in progress. `runmat-config` has dedicated `runmat.toml` composition/discovery/entrypoint APIs with typed contract coverage; CLI/core entrypoint resolution and core known-symbol dependency-alias discovery are wired, but full composition-graph-driven closeout still requires explicit audit.

4. Unified nominal class/builtin metadata
- Evidence: shared callable/class identity and builtin semantics metadata surfaces used by runtime/lowering/analysis.
- Evidence files: `crates/runmat-hir/src/hir.rs`, `crates/runmat-builtins/src/semantics.rs`, `docs-tmp/PLAN.6.md`.
- Current status: in progress.

5. Semantic-fact-driven accel/fusion planning
- Evidence: MIR analysis store + fusion planning interfaces consume semantic products, with runtime/provider owning placement.
- Evidence commands: `rg -n "AnalysisStore|fusion|FusionPlan|Accel" crates/runmat-mir crates/runmat-vm crates/runmat-core`.
- Current status: in progress; requires explicit Plan 7 closeout audit.

6. Validation cadence green
- Required gates per slice:
  - `cargo fmt --all --check`
  - `cargo test -p runmat-core --test semicolon_suppression`
  - `cargo check --workspace`
  - `git diff --check`
- Current status: green for latest slices (see `docs-tmp/PROGRESS.md` log entries).
