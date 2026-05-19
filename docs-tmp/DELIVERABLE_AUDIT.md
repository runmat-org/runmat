# Deliverable Audit

Date: 2026-05-19

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
  - prepared execution artifacts now carry MIR in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/mod.rs), and fusion-plan preview/runtime metadata paths reuse that prepared MIR in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs) and [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs).
  - MIR lowering API used before VM compile in [stress.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-gc/tests/stress.rs)
- Gap:
  - broad consumer migration across all crates remains in progress (see `PLAN.3.md` / `PROGRESS.md`).

### 2) No production legacy path dependence (`met`)

- Evidence:
  - `rg -n "\\bHirProgram\\b|runmat_vm::execute\\b|compile_legacy\\b|LegacyUserFunction\\b" crates` has no hits in production crate code.
  - recent removal/migration commits in `NEXT_STEPS.md`.
- Residual watchpoints:
  - keep this grep in validation cadence to prevent regressions.

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
  - core request-path source loading now also infers `.m` for unresolved file-style path inputs before entrypoint-name fallback in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), matching Plan 5 path-target inference behavior.
  - CLI script target resolution now applies the same `.m` inference before named-entrypoint fallback in [script.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/script.rs).
  - core compile/lowering path now discovers composition source-index symbols from source context and passes them into HIR lowering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs).
  - core composition symbol discovery now includes root dependency alias-qualified symbols (`alias.symbol`) derived from composition graph root dependency mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), enabling wildcard import resolution through manifest dependency aliases.
  - HIR wildcard import resolution now uses project source-index symbol candidates for call/function-handle target resolution in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs) via lowering-context symbol inputs from [lowering_context.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering_context.rs).
  - core integration coverage now includes wildcard import resolution through project source-index symbols (`compile_input_resolves_wildcard_import_from_project_source_index`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - core integration coverage now includes wildcard import resolution through dependency aliases (`compile_input_resolves_wildcard_import_from_dependency_alias`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - core integration coverage now includes dependency-alias wildcard import resolution for function-handle lowering (`compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - dependency-alias wildcard function-handle coverage now asserts exact alias-qualified lowering identity (`CreateExternalFunctionHandle("statsdep.summarize")`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
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
  - fusion snapshot planner metadata now records MIR analysis fact counts plus MIR semantic fusion signal/candidate-group counts in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), runtime emission path [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), and [types.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/types.rs).
  - semantic fusion signal/candidate-group extraction is now a VM compile artifact concern in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and both counts and explicit semantic candidate-group artifacts are carried on bytecode products in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs).
  - semantic candidate-group artifacts now include MIR-localized run metadata (`function`, `block`, `stmt_start`, `stmt_end`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived during bytecode compilation in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and surfaced in fusion snapshot semantic-candidate node labels in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - bytecode fusion-group detection in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) is now semantically gated: `detect_fusion_groups` only runs when MIR semantic candidate-group count is non-zero.
  - VM compile now drops accel-graph artifacts entirely when no executable semantic-mapped fusion groups survive candidate/span filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), reducing unused bytecode-graph artifact retention in semantic-candidate/no-executable-group states.
  - fusion snapshot emission now includes semantic candidate nodes/decisions both when bytecode fusion groups are empty and when they are present, and this semantic path no longer hard-requires accel graph presence in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - runtime fusion-plan preparation now consumes semantic candidate-group counts at the VM/accelerate boundary in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) and [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs), with explicit transition diagnostics when semantic candidates exist but executable bytecode groups are absent.
  - VM bytecode now carries explicit semantic async/spawn metadata (`semantic_async_metadata`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived from MIR spawn sites in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and surfaced at interpreter setup in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) so spawned-task transition semantics are explicit at runtime boundaries.
  - VM bytecode semantic async metadata now carries explicit MIR await-site inventory (`mir_await_site_count`, `mir_await_sites`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived from MIR await terminators in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - MIR await boundaries now lower through explicit bytecode `Instr::Await` handling in [core.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/compiler/core.rs), [instr.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/instr.rs), [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), and [compiler.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-turbine/src/compiler.rs), making async boundary opcodes explicit on both Spawn and Await paths.
  - runtime/provider decision telemetry exists in [native_auto.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/native_auto.rs).
  - residency hooks exist in accelerate runtime.
- Blocking gap:
  - executable fusion group construction remains tied to bytecode accel-graph artifacts (though semantically candidate-group-gated and candidate-span-derived):
    - [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) still builds bytecode accel graph artifacts and maps semantic candidate spans onto graph nodes (`derive_semantic_fusion_groups_from_candidates`) as the execution-group boundary.
  - explicit spawned-task provider-handle lifetime/concurrency semantics remain incomplete in runtime/provider boundaries (no full Plan 7 closeout evidence yet).

### 7) Validation cadence (`met` for current slices)

- Latest executed gates:
  - `cargo fmt --all --check`
  - `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_project_source_index`
  - `cargo test -p runmat-core compile_input_resolves_wildcard_import_from_dependency_alias`
  - `cargo test -p runmat-core compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`
  - `cargo test -p runmat-core source_input_path_`
  - `cargo test -p runmat --lib resolve_script_input_`
  - `cargo test -p runmat-vm primary_compile_emits_explicit_spawn_instruction`
  - `cargo test -p runmat-vm primary_compile_interprets_async_call_and_await_via_semantic_value_lane`
  - `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`
  - `cargo test -p runmat-vm semantic_candidates_without_overlap_do_not_build_fusion_groups`
  - `cargo test -p runmat-vm fusion_group_semantic_span_filter_requires_full_group_coverage`
  - `cargo test -p runmat-vm fusion_group_semantic_span_filter_rejects_multi_candidate_union_coverage`
  - `cargo test -p runmat-vm primary_compile_records_semantic_fusion_metadata`
  - `cargo test -p runmat-vm semantic_candidates_build_fusion_groups_from_accel_graph_nodes`
  - `cargo test -p runmat-vm semantic_candidates_without_overlap_do_not_build_fusion_groups`
  - `cargo test -p runmat-vm primary_compile_omits_accel_graph_when_signals_exist_but_no_candidate_group`
  - `cargo test -p runmat-vm primary_compile_records_semantic_spawn_site_metadata`
  - `cargo test -p runmat-vm primary_compile_records_semantic_await_site_metadata`
  - `cargo test -p runmat-vm primary_compile_scopes_await_site_metadata_to_entrypoint_target`
  - `cargo test -p runmat-core --test fusion_regressions`
  - `cargo test -p runmat-core --test semicolon_suppression`
  - `cargo check --workspace`
  - `git diff --check`

## Current Conclusion

Objective is **not achieved**.

Highest-impact unresolved areas:

1. Plan 5 end-to-end composition wiring (CLI/core + resolver graph + source index).
2. Plan 7 shift from bytecode-first fusion planning to semantic/MIR/analysis-driven planning.
