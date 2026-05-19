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
  - async extension/policy semantic failures now carry explicit stable identifiers in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs) (`RunMat:AwaitExtensionDisabled`, `RunMat:AwaitContextInvalid`, `RunMat:SpawnExtensionDisabled`, `RunMat:SpawnLexicalCaptureUnsupported`) with core/HIR identifier-contract assertion coverage in [semantic_lowering.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/tests/semantic_lowering.rs), [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs), and [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - runtime callback builtins now normalize unresolved external callback identities to `RunMat:UndefinedFunction` for both `cellfun` and `arrayfun` in [cellfun.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/cells/core/cellfun.rs) and [arrayfun.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/acceleration/gpu/arrayfun.rs), with direct unresolved external-handle coverage in builtin tests and core session diagnostic-path coverage in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
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
  - discovered composition loading/root-package verification is now shared internally in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `discover_project_composition_from`, reducing duplicated composition-load seams across `resolve_named_entrypoint_from` and `discover_project_symbols_from`.
  - shared source-input path resolver now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `resolve_project_source_input_from`, centralizing direct path acceptance, optional `.m` inference, and single-segment named-entrypoint fallback.
  - shared project-symbol discovery now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `discover_project_symbols_from`, centralizing discovery + composition load + root dependency alias mapping + symbol-set construction (`raw`, `package-qualified`, and `alias-qualified` forms).
  - shared source-name start-path derivation for project-symbol discovery now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `discover_project_symbols_from_source_name(source_name, cwd)`, removing duplicate source-name discovery heuristics from core compile.
  - shared known-symbol fallback policy now exists in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `discover_known_project_symbols_from_source_name(source_name, cwd)` and is consumed by active core/CLI/LSP source-context lowering paths:
    - [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs)
    - [bytecode.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/bytecode.rs)
    - [analysis.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-lsp/src/core/analysis.rs)
  - module/function entrypoint resolution now uses source-index qualified-name matching (`build_project_source_index`) instead of a direct dotted-path file heuristic, including support for class-folder targets (`+pkg/@ClassName/method.m`).
  - CLI run-path integration in [script.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/script.rs) now resolves manifest entrypoint names through the shared discovered-entrypoint resolver for both path targets and module/function targets.
  - config-layer composition graph loading remains in [project.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/src/project.rs) via `build_project_composition_graph`, including local path dependency manifest loading and per-package source indexes.
  - core request-path source loading in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs) now resolves simple named path inputs through the shared discovered-entrypoint resolver before source read.
  - core request-path source loading now also infers `.m` for unresolved file-style path inputs before entrypoint-name fallback in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), matching Plan 5 path-target inference behavior.
  - core request-path source loading now consumes shared config-layer source-input resolution (`resolve_project_source_input_from`) in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs) instead of owning a duplicate path/entrypoint heuristic chain.
  - CLI script target resolution now applies the same `.m` inference before named-entrypoint fallback in [script.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/script.rs).
  - CLI script target resolution now consumes shared config-layer source-input resolution (`resolve_project_source_input_from`) in [script.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/script.rs) instead of owning a duplicate path/entrypoint heuristic chain.
  - CLI benchmark target resolution now consumes shared config-layer source-input resolution (`resolve_project_source_input_from`) in [benchmark.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/benchmark.rs), so benchmark execution now shares manifest-driven named-entrypoint and module/function target resolution behavior.
  - core compile/lowering path now discovers composition source-index symbols from source context and passes them into HIR lowering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs).
  - core interactive `input()` eval-hook lowering now also consumes shared source-context known-symbol discovery in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), reducing divergence between main compile and eval-hook lowering paths.
  - LSP document analysis now also threads source-context project symbols into lowering through shared config discovery in [analysis.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-lsp/src/core/analysis.rs), with backend/wasm URI->source-path wiring in [backend.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-lsp/src/backend.rs) and [exports.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-lsp/src/wasm/exports.rs).
  - core compile source-context symbol discovery now gates path-like source names on local path existence in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), preventing remote/virtual source-name contexts from inheriting local composition symbols.
  - core compile source-context symbol discovery also blocks colon-style remote/virtual source names (for example `remote:main.m`) when they do not resolve to existing local paths in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), with explicit integration coverage in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - core integration coverage now includes this guard boundary via `compile_input_does_not_leak_local_project_symbols_for_remote_source_names` in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - CLI bytecode-emission lowering now also discovers composition source-index symbols from source context via shared config discovery (`discover_project_symbols_from_source_name`) in [bytecode.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/bytecode.rs), and script `--emit-bytecode` now supplies the resolved source name in [script.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/script.rs).
  - CLI bytecode command coverage now includes explicit source-context symbol discovery and wildcard-import emitted-bytecode visibility tests in [bytecode.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/bytecode.rs).
  - CLI bytecode symbol discovery now explicitly gates on existing local source paths in [bytecode.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/bytecode.rs), avoiding local-project symbol bleed for virtual/nonexistent source names.
  - CLI bytecode symbol discovery now also blocks colon-style remote/virtual source names when they do not resolve to existing local paths in [bytecode.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-cli/src/commands/bytecode.rs), with command-level coverage for this guard boundary.
  - core composition symbol discovery now includes root dependency alias-qualified symbols (`alias.symbol`) derived from composition graph root dependency mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), enabling wildcard import resolution through manifest dependency aliases.
  - core composition symbol discovery now consumes shared config-layer project-symbol discovery (`discover_project_symbols_from`) in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs) instead of owning a duplicate composition/source-index traversal loop.
  - HIR wildcard import resolution now uses project source-index symbol candidates for call/function-handle target resolution in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs) via lowering-context symbol inputs from [lowering_context.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering_context.rs).
  - core integration coverage now includes wildcard import resolution through project source-index symbols (`compile_input_resolves_wildcard_import_from_project_source_index`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - core integration coverage now includes wildcard import resolution through dependency aliases (`compile_input_resolves_wildcard_import_from_dependency_alias`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - core integration coverage now includes dependency-alias wildcard import resolution for function-handle lowering (`compile_input_resolves_function_handle_from_dependency_alias_wildcard_import`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - dependency-alias wildcard function-handle coverage now asserts exact alias-qualified lowering identity (`CreateExternalFunctionHandle("statsdep.summarize")`) in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - config integration coverage now includes shared project-symbol discovery alias qualification (`discover_project_symbols_includes_dependency_alias_qualified_names`) in [project_manifest.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/tests/project_manifest.rs).
  - config integration coverage now includes source-name start-path shared discovery (`discover_project_symbols_from_source_name_uses_cwd_for_plain_name`) in [project_manifest.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/tests/project_manifest.rs).
  - config integration coverage now includes shared source-input path resolution for `.m` inference and named-entrypoint fallback (`resolve_project_source_input_from_infers_m_extension`, `resolve_project_source_input_from_resolves_named_entrypoint`) in [project_manifest.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/tests/project_manifest.rs).
  - config integration coverage now also asserts explicit pass-through and error-contract behavior for shared source-input resolution (`resolve_project_source_input_from_returns_plain_candidate_when_name_is_not_entrypoint`, `resolve_project_source_input_from_reports_named_entrypoint_resolution_errors`) in [project_manifest.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/tests/project_manifest.rs).
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
  - VM compile now requires semantic candidate containment of accel-capable bytecode instruction spans before accel-graph construction in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), reducing no-op graph construction and boundary-overlap false positives when candidate spans map only partially to instructions.
  - compile-level coverage now includes logical-only candidate overlap omission (`primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops`) in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), asserting that semantic candidate presence alone does not force accel graph artifacts when overlapping ops are non-accelerable.
  - semantic fusion-group node filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now also requires semantic candidate containment of node-mapped instruction spans, with partial-overlap rejection ratcheted by `semantic_candidates_with_partial_overlap_do_not_build_fusion_groups`.
  - fusion-group semantic span filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now also requires semantic candidate containment of grouped instruction spans (not overlap), aligning candidate/span filtering contracts across pre-gate, node mapping, and executable-group retention.
  - executable fusion-group span boundaries in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) are now derived from semantic candidate spans plus accel-capable bytecode instruction windows (`derive_semantic_fusion_instruction_windows`), with explicit split behavior at non-accel bytecode boundaries ratcheted by `semantic_candidate_instruction_windows_split_on_non_accel_ops`.
  - fusion-group kind classification in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now derives directly from semantic instruction-window signal kinds, with regression coverage `semantic_window_kind_is_not_overridden_by_graph_category` ensuring accel-graph node-category mismatches do not override semantic classification.
  - semantic fusion-group shape classification in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now avoids accel-graph output-shape inference and carries `ShapeInfo::Unknown` at semantic planning time, reducing dependency on graph-derived shape metadata.
  - pre-gate coverage now includes builtin-call classification in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), asserting accel-capable builtin overlap acceptance (`sum`) and explicit non-accelerable builtin overlap rejection (`assert`, `disp`, `fprintf`).
  - bytecode fusion-group detection in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) is now semantically gated: `detect_fusion_groups` only runs when MIR semantic candidate-group count is non-zero.
  - VM compile now drops accel-graph artifacts entirely when no executable semantic-mapped fusion groups survive candidate/span filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), reducing unused bytecode-graph artifact retention in semantic-candidate/no-executable-group states.
  - fusion snapshot emission now includes semantic candidate nodes/decisions both when bytecode fusion groups are empty and when they are present, and this semantic path no longer hard-requires accel graph presence in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - runtime fusion-plan preparation now consumes semantic candidate-group counts at the VM/accelerate boundary in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) and [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs), with explicit transition diagnostics when semantic candidates exist but executable bytecode groups are absent.
  - VM bytecode now carries explicit semantic async/spawn metadata (`semantic_async_metadata`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived from MIR spawn sites in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and surfaced at interpreter setup in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) so spawned-task transition semantics are explicit at runtime boundaries.
  - VM bytecode semantic async metadata now carries explicit MIR await-site inventory (`mir_await_site_count`, `mir_await_sites`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived from MIR await terminators in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - MIR await boundaries now lower through explicit bytecode `Instr::Await` handling in [core.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/compiler/core.rs), [instr.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/instr.rs), [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), and [compiler.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-turbine/src/compiler.rs), making async boundary opcodes explicit on both Spawn and Await paths.
  - provider/runtime spawn-handle sharing policy is now explicit through [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate-api/src/lib.rs) (`SpawnHandleConcurrency`) and enforced at VM spawn execution boundary in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), including nested capture traversal and explicit `RunMat:SpawnGpuHandleUnsupported` / `RunMat:SpawnProviderUnavailable` diagnostics.
  - VM dispatch coverage now explicitly ratchets nested-capture (`Value::Cell`, `Value::Closure`) and provider-unavailable policy failures at the spawn boundary in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) tests (`spawn_policy_rejects_nested_gpu_handles_in_cell_capture`, `spawn_policy_rejects_gpu_handles_captured_by_closure_values`, `spawn_policy_reports_provider_unavailable_for_gpu_handles`).
  - production providers now declare their spawn-sharing policy explicitly:
    - [simple_provider.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/simple_provider.rs) -> `SynchronizedMutation`
    - [provider_impl.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs) -> `SynchronizedMutation`
  - runtime/provider decision telemetry exists in [native_auto.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/native_auto.rs).
  - residency hooks exist in accelerate runtime, and VM residency clearing now recursively traverses nested runtime values (`Cell`/`Struct`/`Object`/`Closure`/`OutputList`) in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs) so nested handle replacement paths clear residency marks.
  - VM overwrite residency clearing now preserves shared nested GPU handles via handle-aware exclusion (`clear_value_excluding`) in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs) and [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs), reducing premature de-residency across replacement paths that share handles.
  - VM residency clear paths now issue best-effort provider release (`AccelProvider::free`) for dropped GPU handles in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs), with provider-backed coverage for dropped-handle release and shared-handle preservation.
  - VM `Instr::Pop` now routes dropped-value handle cleanup through residency/provider release with live-value exclusion in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), with provider-backed completion-flow coverage in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) (`pop_releases_stack_only_provider_handle`, `pop_preserves_provider_handle_when_still_live_in_vars`).
  - VM interpreter cancellation path now clears residency marks for live stack/variable GPU-handle values before returning `ExecutionCancelled`, and completion now clears stack-only handle residency while preserving live-var handles in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs), with direct coverage in `cancellation_clears_gpu_residency_for_live_values` and `completion_clears_stack_only_gpu_residency`.
  - Fusion materialized-store write paths now also consume handle-aware exclusion clearing in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/fusion.rs), extending shared-handle preservation beyond interpreter dispatch overwrite hooks.
  - Fusion materialized-store shared-handle preservation is now directly covered by `fusion_writeback_preserves_shared_gpu_handles` in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/fusion.rs).
  - runtime gather/retry GPU recursion now includes `Value::Closure` captures and `Value::OutputList` entries in [dispatcher.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/dispatcher.rs), with explicit nested-provider-unavailable identifier coverage.
- Blocking gap:
  - executable fusion groups now use semantic/bytecode instruction-window boundaries and semantic kind/shape classification first, but fusion realization still depends on accel-graph node mapping after boundary derivation in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - spawned-task provider-handle policy now has explicit provider declarations plus VM enforcement, and provider-backed release semantics are covered for residency-clear/drop paths (including `Pop`) plus cancellation/completion cleanup, but there is not yet full end-to-end evidence across spawned workload completion boundaries that exercise provider-level release via real spawn/await execution flows.

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
