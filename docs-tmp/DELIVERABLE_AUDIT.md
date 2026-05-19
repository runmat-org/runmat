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
  - undefined-variable semantic lowering now emits stable identifier `RunMat:UndefinedVariable` in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs), with direct identifier-contract coverage in [semantic_lowering.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/tests/semantic_lowering.rs) and VM control-flow coverage in [control_flow.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/control_flow.rs).
  - command-form semantic parsing now preserves zero-argument `clear`/`close`/`clc` command behavior in [command.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-parser/src/parser/command.rs), with parser ratchet coverage in [command_syntax.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-parser/tests/command_syntax.rs) and core command-control identifier-contract coverage in [command_controls.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/command_controls.rs).
  - command-form semantic parsing now also preserves zero-argument `pause` command behavior in [command.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-parser/src/parser/command.rs), with parser ratchet coverage `pause_without_arg_is_command_form` in [command_syntax.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-parser/tests/command_syntax.rs) and runtime interaction execution coverage through command form in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs).
  - import conflict semantic lowering now carries stable identifiers in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs): `RunMat:ImportAmbiguous` (ambiguous call/handle/import resolution) and `RunMat:ImportDuplicate` (duplicate imports), with VM/HIR identifier-contract coverage in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs) and [lowering_extras.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/tests/lowering_extras.rs).
  - VM object paren selector-plan validation now carries stable identifier coverage for additional edge contracts in [shared.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/call/shared.rs): out-of-bounds range dimensions (`RunMat:InvalidRangeSelectorDim`) and unsupported numeric selector value types (`RunMat:ObjectSelectorTypeUnsupported`).
  - VM cell aggregate member-assignment now carries stable identifier coverage for RHS shape mismatch in [cells.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/ops/cells.rs): `RunMat:CellMemberRhsShapeMismatch` (`assign_cell_member_rejects_shape_mismatch_cell_rhs`).
  - VM static-property missing-name semantic coverage in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs) now asserts stable `RunMat:UndefinedVariable` identifier (`unqualified_static_property_without_imports_errors`) instead of message-text proxy checks.
  - core compile-path integration now also ratchets these import identifier contracts in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs): `compile_input_reports_import_ambiguity_identifier` and `compile_input_reports_duplicate_import_identifier`.
  - core parser-stage failure integration coverage in [engine.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/engine.rs) now asserts explicit `RunError::Syntax` contracts (`test_parse_error_handling`, `test_invalid_syntax_handling`) instead of generic error acceptance.
  - core recovery-path integration coverage in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs) now pins the intentional malformed-input failure to parser stage via explicit `RunError::Syntax` assertion (`test_error_recovery_and_continued_execution`) instead of broad `is_err()` acceptance.
  - core execution-attempt accounting coverage in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs) now asserts exact `stats.total_executions == 3` for valid/invalid/recovery sequence in `test_error_recovery_and_continued_execution`, ratcheting deterministic execute-entry stats behavior (including parser-stage failures).
  - core workspace replace-import coverage now also asserts explicit stale-binding materialization failure contract in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs) (`workspace_state_roundtrip_replace_only`): `"Variable 'z' not found in workspace"` after replace-only import.
  - formatter compatibility lexer-error coverage in [repl.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/repl.rs) now asserts `Error` as the leading token for unterminated strings (`unterminated_string_is_error_token`) instead of generic substring matching.
  - async extension/policy semantic failures now carry explicit stable identifiers in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs) (`RunMat:AwaitExtensionDisabled`, `RunMat:AwaitContextInvalid`, `RunMat:SpawnExtensionDisabled`, `RunMat:SpawnLexicalCaptureUnsupported`) with core/HIR identifier-contract assertion coverage in [semantic_lowering.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/tests/semantic_lowering.rs), [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs), and [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
  - core async integration coverage now asserts spawned-handle consumption semantics directly in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs) (`test_spawn_handle_is_consumed_after_await`): first await succeeds with value readback, second await on same handle fails with stable runtime identifier `RunMat:AwaitOperandInvalid`.
  - runtime callback builtins now normalize unresolved external callback identities to `RunMat:UndefinedFunction` for both `cellfun` and `arrayfun` in [cellfun.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/cells/core/cellfun.rs) and [arrayfun.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/acceleration/gpu/arrayfun.rs), with direct unresolved external-handle coverage in builtin tests and core session diagnostic-path coverage in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs).
- Gap:
  - designed gaps still open (async/future/spawn runtime model, aggregate edge behavior, remaining selector-plan normalization).

### 4) Manifest-driven composition/entrypoints (`met`)

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
- Residual watchpoints:
  - Keep core/CLI/LSP resolver path tests in validation cadence to prevent drift back to per-consumer path heuristics.

### 5) Unified nominal class/builtin metadata (`met`)

- Evidence:
  - shared callable identity/fallback policy in [hir.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/hir.rs).
  - builtin semantics surface in [semantics.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-builtins/src/semantics.rs).
  - class-registry metadata behavior in [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-builtins/src/lib.rs) now has explicit tests for primitive nominal static method metadata (`double`/`single`/`logical` -> `zeros`) and parent-chain method lookup resolution through class metadata (`primitive_classes_expose_static_zeros_method_metadata`, `method_lookup_uses_parent_class_metadata_chain`).
  - class-registry inherited metadata lookup in [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-builtins/src/lib.rs) is now cycle-safe for both method and property traversal (`method_lookup_handles_parent_cycle`, `property_lookup_handles_parent_cycle`), with explicit inherited property-owner lookup coverage (`property_lookup_uses_parent_class_metadata_chain`).
  - runtime consumer coverage now includes class-metadata inheritance lookup through `exist(..., 'method')` in [exist.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/io/repl_fs/exist.rs) (`exist_method_uses_registered_class_metadata_including_inheritance`), asserting both direct and inherited method resolution via shared class metadata.
  - runtime property access builtins now have explicit inherited metadata behavior coverage:
    - [getfield.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/getfield.rs): `getfield_inherited_dependent_property_uses_parent_metadata`
    - [setfield.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/setfield.rs): `setfield_rejects_inherited_static_property_assignment`
  - this ratchets that parent-class property metadata controls dependent/static property behavior in runtime object access paths, not only direct-class definitions.
  - runtime object introspection now consumes inherited class-property metadata for `fieldnames` in [fieldnames.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/fieldnames.rs), with explicit inheritance coverage (`fieldnames_object_includes_inherited_class_properties`).
  - runtime handle-object introspection now also ratchets inherited class-property metadata for `fieldnames(handle)` in [fieldnames.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/fieldnames.rs) (`fieldnames_handle_object_includes_inherited_class_properties`), covering child+parent metadata plus target payload fields.
  - runtime nominal ancestry checks for `isa` in [isa.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/introspection/isa.rs) are now cycle-safe with explicit cyclic-parent metadata coverage (`isa_inheritance_walk_handles_parent_cycles`), reducing metadata traversal fragility.
  - VM object static-member resolution paths in [resolve.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/object/resolve.rs) now have explicit inherited metadata consumer coverage:
    - `load_static_member_resolves_inherited_static_property_value`
    - `store_member_updates_inherited_static_property_owner_slot`
    - `load_static_member_resolves_inherited_static_method`
  - this ratchets class-ref static member resolution/writeback to parent metadata ownership behavior instead of direct-class-only lookup assumptions.
  - VM object member protocol gating now consumes inheritance-aware method metadata in [shared.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/call/shared.rs), so `subsref`/`subsasgn` fallback checks are no longer limited to direct-class method maps.
  - explicit coverage now asserts inherited protocol-method detection for child classes:
    - `class_defines_member_subsref_includes_inherited_method_metadata`
    - `class_defines_member_subsasgn_includes_inherited_method_metadata`
  - VM object member resolution in [resolve.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/object/resolve.rs) now has end-to-end inherited protocol dispatch coverage for missing member read/write behavior:
    - `load_member_uses_inherited_subsref_for_missing_property`
    - `store_member_uses_inherited_subsasgn_for_missing_property`
  - this ratchets that child-class member fallback behavior executes inherited `subsref`/`subsasgn` handlers with concrete runtime outcomes, not just metadata-presence checks.
  - runtime constructor fallback dispatch in [dispatcher.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/dispatcher.rs) now consumes inheritance-aware constructor metadata lookup (`lookup_method(class, class)`) and enforces static/public constructor invocation policy.
  - explicit dispatcher coverage now ratchets both inherited-constructor dispatch and private/non-static constructor fallback behavior:
    - `constructor_fallback_uses_inherited_static_constructor_metadata`
    - `constructor_fallback_skips_private_or_non_static_constructor_methods`
  - runtime object construction hierarchy walk in [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/lib.rs) now guards parent traversal against class metadata cycles while applying inherited default-property initialization.
  - explicit runtime coverage (`new_object_builtin_handles_class_parent_cycles`) now ratchets deterministic constructor behavior for cyclic parent metadata graphs.
- Residual watchpoints:
  - Keep a grep check in cadence to prevent reintroduction of direct-class-only method/property dispatch:
    - `rg -n "cls\\.methods\\.get\\(|class_def\\.methods\\.get\\(" crates/runmat-runtime/src crates/runmat-vm/src`
  - Keep inheritance/cycle consumer ratchet tests in cadence across runtime+VM metadata consumers (`isa`, `fieldnames`, `getfield`/`setfield`, object member/static dispatch, constructor fallback).

### 6) Semantic-fact-driven accel/fusion (`open`)

- Evidence:
  - analysis facts exist in [hir.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/hir.rs) and [store.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-mir/src/analysis/store.rs).
  - fusion snapshot planner metadata now records MIR analysis fact counts plus MIR semantic fusion signal/candidate-group counts in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs), runtime emission path [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), and [types.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/types.rs).
  - semantic fusion signal/candidate-group extraction is now a VM compile artifact concern in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and both counts and explicit semantic candidate-group artifacts are carried on bytecode products in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs).
  - semantic candidate-group artifacts now include MIR-localized run metadata (`function`, `block`, `stmt_start`, `stmt_end`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived during bytecode compilation in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and surfaced in fusion snapshot semantic-candidate node labels in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - VM compile now gates accel-graph realization on semantic instruction-window artifacts (`semantic_instruction_windows.is_empty()`) in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), removing duplicated runtime gating by a second bytecode accel-capability scan and keeping semantic-window metadata as the primary executable-fusion admission signal.
  - compile-level coverage now includes logical-only candidate overlap omission (`primary_compile_omits_accel_graph_when_candidates_overlap_only_logical_ops`) in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), asserting that semantic candidate presence alone does not force accel graph artifacts when overlapping ops are non-accelerable.
  - semantic fusion-group node filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now also requires semantic candidate containment of node-mapped instruction spans, with partial-overlap rejection ratcheted by `semantic_candidates_with_partial_overlap_do_not_build_fusion_groups`.
  - fusion-group semantic span filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now also requires semantic candidate containment of grouped instruction spans (not overlap), aligning candidate/span filtering contracts across pre-gate, node mapping, and executable-group retention.
  - executable fusion-group span boundaries in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) are now derived from semantic candidate spans plus accel-capable bytecode instruction windows (`derive_semantic_fusion_instruction_windows`), with explicit split behavior at non-accel bytecode boundaries ratcheted by `semantic_candidate_instruction_windows_split_on_non_accel_ops`.
  - semantic instruction-window artifacts (instruction span + semantic kind hint) are now persisted directly on bytecode semantic fusion metadata in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs) / [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and fusion realization now consumes that precomputed metadata instead of re-deriving windows during accel-graph realization.
  - fusion planner metadata now carries semantic instruction-window counts end-to-end in [types.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/types.rs), with compile/runtime snapshot propagation in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/compile.rs) and [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs), plus snapshot summary diagnostics in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - fusion snapshot artifacts now include explicit semantic instruction-window nodes/decisions (`SemanticWindow`) in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs), with compile/runtime snapshot builders passing bytecode semantic instruction windows through from VM metadata.
  - fusion-group kind classification in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now derives directly from semantic instruction-window signal kinds, with regression coverage `semantic_window_kind_is_not_overridden_by_graph_category` ensuring accel-graph node-category mismatches do not override semantic classification.
  - semantic fusion-group shape classification in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now avoids accel-graph output-shape inference and carries `ShapeInfo::Unknown` at semantic planning time, reducing dependency on graph-derived shape metadata.
  - pre-gate coverage now includes builtin-call classification in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), with control/assertion rejection ratcheted as the primary negative contract (`semantic_candidate_accel_capability_gate_rejects_control_assert_builtin`) and display/stream sink rejection retained as secondary classification coverage (`semantic_candidate_accel_capability_gate_rejects_sink_builtins`).
  - bytecode fusion-group detection in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) is now semantically gated: `detect_fusion_groups` only runs when MIR semantic candidate-group count is non-zero.
  - VM compile now drops accel-graph artifacts entirely when no executable semantic-mapped fusion groups survive candidate/span filtering in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), reducing unused bytecode-graph artifact retention in semantic-candidate/no-executable-group states.
  - fusion snapshot emission now includes semantic candidate nodes/decisions both when bytecode fusion groups are empty and when they are present, and this semantic path no longer hard-requires accel graph presence in [snapshot.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/fusion/snapshot.rs).
  - runtime fusion-plan preparation now consumes semantic candidate-group counts at the VM/accelerate boundary in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) and [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs), with explicit transition diagnostics when semantic candidates exist but executable bytecode groups are absent.
  - runtime fusion-plan preparation in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs) now sanitizes compile-provided semantic groups against the live runtime accel graph before executable planning, including stale-node filtering, kind compatibility checks, bounded nearby-node recovery for empty mappings, and explicit drop of unresolved groups.
  - runtime sanitization now also validates span proximity for pre-mapped compile node IDs before retaining them (not just kind compatibility), so stale mapped node IDs outside group span are dropped and re-resolved via nearby-node recovery.
  - `runmat-accelerate` unit coverage now ratchets both sides of this runtime sanitization boundary:
    - `prepare_fusion_plan_recovers_empty_group_nodes_from_runtime_graph`
    - `prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_graph_is_too_far`
    - `prepare_fusion_plan_replaces_stale_mapped_nodes_using_runtime_span_recovery`
  - `runmat-vm` compile now preserves semantic-window fusion group scaffolding even when accel-node mapping yields zero nodes, by emitting semantic-window-derived groups with empty `nodes` lists in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) (`derive_semantic_fusion_groups_from_instruction_windows` fallback).
  - compile-level coverage now ratchets this fallback via `semantic_windows_fallback_to_empty_node_groups_when_mapping_drops_all_nodes`, reducing early compile-time drop dependence on accel-graph node mapping and deferring recovery/drop to runtime sanitization.
  - compile fusion-group derivation now also preserves unmapped windows in mixed mapping outcomes via [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) (`derive_semantic_fusion_groups_preserving_unmapped_windows`), so partially mapped window sets no longer drop unmapped windows at compile time.
  - compile-level coverage now ratchets this mixed mapping boundary via `semantic_windows_preserve_unmapped_windows_alongside_mapped_groups`.
  - VM bytecode now carries explicit semantic async/spawn metadata (`semantic_async_metadata`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived from MIR spawn sites in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs), and surfaced at interpreter setup in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs) so spawned-task transition semantics are explicit at runtime boundaries.
  - VM bytecode semantic async metadata now carries explicit MIR await-site inventory (`mir_await_site_count`, `mir_await_sites`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), derived from MIR await terminators in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - VM bytecode semantic async metadata now also carries an explicit runtime execution model contract (`SemanticAsyncRuntimeModel`) in [program.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/program.rs), currently recorded as `EagerValueLane` in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) and surfaced at interpreter startup diagnostics in [state.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/state.rs).
  - compile-level async metadata ratchets now assert this runtime-model contract in `primary_compile_records_semantic_spawn_site_metadata` and `primary_compile_records_semantic_await_site_metadata` in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - MIR await boundaries now lower through explicit bytecode `Instr::Await` handling in [core.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/compiler/core.rs), [instr.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/instr.rs), [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), and [compiler.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-turbine/src/compiler.rs), making async boundary opcodes explicit on both Spawn and Await paths.
  - VM runtime dispatch now also models explicit spawned-task value-lane semantics in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs): `Instr::Spawn` wraps payload values into task-handle records with explicit task IDs; `Instr::Await` pass-throughs non-task values for compatibility while validating spawned-task handle shape/ID and rejecting malformed or stale consumed handles with `RunMat:AwaitOperandInvalid`.
  - VM dispatch drop/overwrite boundaries now retire spawn-task IDs for dropped or replaced task-handle values in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) (including `Instr::Pop`, `Instr::ExitScope`, `Instr::StoreVar`, and `Instr::StoreLocal` replacement), reducing stale task-ID registry growth when handles are dropped without await.
  - Spawn-handle overwrite retirement preserves self-reassignment correctness (`t = t`) while still retiring IDs on true replacement, with runner-level await flow coverage in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) (`await_succeeds_after_spawn_handle_self_reassignment`).
  - core integration coverage now ratchets this runtime await contract in [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs) (`test_await_passes_through_non_spawn_operand_at_runtime`), including value-level readback assertion for `y = await(1)`.
  - core control-flow integration/engine coverage no longer relies on display-proxy assertions for unsupported behavior: [integration.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/integration.rs) (`test_control_flow_execution`) and [engine.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/engine.rs) (`test_execution_with_control_flow`) now assert successful control-flow execution and explicit readback values.
  - core engine execution-surface coverage no longer treats empty/whitespace input behavior as “success or arbitrary error text”: [engine.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/engine.rs) (`test_empty_input_handling`, `test_whitespace_only_input`) now assert successful execution with no runtime diagnostics.
  - core semicolon error-surface coverage in [semicolon_suppression.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/semicolon_suppression.rs) (`test_errors_always_shown`) now asserts stable undefined-variable identifier contracts (`RunMat:UndefinedVariable`) instead of “any error” proxy checks.
  - core async stdin interaction failure coverage in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs) (`pending_handler_returns_error`) now asserts stable runtime identifier `RunMat:interaction:AsyncHandlerError` instead of broad “any error” acceptance.
  - core async interaction coverage in [async_stdin.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/tests/async_stdin.rs) now also ratchets eager spawn-side interaction behavior in `spawn_of_async_function_triggers_pause_handler_before_await`, asserting that `pause` inside a spawned async function triggers the keypress handler at `spawn` time (before explicit `await`).
  - core async interaction coverage now also ratchets ordering semantics for multi-spawn input flows in `parallel_spawn_inputs_follow_spawn_order_not_await_order`: prompt events occur in spawn order (`first`, then `second`) even when await order is reversed (`await(t2)` then `await(t1)`), reinforcing eager spawn-side execution boundaries.
  - core async interaction coverage now also ratchets serial failure boundary behavior in `spawn_error_stops_later_spawn_from_running`: when first spawned async `input` interaction fails, execution reports stable input-wrapper identifier `RunMat:input:InteractionFailed` and later spawn prompts are not executed (only `first: ` observed).
  - core semantic try/catch binding coverage in [tests.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/tests.rs) (`try_catch_binding_uses_semantic_vm`) now asserts the exact bound `err.message` payload shape (`'boom'`) instead of substring matching.
  - semantic class attribute conflict diagnostics now expose stable identifier contracts in [ctx.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-hir/src/lowering/ctx.rs):
    - `RunMat:ClassPropertyAttributeConflict`
    - `RunMat:ClassMethodAttributeConflict`
  - VM semantic compile coverage now ratchets these identifier contracts in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs):
    - `class_property_attribute_conflicts_error`
    - `class_method_attribute_conflicts_error`
  - private object-property access failures now also carry stable identifier contracts across VM/runtime object access surfaces:
    - identifier: `RunMat:PropertyPrivateAccess`
    - VM object-resolve path: [resolve.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/object/resolve.rs)
    - runtime object-field builtins: [getfield.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/getfield.rs), [setfield.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/structs/core/setfield.rs)
  - VM semantic coverage now asserts this identifier contract in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs):
    - `classes_static_and_inheritance`
    - `classes_property_access_attributes`
  - VM basics indexing error coverage in [basics.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/basics.rs) no longer relies on message substring checks for key semantic failure paths; `fft_end_arithmetic_out_of_bounds_raises_error` and `scalar_slice_with_nonnumeric_selector_errors` now assert stable identifier contracts (`RunMat:IndexOutOfBounds`/`RunMat:SubscriptOutOfBounds` and `RunMat:UnsupportedIndexType`/`RunMat:SliceNonTensor`).
  - VM control-flow and indexing-property coverage now similarly ratchets identifier contracts for indexing/object-overload failures in [control_flow.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/control_flow.rs) and [indexing_properties.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/indexing_properties.rs):
    - `index_step_zero_mex` -> `RunMat:IndexStepZero`
    - `unsupported_cell_index_type_mex` -> `RunMat:CellIndexType`
    - `oop_negative_missing_subsref_mex` -> `RunMat:MissingSubsref`
    - `oop_negative_missing_subsasgn_mex` -> `RunMat:MissingSubsasgn`
  - Runtime/VM identifier surfaces were tightened to support these assertions:
    - colon zero-step failures now emit `RunMat:IndexStepZero` in [colon.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/builtins/array/creation/colon.rs)
    - missing object indexing overload dispatch now normalizes `UndefinedFunction` into `RunMat:MissingSubsref`/`RunMat:MissingSubsasgn` in [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/lib.rs), with corresponding VM object-index guard paths in [indexing.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/indexing.rs)
  - semantic fusion-window node mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now uses accel semantic tags (`Unary`/`Elementwise`/`Reduction`/`MatMul`/`Transpose`) instead of a category-only allowlist, with transpose inclusion ratcheted by `semantic_candidates_build_fusion_groups_from_transpose_nodes`.
  - semantic fusion-window node mapping now also enforces window-kind/tag compatibility (for example elementwise windows no longer absorb reduction-tagged nodes), ratcheted by `semantic_elementwise_window_excludes_reduction_nodes` and `semantic_reduction_window_accepts_reduction_nodes`.
  - semantic fusion-window node mapping now also preserves semantic-window realization when accel nodes are span-matched but untagged, reducing hard dependence on accel-graph semantic tag completeness while preserving explicit kind-mismatch filtering for tagged nodes.
  - semantic fusion-window node mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs) now also accepts accel nodes whose instruction spans cover semantic window spans only with bounded widening (±1 instruction), reducing fragility to small accel-graph span widening while preventing broad-node overlap coupling.
  - compile-level coverage now includes both sides of this boundary:
    - `semantic_windows_map_accel_nodes_that_cover_window_span`
    - `semantic_windows_reject_overly_wide_covering_node_spans`
  - semantic fusion-window node mapping now also tolerates small partial-overlap boundary drift (<=1 instruction on both boundaries) when strict contained/covering checks do not match, while preserving strict rejection for larger boundary shifts.
  - compile-level coverage now also ratchets both sides of this partial-overlap boundary:
    - `semantic_windows_map_accel_nodes_with_small_boundary_shift_overlap`
    - `semantic_windows_reject_partial_overlap_with_large_boundary_shift`
  - semantic fusion-window node mapping now also tolerates small disjoint span gaps (<=1 instruction) when strict overlap matching yields no candidate nodes, while preserving rejection for larger disjoint gaps.
  - compile-level coverage now also ratchets both sides of this disjoint-gap boundary:
    - `semantic_windows_map_accel_nodes_with_small_disjoint_gap`
    - `semantic_windows_reject_accel_nodes_with_large_disjoint_gap`
  - provider/runtime spawn-handle sharing policy is now explicit through [lib.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate-api/src/lib.rs) (`SpawnHandleConcurrency`) and enforced at VM spawn execution boundary in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), including nested capture traversal and explicit `RunMat:SpawnGpuHandleUnsupported` / `RunMat:SpawnProviderUnavailable` diagnostics.
  - VM dispatch coverage now explicitly ratchets nested-capture (`Value::Cell`, `Value::Closure`) and provider-unavailable policy failures at the spawn boundary in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) tests (`spawn_policy_rejects_nested_gpu_handles_in_cell_capture`, `spawn_policy_rejects_gpu_handles_captured_by_closure_values`, `spawn_policy_reports_provider_unavailable_for_gpu_handles`).
  - spawn GPU-handle policy traversal now also recurses through handle-object targets in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), with direct coverage `spawn_policy_rejects_gpu_handles_nested_in_handle_object_target`.
  - production providers now declare their spawn-sharing policy explicitly:
    - [simple_provider.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/simple_provider.rs) -> `SynchronizedMutation`
    - [provider_impl.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs) -> `SynchronizedMutation`
  - runtime/provider decision telemetry exists in [native_auto.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/native_auto.rs).
  - residency hooks exist in accelerate runtime, and VM residency clearing now recursively traverses nested runtime values (`Cell`/`Struct`/`Object`/`Closure`/`OutputList`) in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs) so nested handle replacement paths clear residency marks.
  - VM residency traversal now also recurses into `Value::HandleObject` targets for both clear and keep-set collection paths in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs), with cycle-safe visited-target guards to prevent recursive target loops.
  - residency coverage now includes handle-object target paths in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs):
    - `clear_value_releases_gpu_handles_nested_in_handle_object_target`
    - `clear_value_excluding_preserves_handles_referenced_in_handle_object_target`
  - VM overwrite residency clearing now preserves shared nested GPU handles via handle-aware exclusion (`clear_value_excluding`) in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs) and [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs), reducing premature de-residency across replacement paths that share handles.
  - VM residency clear paths now issue best-effort provider release (`AccelProvider::free`) for dropped GPU handles in [residency.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/residency.rs), with provider-backed coverage for dropped-handle release and shared-handle preservation.
  - VM `Instr::Pop` now routes dropped-value handle cleanup through residency/provider release with live-value exclusion in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), with provider-backed completion-flow coverage in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) (`pop_releases_stack_only_provider_handle`, `pop_preserves_provider_handle_when_still_live_in_vars`).
  - provider-backed spawn/await completion-boundary residency/release behavior is now explicitly covered in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) via `spawn_await_completion_releases_stack_only_provider_handle` and `spawn_await_completion_preserves_provider_handle_when_still_live_in_vars`.
  - provider-backed spawned-task drop paths without await are now explicitly covered in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) via `spawn_pop_releases_stack_only_provider_handle` and `spawn_pop_preserves_provider_handle_when_payload_still_live_in_vars`.
  - provider-backed spawned-task lifecycle coverage now also includes nested payload shapes in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs):
    - `spawn_pop_releases_nested_closure_captured_provider_handle`
    - `spawn_await_completion_releases_nested_output_list_provider_handle`
    - `spawn_await_completion_releases_nested_struct_provider_handle`
    - `spawn_await_completion_releases_nested_object_property_provider_handle`
    - `spawn_await_completion_preserves_nested_object_property_handle_when_alias_live`
    - `spawn_await_completion_releases_nested_cell_provider_handle`
    - `spawn_await_completion_preserves_nested_cell_handle_when_alias_live`
    - `spawn_await_completion_releases_nested_handle_object_target_provider_handle`
    - `spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live`
    - `spawn_pop_releases_nested_handle_object_target_provider_handle`
    - `spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live`
  - this extends release-semantics evidence beyond direct `GpuTensor` payloads to nested closure-capture, output-list, struct, object-property, cell, and handle-object-target task payloads.
  - VM `Instr::ExitScope` local-drop cleanup in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) now applies handle-aware live-value exclusion (stack + vars + remaining locals) before residency/provider release.
  - provider-backed runner coverage in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) now asserts both sides of this contract:
    - `exit_scope_releases_local_only_provider_handle`
    - `exit_scope_preserves_provider_handle_when_still_live_in_vars`
    - `exit_scope_releases_nested_handle_object_local_provider_handle`
    - `exit_scope_preserves_nested_handle_object_provider_handle_when_still_live_in_vars`
    - `await_rejects_spawn_task_handle_after_scope_exit_retires_id`
  - this closes a concrete shared-liveness bug class where exiting a local scope could previously free provider storage for handles still live in variable slots.
  - VM overwrite cleanup for `Instr::StoreVar` / `Instr::StoreLocal` in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) now applies handle-aware live-value exclusion across stack/vars/locals before residency/provider release.
  - provider-backed runner coverage in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) now asserts overwrite-path shared-liveness preservation:
    - `store_var_overwrite_preserves_provider_handle_when_shared_in_other_var`
    - `store_var_overwrite_preserves_provider_handle_when_shared_in_local`
    - `store_local_overwrite_releases_provider_handle_when_unaliased`
    - `store_local_overwrite_preserves_provider_handle_when_shared_in_var`
    - `store_local_overwrite_preserves_provider_handle_when_shared_in_other_local`
    - `store_var_overwrite_releases_nested_handle_object_provider_handle_when_unaliased`
    - `store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_var`
    - `store_var_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_local`
    - `store_local_overwrite_releases_nested_handle_object_provider_handle_when_unaliased`
    - `store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_var`
    - `store_local_overwrite_preserves_nested_handle_object_provider_handle_when_shared_in_other_local`
  - this closes a second concrete shared-liveness bug class where overwriting one slot could previously free provider storage for handles still referenced in another live slot.
  - VM spawn-task ID retirement for drop/overwrite paths in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) now checks alias liveness across stack/vars/locals before retiring IDs.
  - spawn-task ID extraction/retirement now traverses nested runtime values (including handle-object targets) in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs), with direct coverage:
    - `dropped_nested_spawn_task_handle_in_handle_object_retires_task_id`
    - `dropped_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live`
    - `replaced_nested_spawn_task_handle_in_handle_object_retires_task_id_when_unaliased`
    - `replaced_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live`
    - `replaced_nested_spawn_task_handle_in_local_slot_retires_with_excluded_local`
    - `replaced_nested_spawn_task_handle_in_local_slot_keeps_id_when_other_local_alias_live`
  - coverage now includes alias-preserving task-ID behavior in [mod.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/dispatch/mod.rs) (`dropped_spawn_task_handle_keeps_id_when_alias_still_live`) and runner-level `await` success after alias overwrite in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) (`await_succeeds_after_overwriting_one_spawn_handle_alias`).
  - runner-level alias-liveness coverage in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) now also includes local-slot alias overwrite flows:
    - `await_succeeds_after_overwriting_one_local_spawn_handle_alias`
    - `await_succeeds_after_overwriting_var_alias_when_local_spawn_handle_alias_live`
  - this closes a stale-ID bug class where dropping one alias could previously invalidate `await` on another still-live alias.
  - VM interpreter cancellation path now clears residency marks for live stack/variable GPU-handle values before returning `ExecutionCancelled`, and completion now clears stack-only handle residency while preserving both live-var and live-local aliases in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs), with direct coverage in `cancellation_clears_gpu_residency_for_live_values` and `completion_clears_stack_only_gpu_residency`.
  - completion/pop lifecycle coverage now explicitly includes local-alias preservation for spawned payload handles and nested handle-object targets in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs):
    - `spawn_pop_preserves_provider_handle_when_payload_still_live_in_locals`
    - `spawn_await_completion_preserves_nested_handle_object_target_handle_when_alias_live_in_locals`
    - `spawn_pop_preserves_nested_handle_object_target_handle_when_alias_live_in_locals`
  - provider-backed semantic spawned-workload lifecycle coverage now also exists in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs), executing compiled semantic function bodies (HIR->MIR->VM function bytecode) instead of synthetic interpreter-state setup:
    - `semantic_spawn_overwrite_releases_unaliased_provider_handle`
    - `semantic_spawn_overwrite_preserves_provider_handle_when_alias_retained`
    - `semantic_async_spawn_await_overwrite_unaliased_executes_with_scalar_output`
    - `semantic_async_spawn_await_overwrite_preserves_provider_handle_when_alias_retained`
  - semantic-function invocation now also clears non-output semantic temp-slot GPU handle residency/storage after output extraction in [runner.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/interpreter/runner.rs) (`clear_semantic_function_temp_residency`), preserving returned outputs plus runtime global/persistent roots as keep-set.
  - provider-backed async helper/callee unaliased release coverage now ratchets this semantic-invoker path directly in [spawn_semantic_lifecycle.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/spawn_semantic_lifecycle.rs):
    - `semantic_async_spawn_await_helper_overwrite_releases_unaliased_provider_handle`
    - `semantic_async_spawn_await_struct_helper_releases_unaliased_provider_handle`
    - `semantic_async_spawn_await_cell_helper_releases_unaliased_provider_handle`
    - `semantic_async_spawn_multi_output_helper_unrequested_handle_releases`
    - `semantic_async_spawn_varargout_helper_unrequested_handle_releases`
    - `semantic_async_spawn_varargout_nested_unrequested_handle_releases`
  - semantic async lifecycle coverage now also includes multi-outstanding spawn-task flows in one semantic function frame (including out-of-order await):
    - `semantic_async_spawn_parallel_await_keeps_retained_handle_and_releases_dropped_handle`
    - `semantic_async_spawn_parallel_await_releases_both_unaliased_handles`
  - these ratchets assert per-handle independence across two concurrent spawn handles (`t1`/`t2`), including retained-handle preservation and dropped-handle cleanup through residency/provider-release surfaces.
  - Fusion materialized-store write paths now also consume handle-aware exclusion clearing in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/fusion.rs), extending shared-handle preservation beyond interpreter dispatch overwrite hooks.
  - Fusion materialized-store shared-handle preservation is now directly covered by `fusion_writeback_preserves_shared_gpu_handles` in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/accel/fusion.rs).
  - runtime gather/retry GPU recursion now includes `Value::Closure` captures and `Value::OutputList` entries in [dispatcher.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-runtime/src/dispatcher.rs), with explicit nested-provider-unavailable identifier coverage.
- Blocking gap:
  - executable fusion groups now preserve semantic-window scaffolding across both zero-mapping and partial-mapping compile outcomes, and runtime sanitization now revalidates both kind and span for pre-mapped nodes before recovery/drop; remaining gap is further minimizing compile-time node assignment heuristics in favor of runtime-only assignment.
  - spawned-task provider-handle policy now has explicit provider declarations plus VM enforcement, and provider-backed release semantics are covered for residency-clear/drop paths (including `Pop`) plus spawn/await completion/cancellation and compiled semantic spawn-overwrite lifecycle flows; semantic async spawn/await lifecycle coverage now includes unaliased helper/callee release behavior via semantic invoker path for direct, struct-nested, cell-nested, multi-output, `varargout`, nested-unrequested `varargout`, and multi-outstanding two-task await flows, with core interaction tests now ratcheting explicit serial eager spawn execution/order/failure boundaries.

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
  - `cargo test -p runmat-vm expansion_on_non_cell_errors -- --nocapture`
  - `cargo test -p runmat-vm mixed_range_end_assign_shape_mismatch_error -- --nocapture`
  - `cargo test -p runmat-vm semantic_windows_reject_disjoint_gap_at_compile_mapping_stage -- --nocapture`
  - `cargo test -p runmat-vm semantic_windows_reject_partial_overlap_at_compile_mapping_stage -- --nocapture`
  - `cargo test -p runmat-vm semantic_windows_reject_covering_node_span_at_compile_mapping_stage -- --nocapture`
  - `cargo test -p runmat-vm semantic_windows_reject_accel_nodes_with_large_disjoint_gap -- --nocapture`
  - `cargo test -p runmat-vm semantic_windows_reject_overly_wide_covering_node_spans -- --nocapture`
  - `cargo test -p runmat-vm semantic_windows_ -- --nocapture`
  - `cargo test -p runmat-accelerate prepare_fusion_plan_ -- --nocapture`
  - `cargo test -p runmat-accelerate sanitize_runtime_groups -- --nocapture`
  - `cargo check --workspace`
  - `git diff --check`

- Additional contract-hardening ratchet:
  - Converted two VM tests in [functions.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/tests/functions.rs) from message-substring assertions to identifier assertions:
    - `expansion_on_non_cell_errors` now asserts `RunMat:ExpandError`.
    - `mixed_range_end_assign_shape_mismatch_error` now asserts `RunMat:ShapeMismatch`.
  - Purpose: reduce display-message coupling and keep runtime behavior tests pinned to stable identifier contracts.
  - Core entrypoint-resolution error coverage in [run.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-core/src/session/run.rs) now also asserts runtime identifier contract (`RunMat:EntrypointResolveFailed`) instead of matching rendered error text.
  - Manifest/composition tests in [project_manifest.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-config/tests/project_manifest.rs) now assert typed error variants/fields (`ProjectManifestLoadError`, `ProjectSourceIndexError`, `ProjectEntrypointResolveError`, `ProjectCompositionError`) rather than formatted error strings.

- Additional Plan 7 heuristic-reduction ratchet:
  - Removed compile-time disjoint-gap, partial-overlap, and covering-span fallbacks in semantic-window node mapping in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs).
  - Compile-time mapping now rejects disjoint, partial-overlap, and covering graph/window spans and leaves that reconciliation to runtime fusion sanitization (`prepare_fusion_plan` -> `sanitize_runtime_groups`) in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs).
  - Updated compile coverage to assert all compile-stage boundaries (`semantic_windows_reject_disjoint_gap_at_compile_mapping_stage`, `semantic_windows_reject_partial_overlap_at_compile_mapping_stage`, `semantic_windows_reject_covering_node_span_at_compile_mapping_stage`).
  - Tightened missing-tag compile mapping fallback to category-compatible matching only (instead of unconditional acceptance), with direct regression coverage: `semantic_windows_without_tags_reject_category_mismatch`.
  - Compile now emits fusion-group scaffolding from semantic instruction windows without compile-time node assignment, delegating node reconciliation to runtime plan preparation.
  - Compile-time accel-node mapping helper path used for historical fusion-node assignment is now test-only, removing that path from production compilation.
  - Added explicit boundary coverage in [compile.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-vm/src/bytecode/compile.rs): `primary_compile_emits_semantic_window_scaffolds_and_runtime_plan_reconciles_nodes` asserts compile-time groups remain node-empty and runtime `prepare_fusion_plan(...)` performs executable node reconciliation.

- Additional Plan 7 runtime sanitization ratchet:
  - Tightened runtime `sanitize_runtime_groups` span recovery in [fusion.rs](/Users/nallana/Source/runmat-acc-2/runmat/crates/runmat-accelerate/src/fusion.rs) from overlap-or-touch (`<=1` disjoint gap) to overlap-only, then to contained-span-only.
  - Updated runtime sanitization coverage to assert contained-span recovery, explicit rejection of covering spans, and no stale mapped-node remap:
    - `prepare_fusion_plan_recovers_empty_group_nodes_from_contained_runtime_span`
    - `prepare_fusion_plan_rejects_stale_mapped_nodes_without_runtime_remap`
    - `prepare_fusion_plan_rejects_empty_group_nodes_when_runtime_node_covers_group_span`

## Current Conclusion

Objective is **not achieved**.

Highest-impact unresolved areas:

1. Plan 7 shift from bytecode-first fusion realization to semantic/MIR/analysis-driven planning, including remaining accel-graph realization dependency and async spawn-provider lifecycle evidence gaps.
