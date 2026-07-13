# FEA Geometry Agent Change Progress

## Summary

- Overall status: 248 implementation slices completed.
- Estimated remaining: <1%. Remaining work is final guided-flow acceptance consolidation, the requirement-by-requirement completion audit, and any last high-signal boundary fixes it identifies. The largest remaining risk is proving the user-facing guided FEA flow completes across the full supported physics profile catalog through the same run/check/variable/figure/FEA/geometry/runtime/context-attachment surface without side-channel APIs, stress-specific defaults, ambiguous model-facing verbs, stale generated contracts, cross-boundary naming drift, stale local-runtime spelling, stale check/readiness wording, structural-first UX assumptions, stale generated wasm package surfaces, private harness fixture drift, or UI-only result/check/selection side paths. The composed harness now has generic runtime-surface evidence for every public catalog profile class: linear static structural, modal, transient structural, nonlinear structural, thermal, thermo-mechanical, electro-thermal, electromagnetic, acoustic, CFD, CHT, and FSI, and now also proves capability-driven typed study creation from geometry across every current public profile id. Runtime FEA capabilities now expose the Rust-owned rich `physicsProfiles` catalog through wasm/generated TS and the desktop agent pane consumes runtime-provided profile families before falling back to generated constants. Typed `.fea` study creation now requires an explicit `model_profile`, validates it against the Rust-owned profile/run-kind map, persists the derived run kind, has all-current-profile authoring/readiness proof across structural static, transient structural, nonlinear structural, modal, thermal, electromagnetic, acoustic, CFD steady/transient, thermo-mechanical, electro-thermal, CHT, and FSI profiles, and the raw harness create-study schema now tells the model to choose `model_profile` from `fea_capabilities.physicsProfiles` instead of assuming linear static structural analysis. Desktop geometry-to-study creation now requires an explicit visible physics profile selection and passes that profile into the runtime create operation instead of relying on a hidden structural default, with CAD-start UI coverage proving every public profile option is exposed before creation and a coupled FSI profile crosses the creation callback. Desktop geometry-start FEA choices now canonicalize runtime-provided physics family labels so `cfd`, `CFD`, `coupled`, and `coupled physics` capability payloads all expose the correct Flow/Coupled guided choices instead of dropping supported non-structural families. Target/change-plan guided pane docs now use boundary/support region language rather than treating fixed structural supports as the generic workflow step. Desktop guided structural setup now uses constraint/boundary region wording instead of fixed-area wording until a fixed boundary condition is explicitly chosen. Adaptive mesh append/postprocess now requires persisted `analysis_profile` and `run_kind` metadata instead of defaulting missing artifacts to linear static structural refinement. Runtime `.fea` authoring operation errors for driving-condition updates/removals now report `driving_conditions` even though the durable YAML storage key remains `loads`, so model-facing failures no longer leak old load-condition vocabulary. The older runtime `fea.authorStudy` helper now uses boundary-condition/driving-condition terminology at its public intent/evidence boundary and records structural force evidence only when the active default driver is actually a structural force. Generic FEA variable summaries are now explicitly proven to expose stable run, session, and field identity across that Phase 8 matrix, desktop `select_run` now honors the model-visible `session_id` contract, and desktop generic `variables`, `variable`, and `figures` now honor explicit `session_id`/`run_id` selectors when inspecting a non-selected FEA run; `show_figures` is now documented and domain-tested to show a non-selected FEA run figure by stable figure id and return/pin the owning run context. The `execute` tool schema now has explicit test evidence that `target=inline` requires `code` and `target=file` requires `path`, matching the `.fea` execute-file dispatch contract. General workspace primitives now consistently distinguish `open_path` from `select_path`: `open_path` opens visibly without selecting the active target, while `select_path` makes the artifact active for subsequent context, and `copy` now reports explicit `bytes_copied` while preserving parent-dir and undo-plan semantics. Report generation has refreshed non-structural acoustic generic-surface evidence, guided workflow step IDs no longer encode fixed/load structural assumptions, structural guided-pane copy now uses driver terminology instead of loaded-area terminology, composed fork-flow fixtures now use driving-condition/driver naming, model-facing and runtime study edit operations now use driving-condition terminology, model-visible tool/catalog tests now explicitly reject retired FEA-specific run/result/fork/load-condition tools as well as legacy geometry tools, the desktop FEA agent context exposes driving-condition counts/blockers, Rust-owned readiness checks and guided-pane routing are profile-aware across the current public physics profile catalog, model-visible driving-condition tools now accept full-family source/driver types and units, typed material/media authoring now accepts thermal, electromagnetic, acoustic, and fluid properties without requiring structural Young's modulus/Poisson ratio, generated TypeScript contracts and the desktop study review now carry/display non-structural material/media summaries instead of falling back to mechanical-only review copy, the generic harness runtime adapters plus desktop and local CLI runtime/replay mocks no longer use structural displacement/stress as their default FEA result proof, unknown-profile guided copy now stays neutral instead of inheriting structural fixed/load language, unsupported/no-profile study review output suggestions no longer fall back to linear static structural outputs, the TS package now exports a Rust-owned and generated supported physics profile catalog instead of maintaining a hand-written duplicate, stale generated TS operation artifacts and local wasm web package wrappers no longer advertise removed result/load-condition surfaces, desktop and local agent-facing FEA `run_kind` values now use the shared `fea-study` contract instead of an underscore variant, local agent check artifact kinds now use finite-element wording instead of `fea_study`, agent-facing FEA check blockers/warnings/diagnostics now normalize stale load-condition wording to driving-condition wording, UX/target/change-plan docs now state full-family support as a construction invariant rather than a structural workflow extension, model-facing geometry guidance now includes the full session lifecycle including clear-selection and close-session tools, the generic agent `check` bridge now supports both `.fea` studies and `.m` RunMat scripts instead of treating finite element validation as the only checkable artifact, the FEA run orchestrator and agent runtime provider now also reach `.fea` validation through that same generic check dispatcher, the guided FEA pane now exposes an artifact timeline derived from the same study/geometry/run snapshots sent to the model and has every-current-profile visible step-label regression coverage, cache-aware model-frame projection now has non-structural thermal/flow proof that live physics state and render images stay out of the cacheable prefix and late in the frame, FEA current-state projection now enters context through a bounded domain context provider, and the stable FEA/geometry workflow protocol now enters the cacheable prefix through that same domain provider rather than generic tool guidance. The browser agent client now proves the guided runtime bridge preserves structural, thermal, electromagnetic, acoustic, CFD, CHT-style, and FSI-style driver payloads through the same study-operation channel, private harness capability fixtures now mirror the generated full-family profile catalog including CHT/FSI default output fields, the desktop FEA results pane now uses the generic workspace materializer for lazy field preview paging instead of calling the FEA field client directly, the desktop FEA study review surface now runs validation through a generic runtime check dispatcher instead of calling the FEA check client directly, agent runtime inspection session resolution and variable/figure materialization now live in the bridge domain helper instead of runtime-provider UI code, geometry session render requests now normalize bounded visibility and section state for model-render requests, live geometry surfaces now consume model-driven camera view presets, hidden/isolated owner visibility, and section state through the Rust presentation contract, with CPU-side section clipping in `runmat-plot` rather than a React-only workaround, the result-surface audit found no current 3D result hit-test source bypassing shared prompt attachments and confirmed current result field selections use the shared bounded attachment path, Phase 9 raw-topology/context-cost retirement is covered at the tool catalog, geometry-session tool, FEA context, mesh/solver-topology compaction, and geometry-heavy FEA context bounding boundaries, shared scene/file/image/run/figure prompt context attachments now have desktop chip/context wiring plus model-frame projection evidence, and the final naming audit now finds no active standalone `tet`/`tets` shorthand identifiers in the audited FEA/meshing/runtime/plot or private harness/desktop surfaces.
- Current focus: turn the guided FEA pane into a workflow-driven surface while preserving the composed desktop, harness, wasm, and runtime acceptance paths.
- Completion gate: do not mark the remaining goal complete until user-created scene/file/image/run/figure context can enter the prompt attachment strip as removable chips, model-driven camera/visibility/selection/section mutations converge on the same current scene state, chip removal clears the matching shared context, that bounded dynamic state projects into the model frame, and selected context can drive typed `.fea` edits plus generic `check`, `execute`, `variables`, and `figures` without requiring the user or model to type fragile raw selectors.
- Named remaining acceptance slice: close the shared scene/context selection loop
  across STEP/CAD, `.fea`, mesh, and result surfaces. The slice is complete only
  when product selections become prompt chips, chips become bounded dynamic
  model context, and that context is proven to drive typed study edits and the
  generic run/check/result path. Result-field selections now publish through the
  shared scene-selection model and mesh/result selections are model-visible as
  first-class attachment kinds; a composed host-parity path now proves a
  user-created geometry selection chip can drive geometry-session selection,
  typed `.fea` region/constraint edits, generic `check`, and generic `execute`.
  The desktop FEA study surface now publishes directly into that shared scene
  selection contract and the stale FEA-only selection DTO/prompt adapter is
  removed. Mesh-backed scene picks without named regions now keep `mesh:*`
  selectors and enter prompt context as mesh selections rather than generic
  geometry. `.fea` review-tree region selections now preserve additive
  shift-click intent through the same shared scene-selection mode used by CAD
  preview picks, so multiple rendered study regions can merge into one prompt
  context attachment instead of becoming unrelated prompt prose.
  Prompt chip removal now has explicit next-turn projection evidence: removing
  a scene-backed chip clears shared scene selection and the filtered attachment
  set sent to `buildAgentTurnContext` no longer contains the removed selector.
  Harness context projection now explicitly proves additive/multi-entity scene
  selections stay as one bounded structured `prompt_context_attachments` item
  with multiple selectors, rather than being flattened into prompt prose or
  split into unrelated attachments. The composed host-parity selected-region
  flow now also carries a two-entity additive geometry selection attachment into
  the actual model request before the model chooses the intended entity and
  drives geometry-session selection, typed `.fea` region/constraint edits,
  generic `check`, and generic `execute`. Model-driven `geometry_select` now
  also accepts multi-selector selections and returns structured selected
  entities in the session snapshot; the desktop agent scene-state bridge carries
  those entities into the same shared editor scene-selection contract used by
  user picks and prompt chips. Result-field selections in the desktop FEA
  results pane now also preserve shift-click additive intent, so result
  selections use the same multi-entity merge contract as geometry, mesh, and
  `.fea` review selections instead of being a result-only single-pick path.
  Prompt-context attachment state now lives in a focused agent hook
  instead of the broad agent pane shell. Completed geometry-session tool
  observations now emit typed scene-state events, and the editor shell converts
  model-selected geometry into the same shared `EditorSceneSelectionSnapshot`
  contract used by user clicks and prompt chips. Shared scene selection now also
  drives live geometry-scene selected-region presentation through the editor
  shell, geometry preview, `.fea` preview, `FigureCanvas`, and runtime canvas
  adapter. Model-driven camera view presets, hidden/isolated owner visibility,
  and section state now also flow through the Rust `GeometryScenePresentation`
  contract into live product surfaces, with renderer-side section clipping
  instead of a UI-only workaround. The result-surface audit found no current
  3D result hit-test
  source bypassing this path; richer result-scene picking should use the same
  contract when a hit-testable result scene exists. Prompt context attachment
  projection now also bounds model-visible ids, labels, summaries, paths,
  selectors, and measurement values so an oversized selection chip cannot push
  unbounded text into prompt JSON. The stale public wasm/TypeScript
  `inspectGeometry` wrapper is now removed from the source package, generated
  package declarations/glue, and the desktop-copied public runtime package; only
  runtime-internal bounded geometry inspection helpers and negative tool-catalog
  assertions remain. The checked-in web wasm binaries and desktop public runtime
  copy are now regenerated/aligned so they expose
  `applyFeaStudyDocumentOperation` for typed `.fea` authoring and no longer
  expose retired `inspectGeometry` or `feaResults` exports. The desktop app now
  imports static FEA contract/catalog values through the lightweight
  `runmat/fea-contracts` package subpath, so UI and agent bridge code no longer
  pull Next's resolver through the heavyweight top-level wasm bundler entry
  when only generated FEA constants are needed.
- Working tree policy: keep changes unstaged and uncommitted.

## Completed Slices

### 2026-07-13: OCCT Preview Cancellation Slice 1

Scope completed:

- Diagnosed intermittent `GEOMETRY_PARSE_FAILED: OCCT CAD import cancelled`
  reports as expected OCCT preview cancellation being surfaced through the
  parse-failure path when users switch CAD files while a prior preview is still
  loading.
- Changed native OCCT bridge error mapping so `OCCT CAD import cancelled`
  becomes `GeometryImportError::Cancelled` instead of `ParseFailed`.
- Added a one-shot visible-preview retry in both standalone CAD preview and
  embedded `.fea` geometry preview surfaces, so stale disposal cancellation can
  settle without showing a false parse failure.
- Added focused regression coverage for cancellation retry in CAD and `.fea`
  preview surfaces.

Tests/evidence:

- `npm run test:unit -- src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/fea/fea-study-surface.spec.tsx`
  passed from `../runmat-private/desktop` with 16 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test -p runmat-geometry-io --features occt-native` passed from
  `runmat-analysis` with 52 tests plus doc tests.

Remaining from OCCT preview cancellation:

- Re-test in a freshly restarted desktop process by switching between two STEP
  files while one is still loading. The expected behavior is either successful
  retry of the visible model or a cancellation message, not
  `GEOMETRY_PARSE_FAILED`.

### 2026-07-13: FEA Contract Package Boundary Slice 1

Scope completed:

- Diagnosed the current desktop boot failure as a package-boundary issue:
  desktop UI/agent code imported static FEA constants from the top-level
  `runmat` entry, which made Next resolve the unavailable bundler wasm path
  `./pkg/runmat_wasm.js` even though the desktop runtime uses copied web wasm
  assets.
- Added a lightweight `runmat/fea-contracts` TS package subpath that re-exports
  generated FEA study/run contracts plus shared FEA run artifact ids without
  importing the native wasm loader.
- Kept top-level `runmat` compatibility by re-exporting the same FEA contracts
  from `index.ts`, while moving desktop static FEA value imports to
  `runmat/fea-contracts`.
- Regenerated the TS package output so `dist/fea-contracts.js` and declarations
  exist for desktop/Next resolution.

Tests/evidence:

- `node --input-type=module -e "const m = await import('runmat/fea-contracts'); console.log(m.FEA_SUPPORTED_PHYSICS_PROFILES.length, m.FEA_RUN_KIND, m.feaRunArtifactRefId('dataset'));"` passed from `../runmat-private/desktop`, printing `13 fea-study __fea__:dataset`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/run/fea-run-manifest.spec.ts src/run/fea-run-orchestrator.spec.ts src/app/components/agent/fea-physics-workflow.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/fea/fea-study-surface.spec.tsx`
  passed from `../runmat-private/desktop` with 74 tests.
- `npm run build` passed from `../runmat-private/desktop`; Next compiled and
  generated static pages without the prior `Can't resolve
  './pkg/runmat_wasm.js'` failure.

Remaining from FEA contract package boundary:

- Restart `npm run dev:tauri` from the current checkout so the running desktop
  process picks up the regenerated package output and copied agent/runtime wasm
  assets.

### 2026-07-13: FEA Preview Scene Pick Selection Slice 1

Scope completed:

- Investigated the live Tauri/agent telemetry after the bracket FEA manual test
  showed unselectable rendered regions and stale model tool calls.
- Confirmed the live model frame still advertised retired `geometry_inspect`,
  `geometry_view`, `fea_check`, and `fea_run` tools, and dropped `fea_context`
  during compaction from an oversized stale context frame; current
  model-visible source registry/tests and the checked-in harness web manifest
  are aligned to the session-oriented geometry/generic check-run catalog, so
  the running desktop process needs a fresh rebuild/restart to pick up the
  current harness catalog.
- Moved geometry-scene pick to prompt-selection conversion into the shared
  editor scene-selection module so CAD and `.fea` previews use one selector,
  label, measurement, and mesh/region fallback contract.
- Wired direct canvas picking in `FeaStudySurface` through
  `pickGeometrySceneRegion`, so clicking a rendered `.fea` geometry scene now
  publishes the same shared `EditorSceneSelectionSnapshot` used by CAD preview
  picks and prompt context chips.
- Kept existing review-tree row selection behavior, including shift-add,
  alongside the new direct rendered-scene pick path.

Tests/evidence:

- `otell search geometry_inspect --since 2h -C 8 --limit 50` showed the live
  Tauri session using retired `geometry_inspect`/`geometry_view` tools.
- `otell search 'agent.context.frame_built' --since 30m -C 2 --limit 20`
  showed the live frame had `tool_names` containing retired FEA/geometry tools,
  `token_estimate=147095`, and dropped `fea_context` during compaction.
- `npm test -- src/app/components/fea/fea-study-surface.spec.tsx src/app/components/geometry/geometry-preview-surface.spec.tsx`
  passed from `../runmat-private/desktop` with 14 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from FEA preview scene pick selection:

- Rebuild/restart the desktop Tauri app from the current worktree and verify a
  new telemetry frame advertises the session-oriented geometry tools instead of
  `geometry_inspect`/`geometry_view`, preserves dynamic FEA context after
  compaction, and lets a real click on the rendered `.fea` bracket create a
  removable prompt context chip.

### 2026-07-12: Generic Desktop Runtime Check Boundary Slice 1

Scope completed:

- Added generic desktop runtime check types and a required `RuntimeClient.check`
  method so agent/domain check paths no longer depend on the FEA-specific
  `checkFeaStudy` client method.
- Moved `.fea` check dispatch/conversion into a runtime-client adapter helper
  that concrete browser, Tauri, and mock clients can use internally.
- Updated `checkRuntimePath`, `executeAgentCheck`, the FEA run orchestrator,
  runtime-provider wiring, runtime client mocks, and focused tests to depend on
  the generic `check` surface.
- Kept lower-level `checkFeaStudy` only behind concrete runtime-client adapter
  boundaries and FEA execution fixtures.

Tests/evidence:

- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/runtime-check.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/run/fea-run-orchestrator.spec.ts src/runtime/runtime-provider.spec.tsx src/runtime/hooks/runtimeHooks.spec.tsx src/runtime/hooks/useFileSystem.spec.tsx`
  passed from `../runmat-private/desktop` with 116 tests.
- `rg -n 'Pick<RuntimeClient, "checkFeaStudy"|client: Pick<RuntimeClient, "checkFeaStudy"' ../runmat-private/desktop/src -g '*.{ts,tsx}' --glob '!**/node_modules/**'`
  returned no matches from `runmat-analysis`.
- Remaining `checkFeaStudy` matches under runtime domain/run are adapter tests
  or FEA execution fixtures, not the model-facing check boundary.
- `git diff --check` passed from both `runmat-analysis` and
  `../runmat-private`.

Remaining from generic desktop runtime check boundary:

- Continue the requirement-by-requirement completion audit. The desktop agent
  and run orchestration paths now reach `.fea` validation through
  `RuntimeClient.check`, matching the generic harness `check` direction instead
  of preserving a UI-only FEA check side path.

### 2026-07-12: Final Naming Drift Audit Slice 1

Scope completed:

- Audited active FEA/meshing/runtime/plot and private harness/desktop surfaces
  for standalone `tet`/`tets` shorthand identifiers and meta `production`
  naming drift.
- Replaced the remaining runtime test ids `missing_tet` with
  `missing_tetrahedron`.
- Renamed the private harness parser test from production-build terminology to
  release-build terminology.
- Renamed the desktop markdown roundtrip fixture id from
  `production-notebook-mixed` to `release-notebook-mixed`.
- Fixed a private `ah-cli` compile break found during verification by importing
  the existing `RuntimeInspectionSelector` type at the local runtime adapter
  boundary.

Tests/evidence:

- `rg -n -P "\b[A-Za-z0-9]+_tets?\b|\btets?_[A-Za-z0-9]+\b|\btets?\b" crates/runmat-meshing crates/runmat-analysis crates/runmat-runtime/src/analysis crates/runmat-plot/src -g '*.rs' --glob '!**/target/**'`
  returned no matches from `runmat-analysis`.
- `rg -n -P "\b[A-Za-z0-9]+_tets?\b|\btets?_[A-Za-z0-9]+\b|\btets?\b" ../runmat-private/agent-harness ../runmat-private/desktop/src ../runmat-private/desktop/src-tauri -g '*.{rs,ts,tsx}' --glob '!**/target/**' --glob '!**/node_modules/**'`
  returned no matches from `runmat-analysis`.
- `cargo fmt --manifest-path Cargo.toml -p runmat-runtime` passed from
  `runmat-analysis`.
- `cargo test -p runmat-runtime solid_mesh_quality_reasons_require_renderable_volume_attributed_boundary`
  passed from `runmat-analysis`.
- `cargo test -p runmat-runtime analysis_mesh_render_topology_requires_solver_field_mapping`
  passed from `runmat-analysis`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-cli` passed from
  `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-cli release_build_rejects_debug_context_dir_flag`
  passed from `../runmat-private`.
- `npm test -- src/app/components/markdown/editor/markdown-bridge.spec.ts`
  passed from `../runmat-private/desktop`.

Remaining from final audit:

- Continue the requirement-by-requirement completion audit. The naming drift
  audit is clean for true shorthand identifiers and the private harness now
  compiles through the focused CLI parser test.

### 2026-07-12: Align Public Wasm FEA Authoring Artifacts Slice 1

Scope completed:

- Regenerated the web wasm package from the current Rust `runmat-wasm` source so
  the checked-in binary exports match the current API contract.
- Synced the regenerated wasm package into `bindings/ts/dist`.
- Copied the regenerated `pkg-web`, top-level runtime wrapper, and generated FEA
  contract module into the desktop public runtime package.
- Removed stale copied `feaResults` wrappers from the desktop public runtime
  package and restored the missing `applyFeaStudyDocumentOperation` forwarding
  path used by browser runtime study authoring.
- Verified the actual source/dist/desktop public wasm binaries expose
  `runmatwasm_applyFeaStudyDocumentOperation` and do not expose retired
  `runmatwasm_inspectGeometry` or `runmatwasm_feaResults`.

Tests/evidence:

- `npm run build:wasm:web` passed from `runmat-analysis/bindings/ts` with
  escalation for `wasm-pack`'s wasm-bindgen helper.
- `npm run build:types` passed from `runmat-analysis/bindings/ts`.
- `npm run sync:wasm` passed from `runmat-analysis/bindings/ts`.
- `node --check bindings/ts/pkg-web/runmat_wasm_web.js` passed from
  `runmat-analysis`.
- `node --check bindings/ts/dist/pkg-web/runmat_wasm_web.js` passed from
  `runmat-analysis`.
- `node --check bindings/ts/dist/index.js` passed from `runmat-analysis`.
- `node --check desktop/public/js/runtime/index.js` passed from
  `../runmat-private`.
- `node --check desktop/public/js/runtime/pkg-web/runmat_wasm_web.js` passed
  from `../runmat-private`.
- Dynamic module checks proved both `bindings/ts/dist/index.js` and
  `desktop/public/js/runtime/index.js` import successfully and forward
  `applyFeaStudyDocumentOperation` through the native-session override hook.
- Binary audits with `rg -a -l` proved source/dist and desktop public wasm
  binaries contain `runmatwasm_applyFeaStudyDocumentOperation` and no longer
  contain `runmatwasm_inspectGeometry` or `runmatwasm_feaResults`.

Remaining from final public-surface audit:

- Continue the requirement-by-requirement completion audit. The generated wasm
  package is now aligned with the generic result path and typed study-authoring
  path instead of retaining stale geometry/result escape hatches.

### 2026-07-12: Remove Public Wasm Geometry Inspect Wrapper Slice 1

Scope completed:

- Removed the public `inspectGeometry` wasm export from the RunMat wasm session
  API.
- Removed `inspectGeometry` from the TypeScript `RunMatSessionHandle`,
  `RunMatNativeSession`, and `WebRunMatSession` wrapper.
- Removed the checked-in generated web glue and declaration entries for
  `inspectGeometry` / `runmatwasm_inspectGeometry` from `bindings/ts`.
- Removed the same stale generated wrapper/declaration entries from the
  desktop-copied public runtime package.
- Kept the runtime-internal bounded geometry inspection helper used by
  `previewGeometry` and Tauri preview/session construction.

Tests/evidence:

- `cargo fmt --manifest-path Cargo.toml -p runmat-wasm` passed from
  `runmat-analysis`.
- `cargo check -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host`
  passed from `runmat-analysis`.
- `npm run lint` passed from `runmat-analysis/bindings/ts`.
- `node --check bindings/ts/pkg-web/runmat_wasm_web.js` passed from
  `runmat-analysis`.
- `node --check bindings/ts/dist/pkg-web/runmat_wasm_web.js` passed from
  `runmat-analysis`.
- `node --check bindings/ts/dist/index.js` passed from `runmat-analysis`.
- `node --check ../runmat-private/desktop/public/js/runtime/pkg-web/runmat_wasm_web.js`
  passed from `runmat-analysis`.
- `node --check ../runmat-private/desktop/public/js/runtime/index.js` passed
  from `runmat-analysis`.
- `rg -n "inspectGeometry|runmatwasm_inspectGeometry|runtime_geometry_inspect|geometryInspect" bindings/ts crates/runmat-wasm/src/api/session.rs`
  returned no matches from `runmat-analysis`.
- `rg -n "inspectGeometry|runmatwasm_inspectGeometry|runtime_geometry_inspect|geometryInspect" desktop/public desktop/src desktop/src-tauri -g '*.{ts,tsx,rs,json,md,js,d.ts}'`
  returned no matches from `../runmat-private`.

Remaining from final public-surface audit:

- Continue the requirement-by-requirement completion audit. The public wasm,
  TypeScript, desktop browser, and Tauri surfaces no longer expose the retired
  geometry inspection escape hatch; remaining `geometry_inspect` names are
  runtime-internal bounded parsing helpers or negative model-tool assertions.

### 2026-07-12: Remove Public Geometry Inspect Runtime Path Slice 1

Scope completed:

- Removed `inspectGeometry` from the shared desktop runtime client interface.
- Removed the browser and Tauri client methods that exposed direct geometry
  inspection.
- Removed the browser worker `geometryInspect` command and payload variant.
- Removed the Tauri `runtime_geometry_inspect` command and corresponding
  runtime worker command variant.
- Kept the internal Rust preview helper that builds bounded preview payloads;
  product, agent, and model-facing paths now route through preview/session
  surfaces rather than a public raw inspection API.

Tests/evidence:

- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- --run --reporter=dot src/runtime/clients/browser/index.spec.ts src/runtime/runtime-provider.spec.tsx src/runtime/lanes/runtime-lane-manager.spec.ts src/runtime/hooks/runtimeHooks.spec.tsx src/runtime/hooks/useFileSystem.spec.tsx`
  passed from `../runmat-private/desktop` with 94 tests.
- `cargo fmt --manifest-path desktop/src-tauri/Cargo.toml` passed from
  `../runmat-private`.
- `cargo check --manifest-path desktop/src-tauri/Cargo.toml` passed from
  `../runmat-private`.

Remaining from Phase 9:

- Continue the final requirement audit. The public desktop/browser/Tauri
  geometry runtime surface no longer exposes direct geometry inspection.

### 2026-07-12: Shared Selection Evidence Refresh Slice 1

Scope completed:

- Re-ran the agent-harness tool suite covering runtime tool registration,
  disabled FEA/geometry config, `execute_file`, workspace open/select, geometry
  session tools, and raw-topology compaction.
- Re-ran desktop shared scene-selection and prompt-context tests covering
  prompt attachment rendering/removal, editor scene-selection merge behavior,
  geometry preview picks, mesh identity, shared scene presentation, `.fea`
  rendered-region selections, and additive shift-click behavior.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests`
  passed from `../runmat-private` with 19 tests.
- `npm test -- --run --reporter=dot src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/shell/editor-scene-selection.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/fea/fea-study-surface.spec.tsx`
  passed from `../runmat-private/desktop` with 21 tests.

Remaining from final acceptance audit:

- Continue the requirement-by-requirement audit before claiming full goal
  completion. The shared selection and bounded geometry tool gates have fresh
  passing evidence.

### 2026-07-12: Naming And Unused Geometry Side-Path Audit Slice 1

Scope completed:

- Audited active FEA/geometry desktop and harness code for proprietary sample
  names, the removed `listGeometryRegions` helper, and remaining short
  tetrahedron wording in the touched runtime/plot surface.
- Confirmed the proprietary geometry sample name and unused region-listing side
  path are gone from active desktop/agent code.
- Replaced the remaining `tet` shorthand in a RunMat plot assertion message
  with `tetrahedron`.

Tests/evidence:

- `cargo fmt --manifest-path Cargo.toml -p runmat-plot` passed from
  `runmat-analysis`.
- `rg -n "Mono Strut|061726|listGeometryRegions|tet boundary" desktop/src agent-harness/crates /Users/nallana/Source/runmat-acc-2/runmat-analysis/crates/runmat-plot/src/export/cpu_surface.rs -g '*.{ts,tsx,rs}'`
  returned no matches from `../runmat-private`.

Remaining from naming/legacy audit:

- Continue the final requirement audit. Remaining `production` hits are
  environment/deployment/eval terminology, not FEA/geometry type or file names.

### 2026-07-12: Retire Unused Geometry Region Listing Side Path Slice 1

Scope completed:

- Removed the unused desktop `listGeometryRegions` runtime-client method from
  the shared runtime interface and browser, Tauri, and mock implementations.
- This removes an inspect-based region-listing side path so product geometry
  flows continue to use preview/session state instead of direct legacy region
  inspection.
- Replaced the proprietary sample geometry filename in the browser runtime
  client test with a generic assembly fixture name.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/clients/browser/index.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from legacy/side-channel retirement:

- Continue the final requirement audit. Current remaining `inspectGeometry`
  references are low-level runtime preview helpers/tests rather than app or
  agent region-selection call sites.

### 2026-07-12: Prompt Attachment String-Bounding Slice 1

Scope completed:

- Added model-visible string caps to prompt context attachment projection for
  attachment ids, labels, summaries, paths, session ids, selection ids,
  surfaces, entity selectors, entity summaries, and measurement values.
- Preserved exact product/internal selection state while bounding the dynamic
  context JSON sent to the model.
- Added a regression test proving oversized selection chip text is shortened
  and noisy tails are omitted from model-visible prompt context.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --tests`
  passed from `../runmat-private` with 57 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint`
  passed from `../runmat-private`.

Remaining from Phase 9 context-cost retirement:

- Continue the final requirement audit. Prompt context attachments now bound
  both count and string length before projection.

### 2026-07-12: Domain-Owned FEA Context Protocol Slice 1

Scope completed:

- Moved stable FEA/geometry workflow protocol guidance out of generic tool
  guidance and into the FEA domain context provider.
- Extended the domain provider interface so it owns stable sections as well as
  dynamic attachments.
- Kept tool guidance focused on exposed tools and generic runtime rules while
  the FEA provider owns profile/geometry/session/selection workflow protocol.
- Added assembler evidence that the FEA protocol section is cacheable and
  byte-stable across different live physics/profile selections, while the
  profile-specific FEA state remains in dynamic attachments.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --tests`
  passed from `../runmat-private` with 56 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint`
  passed from `../runmat-private`.

Remaining from Phase 6/9 context boundaries:

- Continue the final requirement audit across composed harness and desktop
  flows. The stable/dynamic provider boundary now has direct test evidence.

### 2026-07-12: Phase 8 Full-Family Solve Matrix Evidence Slice 1

Scope completed:

- Audited current composed Phase 8 solve/postprocess host-parity coverage
  against the change-plan profile-family matrix.
- Re-ran the full `agent_can_solve_` host-parity filter and verified all 12
  composed solve/postprocess paths still pass.
- The passing matrix covers linear static structural, modal, transient
  structural, nonlinear structural, thermal, electromagnetic, acoustic, CFD,
  thermo-mechanical coupled, electro-thermal, CHT-style, and FSI-style flows
  through the same generic runtime operations.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_ -- --nocapture` passed from `../runmat-private` with 12 host-parity tests.

Remaining from Phase 8:

- Phase 8 solve/postprocess has fresh full-family composed evidence. Continue
  the final requirement audit for any remaining UI/guided-pane or context
  boundary gaps before claiming the full plan complete.

### 2026-07-12: Phase 8 Report Composed Evidence Slice 1

Scope completed:

- Audited the Phase 8 report path against the change-plan requirement that
  postprocess/report flows use generic runtime surfaces instead of FEA-specific
  result tools.
- Verified the composed host-parity report path for an acoustic FEA run:
  `select_run`, `variables`, `figures`, `show_figures`, bounded `variable`, and
  `write`.
- Confirmed the report fixture writes a workspace Markdown artifact and rejects
  structural/stress fallback content, inline image blobs, and embedded full
  field data.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_write_fea_report_from_generic_runtime_surfaces` passed from `../runmat-private`.

Remaining from Phase 8:

- Continue the final requirement-by-requirement audit across all Phase 8
  profile-family solve/postprocess cases before claiming the full plan is
  complete.

### 2026-07-12: Additive Result Selection Prompt Context Slice 1

Scope completed:

- Audited the mesh/result selection paths against the shared prompt attachment
  contract.
- Confirmed geometry preview mesh picks already publish `mesh:*` selectors
  through `EditorSceneSelectionSnapshot` and preserve shift-click additive
  intent.
- Extended the desktop FEA results pane so result-field clicks preserve
  `selectionMode`: ordinary clicks replace selection and shift-clicks publish
  additive result-field selection state.
- Kept result selection in the shared scene-selection/prompt-attachment path:
  no FEA-result-specific attachment side channel or full field values are sent.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/fea/fea-results-pane.spec.tsx src/app/components/agent/prompt-context-attachments.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from shared scene/context selection:

- Continue the final completion audit. Result-field selection and mesh triangle
  selection now both route through the shared scene-selection contract; richer
  hit-testable 3D result overlays should plug into the same contract when they
  exist.

### 2026-07-12: Model-Driven Multi-Selection Scene State Slice 1

Scope completed:

- Extended the geometry-session `geometry_select` tool to accept bounded
  multi-selector input through `selectors` and `region_ids` while preserving the
  existing single-selector path.
- Added structured selected entities to `GeometrySelection` snapshots so
  model-driven scene selection can converge with product/user scene selection
  instead of collapsing multi-select state into one opaque selector string.
- Guarded `geometry_create_region` against silently creating an invalid study
  region from a multi-entity current selection unless the model provides an
  explicit selector. This keeps composite-region semantics out of the `.fea`
  document contract until the runtime explicitly supports them.
- Carried structured model-driven selection entities through the desktop
  geometry scene-state event and editor bridge into
  `EditorSceneSelectionSnapshot`.
- Updated model-visible geometry guidance and `geometry_select` argument
  metadata so the model knows when to use `selector`/`region_id` versus
  `selectors`/`region_ids`, and does not treat multi-selection as an implicit
  composite study-region edit.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-harness` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools geometry_session_tools` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context geometry_select_metadata_describes_multi_entity_selection` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context includes_fea_and_geometry_guidance_when_tools_are_exposed` passed from `../runmat-private`.
- `npm test -- --run --reporter=dot src/agent/geometry-scene-state-events.spec.ts src/app/components/shell/editor-agent-geometry-scene-bridge.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from shared scene/context selection:

- Continue the final completion audit for true hit-testable mesh/result 3D
  selection surfaces. The model-driven geometry selection path now supports the
  same multi-entity scene-selection contract as user-created selection chips.

### 2026-07-12: Composed Multi-Entity Selection Host-Parity Slice 1

Scope completed:

- Strengthened the composed `ah-harness` selected-region setup path so the user
  turn carries one additive geometry-selection prompt attachment with two
  selected entities.
- Asserted the actual model request preserves both stable selectors
  (`region:face_mount` and `region:face_bolt`) plus the aggregate selection id
  in the bounded `prompt_context_attachments` current-state block.
- Kept the domain path clean: the model still uses geometry-session selection,
  typed `.fea` region/constraint operations, then generic `check` and
  `execute`; no FEA-specific side channel or raw selector prose path was added.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-harness` passed
  from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint` passed from `../runmat-private`.

Remaining from shared scene/context selection:

- Continue the final completion audit for any remaining result/mesh 3D picking
  paths. The composed region setup path now proves multi-entity selected context
  survives into an actual agent turn and drives the typed study/check/execute
  sequence.

### 2026-07-12: Multi-Entity Prompt Context Projection Slice 1

Scope completed:

- Strengthened the `ah-context` dynamic attachment test for prompt context
  selections to cover a merged additive geometry selection with two selected
  entities.
- Proved the model-visible `prompt_context_attachments` state item preserves the
  merged selection as one authoritative attachment with two bounded entity
  selectors.
- Kept the existing bounded context projection implementation unchanged: the
  context layer already limits attachment count, entity count, and measurement
  count while keeping selectors structured.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/crates/ah-context/Cargo.toml` passed
  from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context dynamic_attachments_include_prompt_context_selections` passed from `../runmat-private`.
- `git diff --check` passed from both `runmat-analysis` and `runmat-private`.

Remaining from shared scene/context selection:

- Continue the composed completion audit for the selected-context workflow
  across actual agent/harness turns. Multi-entity prompt context projection is
  now explicitly covered at the harness context layer.

### 2026-07-12: Prompt Context Chip Removal Projection Slice 1

Scope completed:

- Strengthened the prompt-context removal proof so it covers the actual
  next-turn model frame input, not only local chip display state.
- Confirmed scene-backed chip removal still clears shared scene selection.
- Confirmed the filtered `visiblePromptContextAttachments` list used by
  `buildAgentTurnContext` excludes the removed geometry selector while keeping
  ordinary file context.
- Kept the implementation unchanged because the current domain boundary is
  already correct: the focused prompt attachment hook owns chip visibility and
  removal state, while turn-context construction consumes the visible attachment
  list.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/use-prompt-context-attachment-state.spec.tsx src/app/components/agent/agent-turn-context.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx src/app/components/agent/prompt-context-attachments.spec.ts` passed from `../runmat-private/desktop` with 21 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `runmat-private`.

Remaining from shared scene/context selection:

- Continue the composed completion audit for the full selected-context workflow
  through typed `.fea` edits, generic `check`/`execute`, and generic
  variables/figures result inspection. Chip removal is now proven to clear the
  matching scene context from the next model turn.

### 2026-07-12: FEA Additive Scene Selection Slice 1

Scope completed:

- Closed a product-side gap where CAD preview picks could publish additive
  selections, but `.fea` rendered-region selections from the study review tree
  were always emitted as replacement selections.
- Added shared selection-mode state to `FeaStudySurface`, reset it when the
  preview context changes, and publish shift-clicked review rows with
  `selectionMode: "add"`.
- Kept additive merge ownership in the editor shell's existing
  `mergeEditorSceneSelections` path rather than duplicating selection merge
  logic inside the FEA surface.
- Added a stable internal review-row node id for interaction tests so tests can
  target the domain node instead of duplicated visible labels.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/fea/fea-study-surface.spec.tsx src/app/components/shell/editor-scene-selection.spec.ts src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx` passed from `../runmat-private/desktop` with 21 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `runmat-private`.

Remaining from shared scene/context selection:

- Continue the composed completion audit for the full prompt-chip loop:
  product selection chip -> bounded model context -> typed `.fea` edit ->
  generic `check`/`execute` -> generic result inspection. Additive selection is
  now covered for both CAD preview picks and `.fea` rendered-region review
  picks.

### 2026-07-11: Live Geometry Camera Presentation Slice 1

Scope completed:

- Added a bounded Rust-owned `GeometrySceneViewPreset` presentation field for
  model-driven geometry camera/view commands.
- Applied incoming view presets inside `runmat-plot` through the renderer's
  existing camera preset path, so `geometry_set_camera` can move the live
  product scene instead of only affecting model-side session state.
- Kept view presets as explicit camera actions rather than sticky dynamic
  selection state; later user selections preserve visibility and section state
  but do not keep reapplying the last model camera command.
- Extended the public TS presentation contract and desktop agent geometry
  bridge so `geometry_open_session`/`geometry_set_camera` observations project
  bounded view presets into the same presentation path as selection,
  visibility, and section state.
- Salted camera-action presentation revisions with the tool call id so repeated
  same-view commands can still reach the renderer.

Tests/evidence:

- `cargo fmt` passed.
- `CARGO_TARGET_DIR=/private/tmp/runmat-target-plot-view-test cargo test -p runmat-plot presentation_` passed.
- `CARGO_TARGET_DIR=/private/tmp/runmat-target-plot-view-test cargo test -p runmat-plot geometry_scene_view_presets_map_to_camera_presets` passed.
- `npm run build:types` passed from `bindings/ts`.
- `npm test -- --run --reporter=dot src/app/components/shell/editor-scene-presentation.spec.ts src/app/components/shell/editor-agent-geometry-scene-bridge.spec.ts src/runtime/graphics/figure-canvas-adapter.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/shell/editor-panel.spec.ts src/app/components/agent/use-prompt-context-attachment-state.spec.tsx` passed from `../runmat-private/desktop` with 51 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from live scene presentation:

- Continue the shared scene/context selection gate across STEP/CAD, `.fea`,
  mesh, and result surfaces; model-driven camera, visibility, selection, and
  section state now share the runtime presentation/selection path instead of
  staying as model-only session state.

### 2026-07-11: Live Geometry Section Presentation Slice 1

Scope completed:

- Added Rust-owned `GeometryScenePresentation` section state with explicit
  preserve, clear, and apply semantics.
- Implemented renderer-side section clipping for triangle and line render data
  in `runmat-plot`, so model-driven sectioning changes actual scene data
  instead of relying on desktop-only display state.
- Preserved existing section state when sparse presentation updates omit it,
  matching the hidden/isolated owner visibility presentation contract.
- Extended the public TS presentation surface and desktop shell bridge so
  `geometry_section` tool observations update the same live geometry scene
  presentation path used by user selections and visibility changes.
- Kept geometry snapshots as full current-state events: section mutations also
  publish current visibility, so the product and model converge on one scene
  state rather than sparse transcript patches.

Tests/evidence:

- `cargo fmt` passed.
- `CARGO_TARGET_DIR=/private/tmp/runmat-target-plot-section-test cargo test -p runmat-plot presentation_section` passed.
- `CARGO_TARGET_DIR=/private/tmp/runmat-target-plot-section-test cargo test -p runmat-plot presentation_` passed.
- `npm run build:types` passed from `bindings/ts`.
- `npm test -- --run --reporter=dot src/app/components/shell/editor-scene-presentation.spec.ts src/app/components/shell/editor-agent-geometry-scene-bridge.spec.ts src/runtime/graphics/figure-canvas-adapter.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/shell/editor-panel.spec.ts` passed from `../runmat-private/desktop` with 48 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from live scene presentation:

- Continue the shared scene/context selection gate across STEP/CAD, `.fea`,
  mesh, and result surfaces; section state now shares the runtime presentation
  contract and is no longer the renderer gap.

### 2026-07-11: Result Surface Probe Source Audit Slice 1

Scope completed:

- Audited the current desktop result surfaces for selectable 3D result-scene
  sources.
- Confirmed geometry and `.fea` preview surfaces already route hit-tested
  geometry/mesh picks through `pickGeometrySceneRegion` into the shared
  `EditorSceneSelectionSnapshot` contract.
- Confirmed current FEA result inspection does not expose a separate 3D
  node/element hit-test source; result context enters the prompt strip through
  bounded field selections and generic run/figure attachments.
- Kept the future requirement explicit: if/when a hit-testable result scene is
  added, it should publish result probe/node/element selections through the same
  shared attachment contract instead of adding a result-only side channel.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/fea/fea-results-pane.spec.tsx src/app/components/agent/prompt-context-attachments.spec.ts`
  passed from `../runmat-private/desktop` with 11 tests.
- The existing field-selection tests prove result fields become
  `renderTargetKind: "result"` scene selections without field values.
- The prompt-context tests prove result selections become `result_selection`
  attachments through the shared attachment path.

Remaining:

- No current product result-pick source remains to wire. Future result-scene
  3D probes must enter through the same shared scene-selection contract.

### 2026-07-11: Live Geometry Owner Visibility Presentation Slice 1

Scope completed:

- Extended the Rust `GeometryScenePresentation` contract with optional
  `hidden_owner_node_ids` and `isolated_owner_node_ids`, keeping owner
  visibility in the plotting/runtime domain rather than React-only state.
- Taught `PlotRenderer` to resolve hidden and isolated owner visibility through
  the existing geometry owner-node visibility filter while preserving current
  visibility when selection-only presentation updates omit visibility fields.
- Updated the public TypeScript `GeometryScenePresentationState` source and
  regenerated the local package declarations used by desktop typecheck.
- Added desktop shell presentation state so model-driven `geometry_set_visibility`
  observations become live presentation updates without being stored in prompt
  chip attachment data.
- Merged model-driven owner visibility with user/model selected-region
  presentation before passing state through `FigureCanvas` and the runtime
  adapter.

Tests/evidence:

- `CARGO_TARGET_DIR=/private/tmp/runmat-target-plot-test cargo test -p runmat-plot presentation_resolves_hidden_and_isolated_owner_visibility`
  passed.
- `npm test -- --run --reporter=dot src/app/components/shell/editor-scene-presentation.spec.ts src/app/components/shell/editor-agent-geometry-scene-bridge.spec.ts src/runtime/graphics/figure-canvas-adapter.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/shell/editor-panel.spec.ts`
  passed from `../runmat-private/desktop` with 45 tests.
- `npm run build:types` passed from `bindings/ts`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-private` and `runmat-analysis`.

Remaining:

- Section state is still not live product rendering state. Closing that requires
  a real renderer/runtime clip-section contract, not a desktop-only workaround.
- Result-surface probe source audit is complete for the current product surface;
  no current 3D result-pick source remains to wire.

### 2026-07-11: Geometry Scene Presentation Bridge Slice 1

Scope completed:

- Added a focused shell presentation snapshot derived from the shared
  `EditorSceneSelectionSnapshot`, keeping scene presentation separate from
  prompt-chip attachment data.
- Threaded the active geometry presentation through `EditorPanel`,
  `EditorSurfaceHost`, `EditorSurfaceRegistry`, `GeometryPreviewSurface`, and
  `FeaStudySurface`.
- Extended `FigureCanvas` and the canvas adapter so geometry scene surfaces
  apply `GeometryScenePresentationState` immediately after binding a geometry
  scene handle and before presenting the surface.
- Defaulted geometry scene canvases to a clearing presentation when no selected
  region is active, preventing stale selected-region state on reused surfaces.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/graphics/figure-canvas-adapter.spec.ts src/app/components/shell/editor-scene-presentation.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/shell/editor-agent-geometry-scene-bridge.spec.ts src/app/components/shell/editor-panel.spec.ts`
  passed from `../runmat-private/desktop` with 42 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-private` and `runmat-analysis`.

Remaining:

- Superseded by the live owner-visibility slice for visibility. Section state
  still needs a real renderer/runtime clip-section contract before it can be
  represented in live product surfaces.
- Continue auditing result-scene 3D probe picking versus the current field and
  figure selection attachment path.


### 2026-07-11: Runtime And Wasm FEA Contract Verification Slice 1

Scope completed:

- Fixed the runtime operation contract fixture so successful
  `geometry.prep_for_analysis` coverage uses topology-backed geometry with real
  region-to-mesh mappings instead of a metadata-only STEP product listing.
- Added explicit typed coverage that metadata-only STEP geometry without
  topology fails analysis prep through `geometry.prep_for_analysis/v1` rather
  than inventing mesh mappings or falling through an ambiguous runtime path.
- Updated the wasm FEA document operation test to use the Rust-owned explicit
  `model_profile` create contract and assert the returned model profile plus
  derived run kind.
- Preserved the boundary: STEP metadata can still drive material/model
  inference, but analysis prep success requires topology-backed mapped geometry.

Tests/evidence:

- `cargo test -p runmat-runtime --test operation_contracts -- --nocapture`
  passed with 52 tests.
- `wasm-pack test --node . --test fea_document_operation` passed from
  `crates/runmat-wasm` after clearing generated wasm target artifacts to recover
  local disk space.
- `git diff --check` passed from `runmat-analysis`.

Remaining:

- Continue the completion audit with the shared scene/context gate and any
  remaining composed desktop/harness/runtime checks. The runtime and wasm
  operation contracts no longer rely on hidden structural defaults or
  metadata-only geometry for analysis prep success.

### 2026-07-11: Agent Geometry Scene State Bridge Slice 1

Scope completed:

- Added a typed agent geometry scene-state event bridge that correlates
  `tool_call_requested` and `tool_call_completed` events, parses bounded
  geometry-session snapshots, and dispatches model-driven scene state changes
  without coupling the agent provider directly to editor state.
- Wired the editor provider to subscribe to those events and convert model
  geometry selections into the shared `EditorSceneSelectionSnapshot` path used
  by graphical user selections.
- Kept conversion logic in a focused shell bridge helper so the broad editor
  panel does not own geometry-session parsing semantics.
- Covered clear/close behavior so matching model-driven geometry sessions clear
  the shared scene selection rather than leaving stale prompt chips.

Tests/evidence:

- `npm test -- --run --reporter=dot src/agent/geometry-scene-state-events.spec.ts src/agent/agent-provider.spec.tsx src/app/components/shell/editor-agent-geometry-scene-bridge.spec.ts src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/agent/use-prompt-context-attachment-state.spec.tsx`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final audit for result-scene 3D probe picking and broader composed
  desktop/harness verification, but model-driven geometry selections now enter
  the same scene-selection/prompt-chip contract as user-created selections.

### 2026-07-11: Prompt Context Attachment State Boundary Slice 1

Scope completed:

- Extracted prompt context attachment assembly, removed-chip tracking, image
  attachment state, and scene-selection chip clearing out of `agent-panel.tsx`
  into `usePromptContextAttachmentState`.
- Kept agent turn dispatch in the panel while moving reusable prompt context
  state into the agent prompt-context domain.
- Added hook coverage for shared scene chip removal and image attachment add/remove
  behavior so the extraction is pinned independently of the full panel.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/use-prompt-context-attachment-state.spec.tsx src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/agent/prompt-image-attachments.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx src/app/components/agent/agent-turn-context.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final requirement audit. The broad agent pane still owns shell/session
  orchestration, but prompt context attachment state no longer lives there.

### 2026-07-11: Mesh Scene Pick Attachment Classification Slice 1

Scope completed:

- Tightened geometry-scene pick projection so picked triangles with runtime
  `meshId` and no named region now publish `mesh:*` selectors with
  `mesh_triangle` entity type.
- Preserved named CAD/geometry region picks as `region:*` geometry selections.
- Added regression coverage for the no-region mesh pick path so it cannot fall
  through as generic `triangle:*` geometry context.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/agent/prompt-context-attachments.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Result selections currently have field/figure selector coverage. Continue the
  final audit only for richer result-scene 3D probe picking if/when that surface
  exposes node/element hit tests.

### 2026-07-11: Shared Scene Selection Source Cleanup Slice 1

Scope completed:

- Removed the stale `FeaEditorSelectionSnapshot` DTO and the FEA-specific prompt
  attachment adapter.
- Changed the FEA study surface to publish `EditorSceneSelectionSnapshot`
  directly, with active path, source path, selector, surface, render state, and
  bounded entity metadata.
- Updated the agent pane and FEA visual-state projection to consume the same
  shared scene selection that produces prompt chips, instead of maintaining a
  separate FEA visual-selection state path.
- Kept prompt context attachments derived from one product-level selection
  source across FEA study, geometry, mesh, and result surfaces.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/fea/fea-study-surface.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/use-fea-agent-context-state.spec.tsx src/app/components/agent/agent-turn-context.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final audit for any result/mesh scene probe source that still needs
  richer 3D picking, but the desktop agent no longer has an FEA-only prompt
  selection compatibility path.

### 2026-07-11: Selected Region Attachment Execute Acceptance Slice 1

Scope completed:

- Strengthened the composed selected-region host-parity flow so it no longer
  stops at validation. The scripted model now continues from the user-created
  geometry selection attachment through generic `execute` after adding the typed
  `.fea` region and constraint.
- Updated the acceptance loop to wait for all seven tool completions:
  `geometry_open_session`, `geometry_select`, `geometry_create_region`,
  `finite_element_study_add_region`, `finite_element_study_add_constraint`,
  `check`, and `execute`.
- Added assertions that the execute result returns the expected session/run ids
  and file execution stdout for `studies/bracket.fea`, proving the flow reaches
  the same generic runtime execution surface as other runs.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` passed from
  `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint -- --nocapture`
  passed from `../runmat-private`.

Remaining:

- Continue the final requirement-by-requirement audit for any mesh/result scene
  probe source that still bypasses shared prompt context attachments.

### 2026-07-11: Browser Prompt Context Bridge Coverage Slice 1

Scope completed:

- Added desktop browser-agent client coverage proving a populated
  `UserTurnContextSnapshot` with geometry and result prompt context attachments
  is serialized into the worker `submitUserTurn` payload.
- Covered the representative model-visible selectors used by the shared
  attachment path: `region:*` for geometry selections and `field_id:*` for
  result selections.
- Added regression assertions that the browser bridge payload stays bounded and
  does not carry raw evaluator/topology or field-value payloads.

Tests/evidence:

- `npm test -- --run --reporter=dot src/agent/clients/browser/index.spec.ts src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/fea/fea-results-pane.spec.tsx`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue the requirement-by-requirement final audit; the browser client now
  has direct evidence that prompt context attachments survive the desktop to
  worker bridge.

### 2026-07-11: Prompt Context Chip Selection Clearing Slice 1

Scope completed:

- Added a small prompt-attachment domain helper that identifies scene-backed
  context chips: geometry, mesh, and result selections.
- Updated agent chip removal so removing a scene-backed chip clears both the
  shared scene selection state and the older FEA visual-selection fallback.
  This prevents stale geometry/result selections from reappearing on the next
  prompt after the user removed the chip.
- Kept file, image, run, and figure chip removal scoped to prompt attachment
  visibility or image attachment state, since those do not own graphical scene
  selection.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx src/app/components/fea/fea-results-pane.spec.tsx`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue the final audit for any mesh/result scene probe sources that still
  need to publish through the shared selection model.

### 2026-07-11: Result Field Prompt Context Selection Slice 1

Scope completed:

- Added `result` as a shared scene render target so runtime result selections
  do not masquerade as geometry or figure-only selections.
- Classified shared scene selections as `geometry_selection`,
  `mesh_selection`, or `result_selection` from bounded surface/entity metadata
  while preserving the same generic prompt attachment protocol.
- Wired FEA result-field clicks in the runtime results pane into
  `EditorSceneSelectionSnapshot` with stable `field_id:*` selectors, run/source
  identity, compact field metadata, and no field values.
- Threaded runtime result selections through the runtime bottom pane into the
  existing editor/agent shared scene-selection state instead of adding a
  separate FEA-result attachment path.
- Extended harness context coverage so `result_selection` prompt attachments
  project into the bounded dynamic model-frame state.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/fea/fea-results-pane.spec.tsx`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context prompt_context -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.

Remaining:

- A narrow source audit found no separate result-scene picking API beyond
  generic figure display and result-field selection; future mesh/result 3D probe
  surfaces should publish through the same shared selection model rather than
  adding another side path.

### 2026-07-11: Prompt Image Attachment Artifact Slice 1

Scope completed:

- Added first-class prompt image attachments to the desktop agent composer,
  including file-picker and pasted-image entry points, removable chips, and
  image-only submit enablement without letting generic workspace context chips
  count as prompt input.
- Added a bounded `image_attachment` prompt context projection that carries
  artifact id, MIME type, filename, and byte count only; image bytes stay out of
  prompt context JSON.
- Added a real harness artifact write path (`putArtifact` / `agent_put_artifact`
  / wasm `putArtifact`) so submitted image references resolve through the same
  artifact store the model request builder reads from.
- Wired submitted prompt images as `image_ref` user input parts and regenerated
  the browser agent-harness wasm manifest after exposing the new method.
- Added core harness coverage proving a persisted prompt image artifact is
  resolved to model `ImageData` before the provider request.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/prompt-image-attachments.spec.ts src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx src/app/components/agent/agent-submit-gate.spec.ts src/agent/clients/mock/index.spec.ts src/agent/clients/tauri/index.spec.ts src/agent/clients/browser/index.spec.ts src/agent/clients/browser/worker/turn-event-tracker.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core user_image_ref_resolves_from_persisted_artifact_before_model_request -- --nocapture`
  passed from `../runmat-private`.
- `cargo test --manifest-path desktop/src-tauri/Cargo.toml agent::commands::tests::artifact_lookup_returns_bytes_by_handle -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Finish the final completion audit and any high-signal acceptance consolidation
  it identifies for the guided full-family FEA flow.

### 2026-07-11: Prompt Attachment To Solve/Postprocess Acceptance Slice 1

Scope completed:

- Strengthened the composed structural solve/postprocess harness flow so the
  user turn now includes file, selected-run, and shown-figure prompt context
  attachments for an existing finite element study context.
- Added a model-request assertion proving those attachments enter the bounded
  `prompt_context_attachments` dynamic state block before the model proceeds.
- Preserved the target runtime path: typed mesh setup, generic `check`, generic
  `execute`, `select_run`, `variables`, `figures`, `show_figures`, and paged
  `variable` inspection all still run through the same composed flow.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_fea_and_postprocess_with_generic_runtime_surfaces -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.

Remaining:

- Add first-class prompt image attachments when composer image artifact
  submission exists.
- Continue the completion audit for any remaining plan items whose proof is
  still narrow, stale, or indirect.

### 2026-07-11: Prompt Attachment To Typed FEA Edit Acceptance Slice 1

Scope completed:

- Strengthened the composed selected-region harness flow so the user turn now
  includes a `geometry_selection` prompt context attachment for the selected
  mounting face.
- Extended the model-request assertion to prove the bounded
  `prompt_context_attachments` dynamic state block reaches the model with the
  stable `region:face_mount` selector, instead of relying only on FEA visual
  context.
- Kept the existing tool path intact: geometry session open, geometry selection,
  region creation, typed `.fea` region edit, typed constraint edit, and generic
  `check`.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.

Remaining:

- Add prompt image attachments when image artifact submission exists.
- Extend composed acceptance from setup/check into solve/postprocess when a
  selected prompt attachment should drive a full `check`/`execute`/inspect loop,
  not only a setup edit plus readiness check.

### 2026-07-11: Workspace Prompt Context Attachments Slice 1

Scope completed:

- Broadened desktop prompt context attachment projection beyond graphical scene
  selections to include the active/selected workspace file, selected runtime
  run, and shown runtime figure monitor slots.
- Kept the projection bounded and metadata-only: file attachments carry stable
  path identity, run attachments carry session/run/status/kind identity, and
  figure attachments carry shown figure id/slot plus monitor image MIME summary
  without duplicating figure image bytes in the chip metadata.
- Wired file/run/figure attachments into the same removable chip strip and the
  same `prompt_context_attachments` turn-context path as geometry selections.
- Extended harness context projection coverage so non-scene attachment kinds
  survive into the bounded dynamic model-frame state.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/agent/agent-turn-context.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx`
  passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context prompt_context -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Add first-class prompt image attachments when the composer/user-input image
  artifact path is implemented.
- Add mesh/result scene probes and composed acceptance proving selection/file/run
  context chip -> bounded model context -> typed `.fea` edit -> generic
  `check`/`execute`.

### 2026-07-11: Geometry Preview Scene Selection Attachment Slice 1

Scope completed:

- Added a generic desktop `EditorSceneSelectionSnapshot` boundary with additive
  selection merging, so scene selections are no longer represented only as
  FEA-specific visual selections.
- Routed existing FEA visual selections into the shared scene-selection state
  while preserving the FEA-specific snapshot for guided-pane state.
- Added a `FigureCanvas` pointer event callback and connected CAD/STEP geometry
  preview pointer-up events to the existing runtime `pickGeometrySceneRegion`
  API.
- Converted geometry preview picks into bounded scene-selection attachments with
  stable region/triangle selectors, geometry session id, render handle, compact
  measurements, and shift-click additive selection intent.
- Updated the agent pane to prefer shared scene selections when deriving prompt
  context attachments, falling back to the older FEA selection projection only
  when needed.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/agent-turn-context.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/shell/editor-scene-selection.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Broaden prompt context attachments beyond scene selections to prompt images,
  selected files, selected runs, and selected figures.
- Add mesh/result scene probes and the composed acceptance path proving
  selection chip -> bounded model context -> typed `.fea` edit -> generic
  `check`/`execute`.

### 2026-07-11: Prompt Context Attachment Bridge Slice 1

Scope completed:

- Added Rust protocol snapshots for prompt context attachments, entities, and
  measurements, and added `prompt_context_attachments` to
  `UserTurnContextSnapshot`.
- Threaded prompt context attachments through `ah-core` into `ah-context`, where
  they now project as a bounded dynamic model-frame state item named
  `prompt_context_attachments`.
- Added desktop generated protocol types and a small agent-domain helper that
  turns the existing FEA rendered-region selection into a removable
  `geometry_selection` prompt context attachment with stable selectors.
- Added a prompt attachment chip strip above the agent composer input, with
  removable chips, and wired chip removal so removed selections are omitted from
  the next submitted agent turn.
- Connected the existing FEA visual selection path into the shared prompt
  attachment path, so selected rendered regions can reach the model as selector
  context instead of prompt prose or raw topology.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/prompt-context-attachments.spec.ts src/app/components/agent/agent-turn-context.spec.ts src/app/components/agent/composer/agent-composer.spec.tsx`
  passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context prompt_context -- --nocapture`
  passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core user_turn_context_projects_fea_state_into_model_request -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Broaden the shared attachment path beyond the current FEA rendered-region
  selection source: STEP preview direct face/edge/body selections, mesh/result
  scene probes, prompt image attachments, selected runs/figures, and
  multi-select aggregation still need product wiring and composed acceptance.

### 2026-07-11: Shared Scene Selection Context Requirement Slice 1

Scope completed:

- Captured shared product/agent scene selection as a first-class target design
  requirement rather than an implied geometry-session detail.
- Added `ContextAttachment` target categories for files, images, geometry
  selections, mesh selections, runs, and figures, with bounded selection
  summaries and stable selectors instead of raw topology.
- Added plan acceptance criteria for user-created scene selections appearing as
  removable prompt chips, entering the dynamic model-frame state, and driving
  typed `finite_element_study_*` edits without requiring users to type
  selectors.
- Updated the user-experience note so the thin strip above the prompt is the
  common surface for geometry/mesh/result selections and prompt image
  attachments.
- Updated the context-layout note with an authoritative
  `prompt_context_attachments` dynamic state block that supersedes old
  attachment state and stays out of the cacheable prefix.

Tests/evidence:

- Documentation-only capture slice. `git diff --check` passed in
  `runmat-analysis`.

Remaining:

- Implement the shared selection/context attachment store, desktop prompt-chip
  UI, protocol snapshots, context projection, and composed geometry-to-study
  tests before final completion.

### 2026-07-11: Physics-Agnostic Turn Context Fixture Slice 1

Scope completed:

- Updated the agent turn-context FEA fixture so study profile, run kind,
  selected run metadata, and visible figure id are coherent modal data instead
  of mixing a linear-static structural study with modal run results.
- Added regression assertions that the generic FEA turn-context projection
  includes `modal_structural` and does not leak `linear_static_structural` or
  stress-view identifiers.
- Re-ran the existing full-family desktop guardrails proving the guided agent
  context/status workflow is catalog-driven across structural, modal, thermal,
  electromagnetic, acoustic, CFD, and coupled profile families.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/agent-turn-context.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm test -- --run --reporter=dot src/app/components/agent/fea-physics-workflow.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/fea-agent-context.spec.ts -t "supported physics profile|every supported physics|profile-specific requirements|full supported physics"`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in `runmat-private`.

Remaining:

- Continue the final acceptance audit for any remaining model-facing or UI
  surfaces that still treat structural stress as the default FEA path rather
  than one supported profile family.

### 2026-07-11: FEA Physics Workflow Catalog Guardrail Slice 1

Scope completed:

- Added direct workflow-domain tests for `fea-physics-workflow.ts` so the
  profile-family classification, setup requirements, and output examples are
  guarded where the copy is owned.
- The test matrix asserts every current Rust-generated
  `FEA_SUPPORTED_PHYSICS_PROFILES` entry maps to a concrete workflow family:
  structural, modal, thermal, electromagnetic, acoustic, CFD, or coupled.
- Added direct checks that non-structural thermal/modal/CFD output examples do
  not inherit structural von Mises defaults while structural output guidance
  still includes the structural stress path.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/fea-physics-workflow.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm test -- --run --reporter=dot src/app/components/agent/fea-physics-workflow.spec.ts src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final audit for user-flow acceptance gaps that are broader than copy
  and require composed desktop/harness/runtime proof.

### 2026-07-11: Profile-Aware Guided Requirement Copy Slice 1

Scope completed:

- Updated guided study status copy to use the same profile-family workflow
  requirement helper as the workflow state before saying a study needs boundary
  or driving conditions.
- Modal/free-free style studies no longer get generic "needs at least one
  boundary condition" or "needs at least one driving condition" guidance just
  because those counts are zero.
- Added regression coverage proving modal welcome copy says those conditions
  are not required before output/solve review and does not emit the old generic
  requirement lines.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/fea-agent-state.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final audit for any other guided-pane setup copy that should respect
  profile-family requirements rather than generic structural setup assumptions.

### 2026-07-11: Profile Output Copy Domain Boundary Slice 1

Scope completed:

- Moved profile-specific output examples into the FEA physics workflow copy
  module so guided-pane output wording is owned next to the rest of the
  profile-family workflow vocabulary.
- Reused that shared mapping from both study welcome output guidance and the
  `choose_outputs` workflow action prompt.
- Added context-level coverage proving a thermal study that is ready for output
  selection receives thermal examples in its "Choose outputs" action and does
  not receive structural von Mises/displacement hints.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts -t "output"`
  passed from `../runmat-private/desktop`.
- `npm test -- --run --reporter=dot src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue final audit for any other profile-family copy maps that should live
  in the workflow/domain layer instead of the panel/state layer.

### 2026-07-11: Profile-Aware Guided Output Copy Slice 1

Scope completed:

- Replaced the guided study welcome output examples that always started with
  structural displacement/von Mises stress with profile-aware examples derived
  from the selected FEA physics profile family.
- Kept structural output examples available for structural profiles while modal,
  thermal, electromagnetic, acoustic, CFD, coupled, and unknown profiles now
  receive family-appropriate output guidance.
- Added regression coverage proving thermal study copy mentions thermal outputs
  without structural von Mises/displacement hints, modal copy mentions mode
  shapes/natural frequencies, and structural copy still permits von Mises
  stress.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/fea-agent-state.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final audit for other live guided-pane copy that should be driven by
  selected profile family instead of broad structural-first examples.

### 2026-07-11: Non-Stress FEA Context Section Fixture Slice 1

Scope completed:

- Aligned the ah-context FEA section and dynamic-attachment fixtures so study
  identity and selected-run identity are both modal rather than mixing
  linear-static study data with modal result data.
- Added direct assertions that the JSON FEA context section carries
  `model_profile: modal_structural` and `run_kind: modal`.
- Verified the section fixtures no longer contain `bracket_static`,
  `linear_static_structural`, or `run_kind: linear_static` as generic
  context-section evidence.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context fea_context_attachment_uses_provided_workflow_snapshot -- --nocapture`
  passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context dynamic_attachments_include_fea_context_from_provider -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue final audit for any remaining context-section or guided-pane fixtures
  where structural examples are being used as generic proof instead of explicit
  structural coverage.

### 2026-07-11: Non-Stress Context Projection Fixture Slice 1

Scope completed:

- Reworked the ah-context projected FEA workflow fixture so the study identity,
  selected-run identity, and requested outputs are coherently modal instead of
  mixing a linear-static study with a modal run.
- Added projection assertions that the current-turn FEA context carries
  `model_profile: modal_structural`, `run_kind: modal`, and selected-run
  modal identity, and rejects `linear_static_structural` in that generic
  projection fixture.
- Preserved the existing cache-layout proof that thermal/CFD live state and
  render images stay out of the stable cached prefix and late in the model
  frame.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context projects_fea_workflow_snapshot_into_current_turn_state -- --nocapture`
  passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context projection -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue the completion audit for any other generic context, report, replay,
  or composed-flow fixtures that still use structural stress evidence where a
  full-family or non-stress proof is intended.

### 2026-07-11: Non-Stress Harness FEA Model-Frame Fixture Slice 1

Scope completed:

- Reworked the ah-core FEA model-frame projection fixture from a mixed
  linear-static/stress monitor setup into a coherent modal study/run fixture.
- Added assertions that the model request contains `modal_structural` for both
  study and selected-run analysis identity, and explicitly does not contain
  `linear_static_structural` or the old `figure:run_fea_1:stress` id.
- Fixed the ah-core mock runtime test import drift for
  `RuntimeInspectionSelector`, which was exposed by compiling the focused test.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core user_turn_context_projects_fea_state_into_model_request -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue final audit for structural/stress-only fixtures that are still being
  used as generic acceptance evidence rather than explicit structural-path
  coverage.

### 2026-07-11: Shared Desktop FEA Terminology Boundary Slice 1

Scope completed:

- Removed the duplicated agent-context `load condition` to `driving condition`
  text normalizer and routed guided-pane study blockers through the shared
  `runtime/domain/fea-terminology` helper.
- Verified stale runtime/readiness wording still becomes driving-condition
  wording in the FEA agent context, while runtime bridge FEA check shaping keeps
  the same normalization behavior.
- Rechecked the agent/FEA UI surfaces so the only remaining desktop regex
  normalization for this vocabulary now lives in the shared runtime-domain
  helper.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/use-fea-agent-context-state.spec.tsx src/app/components/agent/fea-agent-context.spec.ts`
  passed from `../runmat-private/desktop`.
- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts -t "normalizes FEA checks"`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue final audit for duplicated TS-side FEA normalization or profile
  interpretation logic, especially where guided pane, runtime bridge, replay,
  and report code all touch the same study/run facts.

### 2026-07-11: Physics-Agnostic Agent Check Bridge Slice 1

Scope completed:

- Moved generic agent `check` dispatch into the desktop runtime domain bridge
  helper instead of keeping the `.fea`/`.m` routing in `RuntimeProvider`.
- Kept `.fea` validation behind the generic check surface while preserving raw
  profile/family metadata from the runtime check result, so the bridge does not
  interpret studies as linear structural/stress cases.
- Added bridge coverage for a CFD transient `.fea` study check and a RunMat
  script check through injected file reads/static validation.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts -t "check"`
  passed from `../runmat-private/desktop`.
- `npm test -- --run --reporter=dot src/runtime/runtime-provider.spec.tsx -t "generic agent check"`
  passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue the final audit for any remaining UI, harness, replay, report, or
  persistence paths that accidentally assume linear stress instead of selecting
  behavior from the Rust-owned full physics profile catalog.

### 2026-07-11: Retired FEA Check Tool Guard Slice 1

Scope completed:

- Added `finite_element_study_check` to the negative model-visible tool guards
  in both `ah-tools` and `ah-context`, matching the final design where FEA
  validation routes through the generic `check` tool.
- Verified the generated/Rust `.fea` study operation contract still exposes
  final driving-condition operation names and `set_outputs`, with retired
  load-condition names limited to negative assertions or historical progress
  notes.
- Rechecked retired FEA fork/run/result tool names; remaining hits are negative
  catalog assertions or internal runtime/client names, not model-visible tools.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context tools -- --nocapture`
  passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tool -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.

Remaining:

- Continue final requirement audit around lower-level runtime client adapter
  names such as `checkFeaStudy`/`runFeaStudy`: they are not model-visible tools,
  but should remain thin internal bindings behind generic agent `check` and
  `execute`.

### 2026-07-11: FEA Context Run-Kind/Family Guardrail Slice 1

Scope completed:

- Tightened the agent context projection test fixture so `physics_family` and
  `run_kind`/`analysis_run_kind` are modeled as separate values instead of
  letting the family label stand in for execution kind.
- Added explicit CFD projection assertions proving the late, dynamic model-frame
  state carries `physics_family: "CFD"` while preserving `run_kind: "cfd"` and
  `analysis_run_kind: "cfd"`.
- Kept the cache-layout assertion intact: live profile/output/render state
  remains out of the stable cached prefix and appears late in the current-turn
  frame.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context projection -- --nocapture`
  passed from `../runmat-private`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` completed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final requirement audit for any places where profile id, physics
  family, run kind, requested outputs, fields, or figures are conflated before
  reaching the model, guided pane, report path, persistence, or replay.

### 2026-07-11: Full Physics Profile Authoring Guardrail Slice 1

Scope completed:

- Strengthened the Rust-owned `.fea` typed authoring/readiness test from
  representative family coverage to explicit coverage for every current
  supported physics profile in `ANALYSIS_PHYSICS_PROFILE_CATALOG`.
- Added a catalog-vs-test-matrix assertion so adding, removing, or renaming a
  profile now fails this authoring/readiness test until the full-profile setup
  path is updated.
- Covered structural static, transient structural, nonlinear structural,
  modal, thermal, electromagnetic, acoustic, CFD steady, CFD transient,
  thermo-mechanical, electro-thermal, CHT, and FSI profile authoring through the
  same typed operations: create, add region, add material/media, assign
  material/media, add boundary/constraint, add driving condition when required,
  and set outputs.
- Kept family-specific material/media assertions so non-structural thermal,
  electromagnetic, acoustic, CFD, CHT, and electro-thermal paths do not
  accidentally require structural Young's modulus or Poisson ratio.

Tests/evidence:

- `cargo test -p runmat-runtime typed_authoring_creates_edits_and_checks_every_supported_physics_profile --lib -- --nocapture`
  passed.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.
- `cargo fmt` completed.
- `git diff --check` passed in `runmat-analysis`.

Remaining:

- Continue final audit for any model-facing harness, guided-pane, report, or
  replay code that still treats linear structural stress/displacement as the
  generic FEA path instead of deriving behavior from the selected profile and
  requested outputs.

### 2026-07-11: FEA Run Terminology Normalization Slice 1

Scope completed:

- Added a shared desktop runtime-domain FEA terminology normalizer so stale
  `load condition` wording is rewritten to `driving condition` at one boundary
  instead of being patched independently in each consumer.
- Updated the agent runtime bridge check shaping to use that shared normalizer.
- Updated the FEA run orchestrator to normalize check validation, check
  diagnostics, run diagnostics, progress/result payloads, event diagnostics,
  event progress messages, runtime log messages, and caught failure text before
  storing them on execution sessions or persisted run artifacts.
- Strengthened the validation-failed FEA run persistence test to prove both the
  in-memory completed session and persisted diagnostics dataset no longer carry
  stale `load condition` wording.

Tests/evidence:

- `npm test -- --run --reporter=dot src/run/fea-run-orchestrator.spec.ts`
  passed.
- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts`
  passed.
- `npm run typecheck` passed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.
- `rg -n "load condition|load conditions" desktop/src/run/fea-run-orchestrator.ts desktop/src/runtime/domain/agent-runtime-bridge.ts desktop/src/runtime/domain/fea-terminology.ts desktop/src/run/fea-run-orchestrator.spec.ts desktop/src/runtime/domain/agent-runtime-bridge.spec.ts`
  now shows only the shared normalizer and regression fixtures/assertions.

Remaining:

- Continue final completion audit. Generic `check` and persisted FEA run/replay
  data now use the same driving-condition terminology boundary instead of
  cleaning only the model-facing check result.

### 2026-07-11: Agent Runtime Bridge Domain Dispatch Slice 1

Scope completed:

- Moved the agent runtime bridge command dispatch switch out of
  `runtime-provider.tsx` and into the runtime-domain bridge helper.
- Kept React-owned responsibilities in the provider: current state wiring,
  figure monitor slot updates, logging, Tauri listener lifecycle, and response
  delivery.
- Centralized model-facing command parsing for `execute`, `execute_file`,
  `check`, `finite_element_study_operation`, `fea_capabilities`,
  `geometry_render`, `runs`, `select_run`, `figures`, `variables`, `variable`,
  and `show_figures` at the domain boundary.
- Added domain-level dispatcher coverage proving `execute_file`, `check`,
  `geometry_render`, `runs`, `select_run`, and lazy FEA field `variables`
  continue to route through the generic runtime surface.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts`
  passed.
- `npm test -- --run --reporter=dot src/runtime/runtime-provider.spec.tsx -t "dispatches agent execute_file for .fea through the finite element runner"`
  passed.
- `npm test -- --run --reporter=dot src/runtime/runtime-provider.spec.tsx -t "renders geometry views for agent geometry session tools"`
  passed.
- `npm test -- --run --reporter=dot src/runtime/runtime-provider.spec.tsx -t "selects runtime sessions from the wasm runtime bridge by session id"`
  passed.
- `npm run typecheck` passed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final completion audit. RuntimeProvider is still broad overall, but
  the model-visible runtime command semantics now live with the domain bridge
  helpers instead of in React provider orchestration.

### 2026-07-11: Full-Physics Agent Guidance Slice 1

Scope completed:

- Updated the raw `finite_element_study_create` schema guidance so the model
  must choose `model_profile` from `fea_capabilities.physicsProfiles` without
  assuming structural, stress, or any other profile unless the user's
  engineering question calls for it.
- Updated the agent context metadata wording to match the same catalog-driven,
  full-physics instruction.
- Replaced a hardcoded desktop FEA welcome family list with the generated
  `FEA_SUPPORTED_PHYSICS_FAMILIES` summary.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke`
  passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context create_study_metadata_requires_explicit_physics_profile`
  passed.
- `npm test -- --run --reporter=dot src/app/components/agent/fea-agent-state.spec.ts`
  passed.
- `npm run typecheck` passed.

Remaining:

- Continue final acceptance consolidation. The model-facing create path and
  desktop welcome surface now point at the Rust-owned physics profile catalog
  instead of treating linear stress or structural studies as the assumed center.

### 2026-07-11: Desktop Geometry Render State Preservation Slice 1

Scope completed:

- Extended the public TypeScript `GeometrySceneImageOptions` contract with
  optional visibility and section render state.
- Preserved normalized geometry render visibility and section state through the
  desktop agent runtime bridge summary and image-render request.
- Preserved the same state through browser worker and Tauri client payloads.
- Updated the native Tauri geometry image command to accept the render-state
  payload, while leaving visual application to the lower-level renderer once it
  supports those controls.
- Fixed the native desktop runtime bridge to match the current generic
  `Variables` and `Figures` request variants, forward their inspection
  selectors, and send canonical `session_id` for `select_run`.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts -t "renders bounded geometry view images"`
  passed.
- `npm test -- --run --reporter=dot src/runtime/runtime-provider.spec.tsx -t "renders geometry views for agent geometry session tools"`
  passed.
- `npm test -- --run --reporter=dot src/runtime/runtime-provider.spec.tsx -t "selects runtime sessions from the wasm runtime bridge by session id"`
  passed.
- `cargo test --manifest-path desktop/src-tauri/Cargo.toml runtime_render_geometry_scene_image`
  passed as a compile check.
- `cargo fmt --manifest-path desktop/src-tauri/Cargo.toml --all` passed.
- `npm run typecheck` passed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final acceptance consolidation. Model-driven camera, visibility, and
  section reductions now survive the composed desktop bridge, and generic
  variables/figures/select-run dispatch remains aligned for both FEA and script
  runs.

### 2026-07-11: Geometry Render State Forwarding Slice 1

Scope completed:

- Extended the shared harness runtime `GeometryRenderRequest` with typed,
  optional bounded visibility and section state.
- Taught the geometry session store to convert current session visibility and
  section state into that runtime request shape.
- Updated `geometry_render` so model-driven `geometry_set_visibility` and
  `geometry_section` mutations are forwarded to the runtime render backend,
  instead of only changing local JSON state.
- Added focused `ah-tools` coverage proving a render after isolate and section
  forwards the current view, isolated ids, and section plane label.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke`
  passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-runtime-adapter-direct delegates_unified_fea_and_geometry_runtime_surface`
  passed.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` passed.
- `npm run typecheck` passed.

Remaining:

- Continue final acceptance consolidation. GeometrySession camera, visibility,
  and section mutations now have a real path into the current rendered state
  used by the model-guided FEA pane.

### 2026-07-11: Physics Catalog Create-Schema Guard And Copy Contract Slice 1

Scope completed:

- Added raw `finite_element_study_create` schema guidance that requires
  `model_profile` to come from `fea_capabilities.physicsProfiles` and explicitly
  tells the model not to assume linear static structural analysis by default.
- Added `ah-tools` schema coverage for that model-facing create-study guidance,
  complementing the existing metadata-level catalog guard.
- Renamed the copy tool's copy-size payload from generic `bytes` to explicit
  `bytes_copied` and kept desktop progress summaries readable for that field.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke`
  passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools filesystem_tools_mutate_within_scope`
  passed.
- `npm run typecheck` passed.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` passed.

Remaining:

- Continue final acceptance consolidation. The create-study model-facing schema
  now reinforces full-physics catalog selection at the raw tool boundary, and
  copy/open/select primitives now have clearer final semantics for study forks.

### 2026-07-11: Workspace Open Versus Select Semantic Alignment Slice 1

Scope completed:

- Aligned the `ah-tools` workspace test fixture with the target contract:
  `open_path` opens a path visibly without selecting it as the active agent work
  target.
- Aligned the desktop browser-agent workspace bridge fixture so `open_path`
  returns `selected: false` and only `select_path` returns selected state.
- Rechecked remaining matching fixtures; remaining `selected: true` hits are
  select-path or selected-run expectations, not open-path semantics.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools workspace_tools_open_and_select_paths`
  passed.
- `npm test -- --run --reporter=dot src/agent/clients/browser/index.spec.ts -t "routes worker workspace bridge requests"`
  passed.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` passed.
- `npm run typecheck` passed.

Remaining:

- Continue final acceptance consolidation. Phase 1 workspace primitives now have
  consistent low-level, browser bridge, desktop bridge, model metadata, and
  composed fork-flow evidence for the open-vs-select distinction.

### 2026-07-11: Execute Target-Specific Schema Evidence Slice 1

Scope completed:

- Tightened the `ah-tools` runtime smoke coverage for the generic `execute`
  tool schema so the target-specific variants prove their required fields.
- The test now asserts `target=inline` requires `code` plus `target`, and
  `target=file` requires `path` plus `target`, matching the plan requirement
  for unambiguous file execution of `.m`, notebook, and `.fea` artifacts.
- Confirmed the existing composed evidence already routes `.fea` execution
  through the finite element runner rather than source-text execution.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke`
  passed.

Remaining:

- Continue final acceptance consolidation. The central `execute({ target:
  "file", path })` model-facing schema is now explicitly guarded at the harness
  tool-schema level.

### 2026-07-11: Generic Show Figures Non-Selected FEA Run Contract Slice 1

Scope completed:

- Clarified model-facing `show_figures` guidance so the agent uses stable
  figure ids returned by `figures()` and understands the host may select the
  owning run when a requested figure belongs to a non-selected run.
- Added desktop runtime-domain coverage proving `show_figures` can show an
  acoustic FEA result figure by stable artifact figure id while a script run is
  selected, returning the FEA run/session context and monitor image.
- Added ah-context metadata regression coverage so the generic figure surface
  stays documented as run-agnostic instead of selected-run-only.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context show_figures_metadata_describes_non_selected_run_figures`
  passed.
- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts`
  passed with 1 file and 19 tests.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` passed.
- `npm run typecheck` passed.

Remaining:

- Continue final acceptance consolidation. Generic variables/figures/show_figures
  now have explicit non-selected FEA run semantics across model guidance and the
  desktop bridge domain.

### 2026-07-11: Desktop CAD-Start Full-Profile Selection Coverage Slice 1

Scope completed:

- Enabled FEA authoring inside the geometry preview surface spec so the
  CAD-start study creation controls are exercised rather than only the hidden
  feature-off path.
- Added coverage that the physics selector exposes every current generated
  public profile id before study creation.
- Proved a non-structural coupled selection (`fsi_coupled`) is passed through
  the geometry-to-study creation callback, preserving the explicit model
  profile instead of falling back to structural defaults.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/geometry/geometry-preview-surface.spec.tsx`
  passed with 1 file and 5 tests.
- `npm run typecheck` passed.

Remaining:

- Continue the final acceptance audit. CAD-start now has direct UI evidence
  for full-catalog profile selection, complementing the composed harness
  full-catalog typed study creation coverage.

### 2026-07-11: Desktop Guided Pane Full-Profile Label Coverage Slice 1

Scope completed:

- Tightened desktop guided status-panel coverage from representative family
  samples to every current Rust/generated public physics profile id.
- Added assertions that the visible guided step labels remain profile-aware for
  linear static structural, transient structural, nonlinear structural, modal,
  thermal, thermo-mechanical, electro-thermal, electromagnetic, acoustic, CFD
  steady/transient, CHT, and FSI profiles.
- Kept the test catalog-driven so future profile additions fail visibly instead
  of silently inheriting structural/default guided wording.

Tests/evidence:

- `npm test -- --run --reporter=dot src/app/components/agent/fea-agent-status-panel.spec.tsx`
  passed with 1 file and 5 tests.
- `npm run typecheck` passed.

Remaining:

- Continue final acceptance consolidation. The guided pane now has visible
  label coverage for every current public profile id, while the composed
  harness already covers full-catalog create and generic solve/postprocess
  paths.

### 2026-07-11: Desktop Agent Runtime Inspection Domain Boundary Slice 1

Scope completed:

- Moved agent bridge run/session inspection resolution for variables, figures,
  and materialization out of `runtime-provider` and into the runtime-domain
  `agent-runtime-bridge` helper.
- Added domain helpers for resolving explicit `session_id` / `run_id`
  selectors, creating variable and figure summaries, materializing FEA fields
  with bounded pages, and materializing ordinary runtime variables.
- Kept the provider responsible for UI effects only: selected-run state,
  monitor slots, and figure display.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx`
  passed with 2 files and 81 tests.
- `npm run typecheck` passed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final acceptance audit and any high-signal boundary extraction
  it identifies. The provider still owns dispatch and UI effects, but FEA field
  inspection/materialization is no longer a UI-owned side path.

### 2026-07-11: Harness Full-Profile Study Creation From Capabilities Slice 1

Scope completed:

- Added composed agent-harness coverage where the model first calls
  `fea_capabilities`, then creates `.fea` studies from one shared geometry for
  every current public physics profile id.
- Covered linear static structural, modal, transient structural, nonlinear
  structural, thermal, thermo-mechanical, electro-thermal, electromagnetic,
  acoustic, CFD steady, CFD transient, CHT, and FSI profile ids through the same
  `finite_element_study_create` tool path.
- Extended the desktop-runtime mock's typed study-operation response with the
  original operation input so tests can prove `model_profile`, `geometry_path`,
  and destination `.fea` path cross the composed agent/runtime boundary.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity agent_can_create_fea_studies_for_full_physics_catalog_from_capabilities`
  passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity`
  passed with 23 tests, including the new full-catalog create test and the
  existing full-family generic solve/postprocess tests.
- `cargo fmt --manifest-path agent-harness/Cargo.toml --all` passed.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final guided-flow acceptance consolidation. Full-family support now
  has composed harness evidence for both creating studies from capabilities and
  solving/postprocessing existing studies through generic runtime surfaces.

### 2026-07-11: Desktop FEA Shared Check Dispatch Slice 2

Scope completed:

- Moved the shared `.fea` path predicate into the runtime-domain check helper
  so agent/runtime bridge code does not keep its own finite-element path test.
- Routed the FEA run orchestrator's pre-run validation through
  `checkRuntimePath({ kind: "auto" })`, keeping the lower-level FEA checker
  contained inside the generic runtime check dispatcher.
- Routed the desktop agent runtime provider's `.fea` `check` implementation
  through the same generic runtime check dispatcher before converting back to
  the model-facing FEA check payload.

Tests/evidence:

- `npm test -- --run --reporter=dot src/runtime/domain/runtime-check.spec.ts src/run/fea-run-orchestrator.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx`
  passed with 4 files and 80 tests.
- `npm run typecheck` passed.
- `rg -n "checkFeaStudy|isFeaStudyPath" ../runmat-private/desktop/src/runtime ../runmat-private/desktop/src/run`
  shows direct `checkFeaStudy` usage only in the dispatcher, client bindings,
  and tests/mocks; runtime provider and run orchestration now use the shared
  dispatcher.

Remaining:

- Continue the final acceptance audit. The generic check boundary now covers
  FEA study review, FEA run orchestration, and agent-facing `.fea` validation,
  which removes the main UI/provider side paths without weakening the
  full-family physics invariant.

### 2026-07-11: Desktop FEA Study Generic Check Dispatch Slice 1

Scope completed:

- Added a desktop runtime-domain `checkRuntimePath` helper that exposes the
  generic `check({ path, kind })` boundary and dispatches `.fea` paths to the
  lower-level FEA checker internally.
- Updated the FEA study review surface to call the generic check helper and
  convert the generic result back to FEA validation only at the render boundary.
- Added regression coverage that `.fea` paths dispatch through the generic
  check helper and unsupported non-FEA paths do not fall back to finite element
  validation.

Tests/evidence:

- `npm test -- --run src/runtime/domain/runtime-check.spec.ts src/app/components/fea/fea-study-surface.spec.tsx`
  passed.
- `npm run typecheck` passed.
- `rg -n "checkFeaStudy|FeaCheckResult" ../runmat-private/desktop/src/app/components/fea/fea-study-surface.tsx`
  returned no matches.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final acceptance consolidation. The FEA study UI now validates
  through the same generic check boundary expected by the agent/runtime plan,
  with the FEA-specific client call contained inside the runtime-domain
  dispatcher.

### 2026-07-11: Desktop FEA Results Generic Field Materialization Slice 1

Scope completed:

- Routed the desktop FEA results pane field preview through the generic
  `useWorkspace().materializeVariable` path instead of calling `getFeaField`
  directly from the pane.
- Extended the desktop workspace materializer's local option/result types so
  FEA field workspace entries preserve paged `offset`, `limit`, and returned
  page metadata behind the generic materialization API.
- Kept the FEA-specific field client hidden inside the workspace materializer,
  matching the same generic variable surface used by the model/runtime bridge
  and Variables pane.

Tests/evidence:

- `npm test -- --run src/app/components/fea/fea-results-pane.spec.tsx src/runtime/hooks/useWorkspace.spec.tsx`
  passed.
- `npm run typecheck` passed.
- `rg -n "getFeaField|FeaFieldResult|FeaRuntime|hasFeaFieldAccess" ../runmat-private/desktop/src/app/components/fea/fea-results-pane.tsx ../runmat-private/desktop/src/app/components/fea/fea-results-pane.spec.tsx`
  returned no matches.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final acceptance consolidation. The result pane no longer has a
  UI-only FEA field materialization side path; FEA fields are inspected through
  the same generic lazy variable/materialization boundary as other run data.

### 2026-07-11: Private Harness Full-Physics Capability Fixture Sync Slice 1

Scope completed:

- Synced the duplicated private agent-harness `physicsProfiles` test fixtures
  with the Rust/generated full physics profile catalog.
- Restored all generated default outputs in the private capability mocks,
  including multi-field outputs for thermal, modal, acoustic, CFD, CHT, FSI,
  electromagnetic, electro-thermal, thermo-mechanical, and structural profiles.
- Removed stale CHT/FSI fixture wording that reduced coupled physics to
  structural-style or fluid-velocity-only examples.

Tests/evidence:

- `cargo test -p ah-tools runtime_tools_smoke` passed.
- `cargo test -p ah-harness --test host_parity agent_can_solve` passed with
  12 FEA solve/postprocess parity tests across structural, modal, transient,
  nonlinear, electromagnetic, thermal, thermo-mechanical, electro-thermal,
  CFD, acoustic, CHT, and FSI paths.
- `rg -n -F -e "coupled fluid flow and heat transfer" -e "flow fields, deformation" -e "fluid velocity, pressure" ../runmat-private/agent-harness/crates/ah-tools/src/tests.rs ../runmat-private/agent-harness/crates/ah-harness/tests/support/host.rs`
  returned no matches.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue final completion audit for any remaining structural-first examples
  that are legitimate tests rather than active defaults. The private harness
  capability surface now exercises the same full profile-family catalog shape
  as the generated desktop/runtime contracts.

### 2026-07-11: WASM Registry Explicit FEA Profile Sync Slice 1

Scope completed:

- Regenerated the wasm builtin registry after changing runtime FEA constructor
  metadata and explicit-profile requirements.
- Updated the generated registry source fingerprint and entry count so the
  wasm/plot-web surface stays synchronized with current Rust builtin metadata.

Tests/evidence:

- `node scripts/regenerate-wasm-registry.mjs` passed and wrote
  `crates/runmat-runtime/src/builtins/generated_wasm_registry.rs` with 1506
  registry entries and 665 builtins.
- `rg -n "Author a linear static|region selectors, ForceN" crates/runmat-runtime/src/builtins/generated_wasm_registry.rs crates/runmat-runtime/src/builtins/fea/mod.rs`
  returned no matches.

Remaining:

- Continue final generated-surface audit for TS package exports and private
  desktop/harness model-facing metadata.

### 2026-07-11: Runtime Refinement Label Contract Cleanup Slice 1

Scope completed:

- Removed duplicated profile/run-kind string label maps from `.fea` refinement
  validation and runtime adaptive mesh artifact generation.
- Routed refinement validation errors, persisted `analysis_profile`, persisted
  `run_kind`, and default adaptive indicator lookup through the contract-owned
  `AnalysisCreateModelProfile::as_snake_case()` and
  `AnalysisRunKind::as_snake_case()` helpers.

Tests/evidence:

- `cargo test -p runmat-runtime fea_document_refinement_indicator_applicability_matches_profile_context`
  passed.
- `cargo test -p runmat-runtime fea_document_mesh_options_accept_physics_refinement_namespaces`
  passed.
- `cargo test -p runmat-runtime fea_document_mesh_options_reject_unknown_refinement_indicators`
  passed.
- `cargo test -p runmat-runtime append_solved_adaptive_mesh_summary_uses_thermal_element_vector_fields`
  passed.
- `cargo test -p runmat-runtime append_solved_adaptive_mesh_summary_rejects_missing_analysis_profile`
  passed.
- `cargo test -p runmat-runtime append_solved_adaptive_mesh_summary_uses_persisted_analysis_context`
  passed.

Remaining:

- Continue final audit for duplicated profile/run-kind serialization in tests or
  lower-level fixtures. Runtime refinement metadata now shares the same profile
  serialization contract as `.fea` authoring and generated capabilities.

### 2026-07-11: Runtime FEA Constructor Explicit Profile Slice 1

Scope completed:

- Removed the hidden linear-static structural fallback from the older runtime
  `fea.study`, `fea.model`, and `fea.authorStudy` constructors.
- Made `Profile` required for those runtime-user-facing constructors, with
  errors pointing callers to `fea.capabilities().physicsProfiles`.
- Kept `RunKind` as an optional consistency check: when provided, it must match
  the selected profile's derived run kind.
- Updated `fea.authorStudy` tests to pass explicit profiles and reframed its
  builtin summary as typed FEA authoring rather than linear-static authoring.

Tests/evidence:

- `rg -n 'default_profile_for_run_kind|unwrap_or\(AnalysisCreateModelProfile::LinearStaticStructural\)|let mut profile = AnalysisCreateModelProfile::LinearStaticStructural|Author a linear static' crates/runmat-runtime/src/builtins/fea`
  returned no matches.
- `cargo test -p runmat-runtime fea_study_requires_profile` passed.
- `cargo test -p runmat-runtime fea_model_requires_profile` passed.
- `cargo test -p runmat-runtime requires_profile` passed.
- `cargo test -p runmat-runtime typed_constructors_build_full_study_and_sweep_objects`
  passed.
- `cargo test -p runmat-runtime builtins::fea::author_study` passed.

Remaining:

- Continue final acceptance consolidation. Runtime FEA constructors no longer
  silently interpret omitted physics as linear static structural.

### 2026-07-11: Runtime Study ID Normalization Cleanup Slice 1

Scope completed:

- Removed the stale `fea_study` fallback from runtime `.fea` document authoring
  ID normalization.
- Made explicit study, region, material, boundary, and driving-condition IDs
  fail clearly when they do not normalize to a stable YAML key.
- Kept only a neutral `study` fallback for unusable path stems during new study
  creation.
- Added a catalog-driven authoring regression proving every Rust-owned
  supported physics profile can create a `.fea` study, persist its derived run
  kind, publish default outputs, and map to readiness checks. Runtime authoring
  now uses contract-owned profile parsing, profile string serialization, catalog
  lookup, and run-kind serialization instead of its own profile/run-kind string
  table.

Tests/evidence:

- `cargo test -p runmat-runtime document_authoring_accepts_every_supported_physics_profile`
  passed.
- `cargo test -p runmat-runtime physics_profile_catalog_covers_every_supported_profile_once`
  passed.
- `cargo test -p runmat-runtime readiness_uses_profile_specific_requirements_across_supported_physics`
  passed.
- `cargo test -p runmat-runtime typed_authoring_creates_edits_and_checks_representative_physics_families`
  passed.
- `cargo test -p runmat-runtime neutral_study_id_fallback` passed.
- `cargo test -p runmat-runtime normalize_to_stable_yaml_keys` passed.
- `rg -n "fea_study" crates/runmat-runtime/src/analysis/fea_document_authoring.rs`
  found only function/test identifier references, not a persisted or
  model-facing fallback value.

Remaining:

- Continue final full-family acceptance consolidation and requirement audit.
  Runtime authoring no longer leaks an old meta/default study name through
  generated `.fea` sources.

### 2026-07-11: Desktop Readiness Copy Fixture Cleanup Slice 1

Scope completed:

- Removed the remaining desktop runtime-provider test fixture wording that
  described readiness as missing a generic `load`.
- Reworded it to `driving condition` while leaving the durable `.fea` YAML
  `loads:` storage key isolated as an internal serialization detail.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed
  from `../runmat-private/desktop` with 63 tests.
- `rg -n "boundary condition, load|load, and mesh|Add a boundary condition, load" desktop/src/runtime desktop/src/app/components/agent desktop/src/app/components/fea`
  returned no matches from `../runmat-private`.
- `git diff --check` passed in both repos.

Remaining:

- Continue final requirement audit, with emphasis on any live user/model-facing
  wording or generated contract surface that still preserves old load-condition
  terminology outside explicit backward-normalization tests.

### 2026-07-11: Composed Harness Boundary Copy Cleanup Slice 1

Scope completed:

- Removed stale `Use selected fixed area` wording from the composed
  agent-harness guided-region context fixture.
- Updated the guided region workflow prompt to refer to the current selection
  as a constraint or boundary region while preserving structural `fixed`
  semantics only where the user explicitly asks for a fixed constraint.
- Updated the desktop runtime adapter test host used by host parity coverage to
  accept selector-carrying `variables` and `figures` requests after the generic
  run-scoped inspection contract change.

Tests/evidence:

- `cargo test -p ah-harness --test host_parity agent_can_turn_selected_geometry_region_into_typed_fea_constraint`
  passed from `../runmat-private/agent-harness`.
- `cargo test -p ah-harness --test host_parity` passed from
  `../runmat-private/agent-harness` with all 22 composed host parity tests,
  covering the generic runtime postprocess surface across structural, modal,
  transient structural, nonlinear structural, thermal, electromagnetic,
  acoustic, CFD, CHT, FSI, coupled, replay, report, and fork/open/select flows.
- Targeted search found no positive `Use selected fixed area`, `fixed area`, or
  `loaded area` wording in `../runmat-private/agent-harness` composed harness
  tests or the desktop browser agent client. Remaining `load condition` hits are
  negative assertions or backward-normalization fixtures.

Remaining:

- Continue the final full-family acceptance audit across generated contracts,
  replay/persistence, and user-facing guided workflow surfaces.

### 2026-07-11: Harness Run-Scoped Inspection Contract Slice 1

Scope completed:

- Added a generic runtime inspection selector carrying optional `session_id` and
  `run_id` through the Rust agent-harness runtime interface.
- Made the model-visible `variables` and `figures` tools accept those selectors,
  and updated tool metadata so the model knows when to inspect a specific run
  without changing selection.
- Forwarded the selector through the desktop runtime adapter and web host bridge;
  updated direct/local/mock runtime implementations to keep the trait boundary
  coherent.
- Added regression coverage proving selector payloads reach the runtime layer
  and that FEA field identity still survives materialization.

Tests/evidence:

- `cargo test -p ah-tools runtime_tools_smoke` passed from
  `../runmat-private/agent-harness`.
- `cargo test -p ah-runtime-adapter-desktop variables_and_materialize_var_preserve_fea_field_selectors`
  passed from `../runmat-private/agent-harness`.
- `cargo test -p ah-context` passed from `../runmat-private/agent-harness`.
- `cargo test -p ah-runtime-adapter-direct` passed from
  `../runmat-private/agent-harness`.
- `cargo test -p ah-cli --lib` passed from
  `../runmat-private/agent-harness`.
- `cargo test -p ah-web-host --lib` passed from
  `../runmat-private/agent-harness`.
- `git diff --check` passed in both repos.

Remaining:

- Continue final full-family acceptance audit. The generic runtime inspection
  surface now supports structural, modal, thermal, electromagnetic, acoustic,
  CFD, CHT, FSI, and coupled result sets by run/session identity rather than by
  selected-state side effects.

### 2026-07-11: Desktop Session-Scoped Inspection Commands Slice 1

Scope completed:

- Made the desktop agent runtime bridge honor explicit `session_id` and `run_id` selectors for generic `variables` and `figures`, matching the composed browser-client payloads used after `select_run`.
- Shared the same projected `ExecutionSession` resolver across `variables`, `variable`, and `figures` so selected-run drift cannot make generic inspection commands read from the wrong script/FEA run.
- Added regression coverage where a script run is selected but `variables`, `figures`, and `variable` all inspect a replayed acoustic FEA run through `session_id`.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 63 focused runtime-provider tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final composed runtime-surface audit. The desktop bridge no longer depends on React selected-run propagation timing for the generic FEA variable/figure inspection sequence.

### 2026-07-11: Desktop Select Run Contract Alignment Slice 1

Scope completed:

- Aligned the desktop agent runtime bridge `select_run` command with the model-visible tool contract by accepting `session_id`.
- Kept `sessionId` as a defensive compatibility fallback inside the in-process bridge, but made the public error message and regression use `session_id`.
- Tightened the run-selection test so generic selected-run inspection follows the same snake-case contract used by the harness tool schema and browser agent client.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 63 focused runtime-provider tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final composed runtime-surface audit. The generic `select_run` path now matches the model-visible schema used before `variables`, `variable`, `figures`, and `show_figures`.

### 2026-07-11: Desktop Run-Scoped Variable Materialization Slice 1

Scope completed:

- Made the desktop agent runtime bridge honor explicit `session_id` and `run_id` selectors on the generic `variable` tool instead of always materializing against the currently selected run.
- Routed explicit variable selectors through the matching projected `ExecutionSession` workspace snapshot before deciding whether the request is a finite element field page or an ordinary runtime variable.
- Added regression coverage where a script run is selected but the agent materializes an acoustic FEA field from a different replayed FEA session using `session_id` plus `field_id`.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 63 focused runtime-provider tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final runtime-surface acceptance consolidation. Generic `variable` materialization now respects stable run/session identity for finite element fields instead of relying on ambient selected-run state.

### 2026-07-11: Desktop Structural Boundary Copy Generalization Slice 1

Scope completed:

- Removed the remaining active desktop guided-workflow copy that treated structural boundary setup as a fixed-area action by default.
- Updated structural setup labels/prompts from fixed-area wording to structural constraint/boundary region wording, while leaving fixed constraints as a possible typed boundary condition when explicitly chosen.
- Added regression coverage so selected structural boundary prompts stay generic and do not reintroduce fixed-area language.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 34 focused tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `rg -n "Use selected fixed area|Pick Fixed Areas|as the fixed area|fixed-area workflow|fixed area" desktop/src/app/components/agent desktop/src/app/components/fea` now returns only negative test assertions in `../runmat-private`.

Remaining:

- Continue final full-family acceptance consolidation. The guided workflow now treats structural as one supported physics family, not as the vocabulary template for the rest of FEA setup.

### 2026-07-11: Runtime Driving Condition Error Vocabulary Slice 1

Scope completed:

- Split internal YAML sequence keys from public error labels in the runtime `.fea` document authoring helpers.
- Routed add/update/remove driving-condition operations through the labeled sequence helpers so missing or duplicate entries report `driving_conditions` instead of the durable YAML storage key `loads`.
- Tightened the focused authoring regression so `update_driving_condition` errors explicitly reject old `loads` wording.

Tests/evidence:

- `cargo fmt -p runmat-runtime` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime fea_document_authoring -- --nocapture` passed from `runmat-analysis` with 10 focused authoring tests.
- `rg -n "loads block does not exist|loads entry does not exist|load condition|load_condition|load_conditions" crates/runmat-runtime/src/analysis/fea_document_authoring.rs bindings/ts/src/generated/fea-study-document-contracts.ts bindings/ts/src/index.ts` returned no matches.

Remaining:

- Continue final audit. The model-facing typed authoring surface no longer leaks old load-condition wording through mutation error messages.

### 2026-07-11: Target Docs Boundary Vocabulary Slice 1

Scope completed:

- Removed the remaining generic fixed-area language from the target guided pane shape and Phase 7 acceptance text.
- Reframed the guided pane step as boundary/support region selection and the acceptance outcome as the selected profile's typed boundary condition or constraint.

Tests/evidence:

- `rg -n "Pick fixed areas|Pick loaded areas|fixed area|loaded area|fixed areas|loaded areas|load-condition|load condition" FEA_GEOMETRY_AGENT_TARGET_DESIGN.md FEA_GEOMETRY_AGENT_CHANGE_PLAN.md FEA_GEOMETRY_AGENT_USER_EXPERIENCE.md FEA_GEOMETRY_AGENT_CONTEXT_LAYOUT.md` returned no matches.

Remaining:

- Continue final acceptance consolidation. The target documents no longer present structural fixed supports as the generic guided FEA step.

### 2026-07-11: Desktop Physics Family Canonicalization Slice 1

Scope completed:

- Canonicalized runtime-provided FEA physics family labels before geometry-start guided choice matching in the desktop FEA agent context.
- Preserved generated catalog display labels while allowing host capability payloads such as `cfd`, `flow`, `coupled`, and `coupled-physics` to map to the same Flow/Coupled guided options as the Rust-owned catalog.
- Added regression coverage proving lowercase/shorthand runtime capability families still expose non-structural Flow and Coupled choices without reintroducing Structural/Thermal choices.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 29 focused tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in `../runmat-private`.

Remaining:

- Continue final guided-flow acceptance consolidation. Runtime-provided capability family casing no longer controls whether supported CFD/coupled workflows appear in the first geometry-start pane.

### 2026-07-11: Adaptive Mesh Profile Contract Slice 1

Scope completed:

- Removed the adaptive mesh append/postprocess fallback that treated missing `analysis_profile` as `linear_static_structural` and missing `run_kind` as `linear_static`.
- Made solved adaptive mesh summary append fail clearly when persisted mesh artifacts lack profile/run-kind metadata, directing regeneration from the typed `.fea` study boundary.
- Updated runtime adaptive mesh fixtures so valid structural and uniform-refinement artifacts carry explicit profile/run-kind metadata.
- Added regression coverage proving stale metadata-free artifacts are rejected instead of being interpreted as structural/stress refinement cases.

Tests/evidence:

- `cargo fmt -p runmat-runtime` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime append_solved_adaptive_mesh_summary -- --nocapture` passed from `runmat-analysis` with 20 focused adaptive summary tests.

Remaining:

- Continue final completion audit. Adaptive refinement evidence now preserves the active physics profile instead of using linear static structural as the hidden postprocess default.

### 2026-07-11: Desktop Geometry Create Profile Slice 1

Scope completed:

- Threaded explicit physics profile selection through desktop geometry-to-study creation so the runtime `create` operation receives `model_profile`.
- Added a visible physics profile selector to the geometry preview surface, populated from the generated Rust-owned `FEA_SUPPORTED_PHYSICS_PROFILES` catalog.
- Added a small shell-domain helper for FEA create operation input assembly that rejects empty profile selections and preserves non-structural profile ids.
- Fixed strict optional desktop FEA capability fallback typing so full desktop typecheck passes with the generated capability shape.

Tests/evidence:

- `npm test -- src/app/components/shell/editor-panel.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/shell/editor-panel.spec.ts src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-state.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final completion audit. Creating a `.fea` file from an opened CAD/STEP file no longer silently falls back to structural physics.

### 2026-07-11: Runtime Author Study Boundary Slice 1

Scope completed:

- Renamed the public runtime author-study intent/evidence boundary from fixed/load/force terminology to boundary-condition/driving-condition terminology.
- Made selected boundary/driving evidence optional where the active scaffold does not provide that concept, and added `selected_driving_condition_kind` so non-structural drivers are represented by their actual source kind.
- Kept structural force input/evidence explicit as `structural_force_n`, and proved modal/electromagnetic authoring does not persist meaningless structural force evidence.
- Updated `fea.authorStudy` Name/Value parsing and builtin descriptor copy to use `BoundaryConditionRegion`, `DrivingConditionRegion`, and `StructuralForceN` style options.

Tests/evidence:

- `cargo fmt -p runmat-runtime` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime analysis_author_study -- --nocapture` passed from `runmat-analysis` with 7 focused runtime author-study tests.
- `cargo test -p runmat-runtime builtins::fea::author_study::tests -- --nocapture` passed from `runmat-analysis` with 5 builtin wrapper tests.
- `git diff --check` passed in `runmat-analysis`.

Remaining:

- Continue final completion audit. The legacy runtime author-study helper no longer exposes structural fixed/load wording as its generic public contract.

### 2026-07-11: Typed Study Create Profile Slice 1

Scope completed:

- Removed the hidden structural default from typed `.fea` study creation by requiring `model_profile` / `modelProfile` in the runtime authoring operation.
- Validated create-time profiles against the Rust-owned profile-to-run-kind mapping and persisted both `model.profile` and derived `run.kind` into new study documents.
- Updated the private agent harness `finite_element_study_create` schema so `model_profile` is required and added model-facing metadata that points the agent to `feaCapabilities().physicsProfiles`.
- Added representative typed authoring coverage across structural, modal, thermal, electromagnetic, acoustic, CFD, and coupled physics families, including family-appropriate materials/media, boundaries, drivers where needed, outputs, and readiness checks.

Tests/evidence:

- `cargo fmt -p runmat-runtime` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime fea_document_authoring -- --nocapture` passed from `runmat-analysis` with 10 focused authoring tests.
- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-context` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context tools -- --nocapture` passed from `../runmat-private`.
- `git diff --check` passed in both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final completion audit. Study creation now follows the selected physics family instead of silently producing structural-first `.fea` files.

### 2026-07-11: Retired Model Tool Guard Slice 1

Scope completed:

- Strengthened `ah-tools` runtime tool tests so retired FEA-specific model tools cannot silently return: `fea_check`, `fea_run`, `finite_element_fork_study`, `finite_element_study_run`, aggregate result/get-field/render-result tools, and old load-condition tools.
- Strengthened FEA-disabled tool registration tests so those retired names are not registered, not spec-visible, and return `NotFound` when invoked.
- Strengthened `ah-context` model-visible tool catalog and guidance tests so retired FEA run/result/fork/load-condition tools and legacy geometry `geometry_inspect`/`geometry_view` names cannot leak into tool metadata or the FEA guidance section.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-context` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tool -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context tools -- --nocapture` passed from `../runmat-private`.

Remaining:

- Continue final completion audit. The model-visible tool surface now has explicit regression guards for the clean break away from FEA-specific run/result side-channel tools.

### 2026-07-11: Runtime Physics Capability Catalog Slice 1

Scope completed:

- Extended `AnalysisRuntimeCapabilities` with a serialized `physicsProfiles` catalog derived from the Rust-owned `ANALYSIS_PHYSICS_PROFILE_CATALOG`.
- Wired wasm `feaCapabilities()` to return the rich profile catalog with profile id, label, family, target, value, and default outputs for the full public physics family set.
- Updated generated TypeScript contracts so `FeaCapabilities.physicsProfiles` uses the generated `FeaPhysicsProfileCatalogEntry` shape without duplicating interface definitions.
- Updated desktop and harness capability mocks from legacy string-only `profiles` arrays to complete `physicsProfiles` payloads.
- Made the desktop FEA agent geometry-start choices consume runtime-provided physics profile families first, falling back to the generated Rust-owned catalog only when capabilities omit the catalog.

Tests/evidence:

- `cargo fmt -p runmat-runtime -p runmat-wasm` passed from `runmat-analysis`.
- `node bindings/ts/scripts/generate-fea-contracts.cjs` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime physics_profile_catalog -- --nocapture` passed from `runmat-analysis`.
- `cargo check -p runmat-wasm` passed from `runmat-analysis`.
- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts --reporter=dot` passed from `bindings/ts`.
- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-harness` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_ -- --nocapture` passed from `../runmat-private` with all 12 full-family host parity solve/postprocess tests passing.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/use-fea-agent-context-state.spec.tsx src/run/fea-run-orchestrator.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.

Remaining:

- Continue final completion audit. The live capability path is now full-family and catalog-driven instead of structural-first or string-list based.

### 2026-07-11: Geometry-Heavy FEA Context Budget Slice 1

Scope completed:

- Added a bounded reducer at the `ah-context` FEA domain-provider boundary so model-visible FEA current-state JSON no longer blindly trusts upstream snapshot size.
- Capped long strings, large arrays, and excessive nested JSON depth in `fea_context` while preserving the existing stable keys for workflow, visual state, selected run identity, mesh/artifact refs, readiness, and supported geometry extensions.
- Added a geometry-heavy context-cost regression with long workflow strings, many blockers, many choices, many visible result figure ids, many supported geometry extensions, and large selected-run summaries. The regression proves the encoded FEA context stays under a bounded payload threshold and truncates repeated arrays/summaries.
- Kept existing raw-topology and cache-layout behavior intact: useful FEA state still projects into the late current-turn block, not the stable prefix.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-context` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context fea_context_attachment_bounds_geometry_heavy_state_for_model_cost -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context fea_context -- --nocapture` passed from `../runmat-private` with 3 focused FEA context tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context keeps_live_physics_state_out_of_cacheable_prefix_and_late_in_frame -- --nocapture` passed from `../runmat-private`.

Remaining:

- Continue the final completion audit. Phase 9 now has direct context-cost evidence in addition to raw-topology payload evidence.

### 2026-07-11: Rust-Owned Physics Catalog Slice 1

Scope completed:

- Moved the supported FEA physics profile catalog policy into `runmat-runtime` contracts as `ANALYSIS_PHYSICS_PROFILE_CATALOG`, covering profile id, label, family, target, user-facing value, and default outputs for every supported public family.
- Extended the TypeScript FEA contract generator so `FEA_SUPPORTED_PHYSICS_PROFILES`, `FEA_SUPPORTED_PHYSICS_FAMILIES`, and `FeaPhysicsProfileCatalogEntry` are emitted from the Rust catalog instead of being hand-maintained in `bindings/ts/src/index.ts`.
- Re-exported the generated catalog from the public TS package and removed the duplicate local TS catalog block, so desktop guided FEA workflow copy consumes the generated Rust-owned family set.
- Added a Rust contract regression proving the catalog covers every supported `AnalysisCreateModelProfile` exactly once and that every profile has non-empty family/copy/default-output metadata.

Tests/evidence:

- `node bindings/ts/scripts/generate-fea-contracts.cjs` passed from `runmat-analysis`.
- `cargo fmt -p runmat-runtime` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime physics_profile_catalog_covers_every_supported_profile_once -- --nocapture` passed from `runmat-analysis`.
- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts --reporter=dot` passed from `bindings/ts`.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `rg -n "export interface FeaPhysicsProfileCatalogEntry|export const FEA_SUPPORTED_PHYSICS_PROFILES|profile: \"linear_static_structural\"" bindings/ts/src/index.ts bindings/ts/src/generated/fea-study-document-contracts.ts` now shows the catalog definition only in the generated TS contract.

Remaining:

- Continue the final completion audit. The guided desktop family workflow no longer depends on a duplicated hand-written TypeScript profile catalog.

### 2026-07-11: Mesh Topology Compaction Regression Slice 1

Scope completed:

- Audited the Phase 9 raw-topology boundary with the full physics-family invariant in mind: geometry context must stay bounded for solver/mesh-heavy structural, thermal, electromagnetic, acoustic, CFD, and coupled studies, not just CAD face/evaluator payloads.
- Strengthened the `ah-tools` large-geometry tool-boundary fixture with raw element connectivity, raw mesh vertex/element arrays, solver topology fields, and render topology fields.
- Strengthened the geometry summary compaction unit test with the same mesh/topology markers and assertions that none of them enter compact model-facing payloads.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-tools` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools geometry_session_tools_do_not_expose_raw_topology_payloads -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools compact_summary_drops_raw_topology_and_evaluator_payloads -- --nocapture` passed from `../runmat-private`.

Remaining:

- Continue the final requirement audit. The geometry context boundary now has direct regression proof against CAD evaluator leakage and solver/mesh topology leakage.

### 2026-07-11: Full-Family Variable Identity Acceptance Slice 1

Scope completed:

- Audited the Phase 8 host-parity solve/postprocess matrix and confirmed it covers the explicit target profile set: linear static structural, modal, transient structural, nonlinear structural, thermal, electromagnetic, acoustic, CFD, thermo-mechanical, electro-thermal, CHT, and FSI-style coupled studies.
- Strengthened the static structural, modal, electromagnetic, and shared family helper assertions so generic `variables` output must expose `session_id` in addition to `run_id`, `field_id`, lazy paging, and finite-element field kind.
- Preserved the existing generic tool sequence proof: `finite_element_study_set_mesh -> check -> execute -> select_run -> variables -> figures -> show_figures -> variable`.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-harness` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness generic_runtime_surfaces -- --nocapture` passed from `../runmat-private` with 14 host-parity tests, covering live solve/postprocess, replay inspection, and report generation.

Remaining:

- Continue the final requirement audit. Phase 8 now has stronger proof that FEA fields are truly generic variable entries with stable run/session/field identity across the supported physics matrix.

### 2026-07-11: Fork Flow Driver Fixture Terminology Slice 1

Scope completed:

- Audited the Phase 1 composed fork/open/select/edit acceptance path and confirmed the general `copy`, `open_path`, `select_path`, and typed FEA driving-condition edit sequence is covered in `ah-harness`.
- Renamed the model fixture tool-call id from `fork_load` to `fork_driver`, the test prompt from "apply a load" to "apply a structural driver", and local assertion variables from load wording to driver/driving-condition wording.
- Preserved behavior: the same composed flow still copies a `.fea` file, opens it without making that action the active target, selects the copy, and applies `finite_element_study_add_driving_condition` to the copied study.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_fork_open_select_and_edit_copied_fea_study -- --nocapture` passed from `../runmat-private`.
- `rg -n "fork_load|load_output|apply a load|load operation output" agent-harness/crates/ah-harness/tests/support/host.rs agent-harness/crates/ah-harness/tests/host_parity.rs` returns no matches from `../runmat-private`.

Remaining:

- Continue the final requirement audit. Phase 1 primitive composition now has both behavior evidence and final-state terminology in the composed FEA fork fixture.

### 2026-07-11: Stale Wasm FEA Results Wrapper Removal Slice 1

Scope completed:

- Audited the Rust wasm API and TypeScript client contract and found the current source exposes bounded `feaField(runId, fieldId, options)` while ignored local wasm web package outputs still advertised the removed aggregate `feaResults(runId)` wrapper.
- Removed the stale `feaResults` wrapper method and `runmatwasm_feaResults` declarations from both `bindings/ts/pkg-web` and `bindings/ts/dist/pkg-web` local package outputs.
- Kept `feaField` as the only local web package field/result access wrapper, matching the target model where finite element result access flows through bounded field paging and generic variable/figure surfaces rather than an aggregate result side channel.

Tests/evidence:

- `rg -n "feaResults|runmatwasm_feaResults" bindings/ts/pkg-web bindings/ts/dist/pkg-web bindings/ts/src bindings/ts/dist crates/runmat-wasm/src/api/session.rs -g '*.rs' -g '*.ts' -g '*.d.ts' -g '*.js'` returns no matches from `runmat-analysis`.
- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts --reporter=dot` passed from `bindings/ts` with 47 tests.

Remaining:

- Continue the final acceptance audit. The Rust source remains the authoritative wasm API; the next full wasm-pack rebuild should naturally preserve the same `feaField`-only wrapper surface.

### 2026-07-11: Structural Guided Driver Label Slice 1

Scope completed:

- Removed the remaining live guided-pane structural `Pick Loaded Areas` step label and replaced it with `Pick Structural Drivers`.
- Reworded the structural driving next action to say `structural driver regions` instead of `loaded regions`.
- Added status-panel regression coverage for the structural driving-condition step so the structural path also rejects the old loaded-area label, not only the non-structural paths.
- Audited old load-condition tool-name references found in `ah-tools` and desktop tests; those are negative assertions or raw-normalization fixtures rather than active model-visible tools.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 32 tests.
- `rg -n "Pick Loaded Areas|loaded regions visually|Add structural loads|structural loads" desktop/src/app/components/agent desktop/src/app/components/fea desktop/src/runtime/domain -g '*.ts' -g '*.tsx'` now returns only negative test assertions from `../runmat-private`.

Remaining:

- Continue the final acceptance audit. The live guided structural path no longer presents structural driving-condition selection with load/loaded-area terminology.

### 2026-07-11: Full-Family UX And Plan Invariant Slice 1

Scope completed:

- Audited the live desktop/harness surfaces for stress-only assumptions at the guided workflow boundary. The current workflow code derives profile class from the Rust-generated supported physics profile catalog, has family-specific copy for structural, modal, thermal, electromagnetic, acoustic, CFD, coupled, and unknown profiles, and has regression coverage for every supported profile family.
- Reframed `FEA_GEOMETRY_AGENT_USER_EXPERIENCE.md` so the guided flow asks for boundary/support areas, source/driver/interface/excitation areas, and family-appropriate values/material/media details instead of presenting fixed/load structural setup as the universal flow.
- Added a target-design invariant that full-family behavior must be catalog-driven across desktop, harness, model context, tool metadata, study review, run summaries, variables, figures, reports, persistence, and replay.
- Tightened Phase 8 acceptance so the composed solve/postprocess matrix covers structural, modal, transient structural, nonlinear structural, thermal, electromagnetic, acoustic, CFD, and coupled profile classes explicitly.

Tests/evidence:

- `rg -n "FEA_SUPPORTED_PHYSICS_PROFILES|FeaWorkflowProfileClass|routes incomplete guided setup through profile-specific requirements for every supported physics profile" desktop/src/app/components/agent/fea-physics-workflow.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/app/components/fea/fea-study-review-model.ts` confirms the live guided workflow and review surfaces are catalog-driven in `../runmat-private`.
- `rg -n "familySpecificDrivingConditionRequests|current_density|acoustic_pressure|flow_velocity|interface_transfer|coupled_source" desktop/src/agent/clients/browser/index.spec.ts` confirms the browser bridge preserves non-structural driver payloads through the same `finite_element_study_operation` channel.
- `rg -n "Pick fixed areas|Pick loaded areas|Enter load and material details|Compare the load cases" FEA_GEOMETRY_AGENT_USER_EXPERIENCE.md` returns no matches from `runmat-analysis`.

Remaining:

- Continue the final acceptance audit. The target plan now explicitly treats every new FEA surface as full-family unless intentionally scoped otherwise.

### 2026-07-11: Driving Condition View Model Terminology Slice 1

Scope completed:

- Renamed the desktop FEA study editor view-model collection from `loads` to `drivingConditions` so React-domain state no longer advertises structural load terminology for generic physics drivers.
- Updated the FEA study review tree to display a `Driving Conditions` section, use neutral driving-condition annotations, and avoid `load` visual roles/colors in the review model.
- Updated guided structural copy from `Add structural loads` to `Add structural drivers` and kept the agent prompt on materials, constraints, driving conditions, and outputs.
- Updated target/UX design notes where generic study setup still referred to loads instead of driving conditions.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/runtime/domain/fea-study-view-model.spec.ts src/app/components/fea/fea-study-review-model.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 34 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `rg -n "structural loads|Add structural loads|Add loads|load-condition|load condition|load conditions|constraints, loads|\\bloads\\b" FEA_GEOMETRY_AGENT_TARGET_DESIGN.md FEA_GEOMETRY_AGENT_USER_EXPERIENCE.md FEA_GEOMETRY_AGENT_CHANGE_PLAN.md FEA_GEOMETRY_AGENT_CONTEXT_LAYOUT.md` returns no matches from `runmat-analysis`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final acceptance audit. Desktop-visible study state and review copy now use driving-condition terminology outside raw compatibility fixtures and old-YAML normalization boundaries.

### 2026-07-11: Local Runtime FEA Contract Alignment Slice 1

Scope completed:

- Aligned the agent-harness local CLI runtime's `.fea` execution summaries with the shared selected-run contract. Local FEA study runs now emit `run_kind: "fea-study"` and FEA sweep runs emit `run_kind: "fea-study-sweep"` instead of underscore spellings.
- Aligned local generic `check` output artifact kinds with finite-element terminology: `finite_element_study` and `finite_element_study_sweep`.
- Added constants and a focused unit guard in `ah-cli` so future local-runtime drift fails quickly.

Tests/evidence:

- `cargo fmt --manifest-path agent-harness/Cargo.toml -p ah-cli` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-cli local_fea_run_kinds_use_shared_agent_contract_values -- --nocapture` passed from `../runmat-private`.
- `rg -n "fea_study|fea_study_sweep" agent-harness/crates/ah-cli/src/local_env.rs desktop/src/runtime/domain desktop/src/runtime/runtime-provider.spec.tsx` returns no matches from `../runmat-private`.
- `rg -n "fea_study|fea_study_sweep" FEA_GEOMETRY_AGENT_TARGET_DESIGN.md FEA_GEOMETRY_AGENT_CONTEXT_LAYOUT.md` returns no matches from `runmat-analysis`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final plan-wide completion audit. Desktop, harness parity, and local CLI runtime surfaces now agree on the shared FEA run-kind spelling and model-facing local check artifact naming.

### 2026-07-11: Agent Check Terminology Boundary Slice 1

Scope completed:

- Updated `FEA_GEOMETRY_AGENT_TARGET_DESIGN.md` and `FEA_GEOMETRY_AGENT_CONTEXT_LAYOUT.md` so run-kind examples use the shared `fea-study` value instead of the removed `fea_study` variant.
- Hardened desktop agent-facing `.fea` check result shaping so blockers, warnings, and diagnostics normalize stale "load condition" wording to "driving condition" before being returned through the generic `check` bridge.
- Kept the raw check payload unchanged for provenance/debugging while cleaning the shaped model-facing fields.
- Added regression coverage where raw validation issues and diagnostics contain old load-condition wording and the shaped `blockers` output uses driving-condition terminology.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 13 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `rg -n "fea_study" FEA_GEOMETRY_AGENT_TARGET_DESIGN.md FEA_GEOMETRY_AGENT_CONTEXT_LAYOUT.md` returns no matches from `runmat-analysis`.
- `rg -n "load condition|load conditions" desktop/src/runtime/domain/agent-runtime-bridge.ts desktop/src/runtime/domain/agent-runtime-bridge.spec.ts desktop/src/app/components/agent/fea-agent-context.ts` now finds only regression fixtures and terminology-normalization helpers.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final completion audit. The check/readiness boundary now consistently presents driving-condition terminology to the agent, but the full goal still needs final plan-wide verification before completion can be claimed.

### 2026-07-11: Agent Run-Kind Contract Alignment Slice 1

Scope completed:

- Audited the selected-run/runtime contract and found a current desktop-only drift where `ExecutionSession.runKind === "fea-study"` was converted to agent-facing `run_kind: "fea_study"`, while the shared runtime artifact contract and harness host parity use `fea-study`.
- Updated the desktop agent runtime bridge to return the shared `FEA_RUN_KIND` value for FEA runs, so `runs`, `figures`, and related agent-facing summaries use the same hyphenated run-kind identity as persisted run manifests and harness generic runtime surfaces.
- Updated bridge/provider tests that had encoded the underscore variant.
- Verified remaining `fea_study` occurrences in desktop agent/app code are telemetry/logger event names rather than run-kind contract values.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 76 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `rg -n "fea_study" desktop/src/runtime desktop/src/agent desktop/src/app/components/agent` now only finds the `agent.fea_study_summary_failed` logger event name.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final completion audit. The generic selected-run contract is now aligned across desktop agent bridge, shared run manifests, and harness expectations.

### 2026-07-11: Generated Operation Contract Clean-Break Slice 1

Scope completed:

- Audited the current TypeScript FEA document operation contract and found the Rust-owned source, package source, and current `fea-study-document-contracts` dist output use the new driving-condition operation names.
- Removed stale ignored `bindings/ts/dist/generated/fea-study-document-operations.*` artifacts that still advertised removed `add_load_condition` / `set_load_condition` / `remove_load_condition` names from an earlier generated-file shape.
- Rebuilt TypeScript contracts and confirmed the package no longer regenerates or exposes the orphan operation-contract file; the single public generated contract path now advertises `add_driving_condition`, `update_driving_condition`, and `remove_driving_condition`.

Tests/evidence:

- `rg -n "add_load_condition|set_load_condition|remove_load_condition|fea-study-document-operations" bindings/ts/src bindings/ts/dist bindings/ts/scripts` returns no matches from `runmat-analysis`.
- `npm test -- src/index.spec.ts` passed from `bindings/ts` with 47 tests.
- `npm run build:types` passed from `bindings/ts`.

Remaining:

- Continue the final completion audit. This closes a stale generated-contract leak, but does not by itself prove the full guided mesh/solve/postprocess flow.

### 2026-07-11: Full-Family Guided Bridge Acceptance Slice 1

Scope completed:

- Tightened the ah-tools capability smoke test so the model-visible `fea_capabilities` profile assertion covers modal, transient structural, nonlinear structural, thermo-mechanical, and electro-thermal profiles in addition to the already-covered structural, thermal, electromagnetic, acoustic, CFD, CHT, and FSI profiles.
- Expanded the desktop browser-agent composed guided FEA bridge test so `finite_element_study_operation` carries family-specific driver/source payloads unchanged through the same runtime bridge path: structural force, thermal heat source, electromagnetic current density, acoustic pressure, CFD inlet velocity, CHT-style interface transfer, and FSI-style coupled source.
- Kept the bridge assertion explicitly negative for the old `add_load_condition` operation name so the model-facing surface stays on the less-ambiguous driving-condition vocabulary.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke -- --nocapture` passed from `../runmat-private`.
- `npm test -- src/agent/clients/browser/index.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 20 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final requirement-by-requirement audit. The exposed capability and browser bridge layers now have full-family vocabulary proof; the remaining work should focus on any missing composed guided-flow acceptance evidence rather than broad cleanup.

### 2026-07-11: Desktop Material/Media Review Slice 1

Scope completed:

- Regenerated the public TypeScript FEA document contracts so `FeaStudyMaterialEntry` includes thermal, electromagnetic, acoustic, and fluid summary fields from the Rust-owned typed study document result.
- Rebuilt TypeScript declarations used by the desktop app's `runmat` path alias so desktop typechecking sees the regenerated material/media contract.
- Updated the desktop FEA study review model to choose the first available material/media summary instead of hardcoding `mechanicalSummary` or a `"mechanical"` fallback.
- Added desktop review coverage proving a thermal medium renders with thermal summary details and does not show Young's modulus as the default child row.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts` and refreshed generated FEA contracts plus declarations.
- `npm test -- src/index.spec.ts` passed from `bindings/ts` with 47 tests.
- `npm test -- src/app/components/fea/fea-study-review-model.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 5 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final composed-flow audit. The review surface now displays typed non-structural material/media summaries, but final acceptance still needs proof across representative guided solve/result paths.

### 2026-07-11: Full-Family Material/Media Authoring Slice 1

Scope completed:

- Removed the structural-only requirement from typed `.fea` material authoring. `finite_element_study_add_material` / update can now write thermal, electromagnetic, acoustic, and fluid material/media property groups without requiring Young's modulus or Poisson ratio.
- Kept structural material fields working, but made them optional and rejected only truly empty material definitions.
- Expanded model-visible material tool schemas with family-specific fields such as thermal conductivity, specific heat, permittivity, conductivity, acoustic density/speed of sound, fluid density, and dynamic viscosity.
- Updated model-facing metadata so the agent is explicitly guided to provide properties for the active physics family instead of defaulting to structural elastic constants.

Tests/evidence:

- `cargo test -p runmat-runtime material_operation` passed from `runmat-analysis` with the new non-structural material/media and empty-definition tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context material_tool_metadata_is_physics_family_agnostic -- --nocapture` passed from `../runmat-private`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final composed-flow audit. The material/media typed edit surface is now full-family capable, but the end-to-end guided acceptance matrix still needs final proof across representative solve/result paths.

### 2026-07-11: Generic Runtime Check Dispatch Slice 1

Scope completed:

- Added first-class `script` check result shaping beside the finite element check result shaping in the desktop agent runtime bridge.
- Broadened the generic agent `check` command in desktop runtime dispatch so `.fea` studies still use finite element validation while `.m` scripts read workspace bytes and run the existing RunMat static validator.
- Kept unsupported path/kind combinations explicit, so the model sees a clear error instead of a hidden FEA-only fallback.
- Added regression coverage proving `.m` checks do not call the finite element checker and return a `checker_kind: "script"` payload with `read`, `decode`, and `static_validation` phases.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 63 runtime-provider tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final completion audit across the composed desktop/harness/runtime path. The generic check surface now covers FEA studies and RunMat scripts; notebooks or other future artifact types should plug into the same dispatcher rather than getting separate FEA-specific tools.

### 2026-07-11: Geometry Session Lifecycle Guidance Slice 1

Scope completed:

- Audited Phase 4/9 geometry session tools and confirmed `geometry_clear_selection` and `geometry_close_session` exist in the tool registry and metadata.
- Updated model-facing FEA tool guidance so the model sees the complete geometry session lifecycle: open, render, adjust state, select, clear stale selection, create regions, and close sessions.
- Extended the context guidance regression so `geometry_clear_selection` and `geometry_close_session` are explicitly present when FEA/geometry tools are exposed, while `geometry_inspect` and `geometry_view` remain absent.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context tools -- --nocapture` passed from `../runmat-private` with 14 focused context/tool tests.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final requirement audit. The normal model-visible catalog now has the complete geometry session lifecycle documented; lower-level runtime names such as `runtime_geometry_inspect` remain host/runtime adapter internals rather than model-facing tools.

### 2026-07-11: Generated Physics Catalog Drift Guard Slice 1

Scope completed:

- Extended the FEA TypeScript contract generator so it emits `FEA_ANALYSIS_PROFILES` and `FEA_ANALYSIS_RUN_KINDS` arrays directly from the Rust-owned `AnalysisCreateModelProfile` and `AnalysisRunKind` enums.
- Re-exported those generated arrays from the public TS package root.
- Added package-level contract coverage proving the richer public `FEA_SUPPORTED_PHYSICS_PROFILES` catalog has exact profile coverage against the generated Rust-supported profile set, without forcing UI display order to match enum order.
- This keeps the full-family desktop/agent catalog from silently drifting when Rust adds, removes, or renames a supported analysis profile.

Tests/evidence:

- `npm run generate:fea-contracts` passed from `bindings/ts` and regenerated `src/generated/fea-study-document-contracts.ts`.
- `npm test -- src/index.spec.ts` passed from `bindings/ts` with 47 tests.
- `npm run lint` passed from `bindings/ts`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final requirement-by-requirement audit. The profile catalog metadata itself still carries human labels/default outputs in the public TS package, but the supported profile id set is now generated from Rust and guarded against drift.

### 2026-07-11: Full-Family Physics-Agnostic Defaults Slice 1

Scope completed:

- Removed structural fixed/load wording from the unknown-profile guided workflow fallback. Unknown physics now uses neutral boundary/driving labels instead of inheriting linear static structural copy.
- Removed the FEA study review pane's fallback from unsupported/no-profile studies to the first catalog profile's default outputs, which had been linear static structural displacement/von Mises stress.
- Strengthened desktop agent context coverage so the current supported family set is explicitly represented: structural, modal, thermal, coupled physics, electromagnetic, acoustic, and CFD.
- Added review-model regression coverage proving thermal defaults stay thermal and unknown profiles do not show structural displacement/stress defaults.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/fea/fea-study-review-model.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 31 focused tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `runmat-analysis` and `runmat-private`.

Remaining:

- Continue the final requirement-by-requirement audit and composed guided-flow acceptance consolidation. Remaining structural/stress references are expected only in explicit structural representative tests, coupled profiles that legitimately include structural fields, or negative assertions.

### 2026-07-11: Local Agent Runtime FEA Field Defaults Slice 1

Scope completed:

- Removed structural displacement/stress placeholders from the `runmat-agent` local environment FEA field selector/materialization tests.
- Local generic FEA variable summary coverage now uses `thermal.temperature`; bounded field preview coverage uses `acoustic.pressure`.
- This aligns the CLI/local combined agent/runtime path with the desktop and harness rule that generic FEA field handling is physics-family agnostic unless a test is explicitly structural.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-cli local_env -- --nocapture` passed from `../runmat-private` with 4 focused local-env tests after a full ah-cli dependency rebuild.
- `rg -n "displacement|stress|structural\\.displacement|field:run-.*stress" ../runmat-private/agent-harness/crates/ah-cli/src/local_env.rs` returned no matches.

Remaining:

- Continue the final clean-break and completion audit. Remaining structural/stress references in the harness are now concentrated in explicit structural representative coverage or negative assertions.

### 2026-07-11: Desktop Generic FEA Mock Defaults Slice 1

Scope completed:

- Removed structural displacement/stress as the default generic FEA field identity from desktop runtime mocks, replay hydration fixtures, run-history fixtures, run persistence fixtures, and the generic `.fea` execute bridge test.
- Default mock/materialization field identity is now `thermal.temperature`, with scalar field metadata, so generic runtime/replay paths do not silently teach structural mechanics as the default FEA mental model.
- Preserved explicit structural runtime-provider coverage where the test is intentionally proving the structural representative flow.

Tests/evidence:

- `npm test -- src/runtime/testing/fea-field-descriptor.ts src/replay/domain/fea-replay-artifacts.spec.ts src/replay/domain/replay-session-factory.spec.ts src/run/run-history.spec.ts src/run/fea-run-orchestrator.spec.ts src/run/fea-run-persistence.spec.ts src/runtime/lanes/runtime-lane-manager.spec.ts src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 81 tests across the executable focused suites.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue the final clean-break audit for any remaining structural defaults outside intentionally structural tests. The model-visible old FEA/geometry tool names currently remain limited to negative assertions and lower-level runtime internals.

### 2026-07-11: Context Domain Provider Boundary Slice 1

Scope completed:

- Moved FEA current-state attachment composition out of generic dynamic attachment assembly and behind a domain context provider boundary.
- Added `DomainContextSnapshots` and a provider composition point so generic context assembly no longer owns the FEA-specific attachment decision.
- Preserved the focused `sections::fea` attachment builder as the home for FEA JSON/model-block shaping, while `attachments.rs` now composes domain-provided attachments.
- Added provider-level regression coverage using a thermal FEA snapshot and kept existing FEA context/projection/cache coverage green.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context -- --nocapture` passed from `../runmat-private` with 47 tests.

Remaining:

- This closes the most concrete Phase 6 boundary smell. The completion audit still needs to prove the whole plan requirement-by-requirement, especially composed guided-flow evidence across desktop/browser/harness rather than only context architecture.

### 2026-07-11: Full-Family Guided Pane Labels Slice 1

Scope completed:

- Strengthened the guided FEA status panel coverage so the visible user-facing workflow labels are proven across modal, thermal, electromagnetic, acoustic, CFD/flow, and coupled profiles.
- The new regression verifies non-structural families render their own boundary/driver vocabulary instead of falling back to structural fixed/loaded area labels.
- Kept the change at the UI acceptance boundary; no new source of truth was added and the panel still formats labels from the existing physics workflow copy.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 5 focused panel tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue the requirement-by-requirement completion audit. The main remaining proof burden is now whether the current composed browser/desktop/harness evidence is sufficient to mark the whole guided flow complete, not whether the pane vocabulary is still linear-stress-biased.

### 2026-07-10: Cache-Aware Physics Context Layout Slice 1

Scope completed:

- Added projection coverage proving live FEA physics state does not enter the cacheable stable developer prefix.
- The regression mutates active physics profile, physics family, selected region, result figure id, field/result summary, and rendered image bytes across thermal and flow examples while keeping the cacheable section IDs and rendered stable prefix byte-stable.
- Verified the model-visible live state still appears after the current user request inside `current_turn_state`, with structured image blocks attached to the dynamic user message rather than serialized into stable text.
- Used non-structural profiles (`thermal_standalone` and `cfd_steady_state`) so this acceptance path does not silently collapse back to linear stress assumptions.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context projection -- --nocapture` passed from `../runmat-private` with 6 focused projection tests.

Remaining:

- Continue the final requirement-by-requirement audit and one representative composed guided-flow proof. This slice closes the explicit cache/layout acceptance gap but does not by itself mark the whole change plan complete.

### 2026-07-10: Guided FEA Pane Artifact Timeline Slice 1

Scope completed:

- Added an artifact timeline to the guided FEA status pane, covering the active study, referenced geometry, current graphical selection/render, selected run, mesh artifacts/evidence, field count/requested outputs, and result figures.
- Built the timeline from the existing `FeaTurnContextSnapshot`, `FeaVisualStateSnapshot`, and `FeaSelectedRunSnapshot` rather than introducing a parallel UI-only source of truth.
- Kept the pane compact and workflow-oriented: the timeline is an unframed ordered list under the existing workflow/status sections, not another nested card system.
- Added test coverage proving selected-run result views surface timeline entries for study/run/mesh/fields/result figures.
- Freed generated build/test caches after the first verification rerun hit `ENOSPC`; deleted `target`, `../runmat-private/agent-harness/target`, `../runmat-private/desktop/.next`, and `../runmat-private/desktop/test-results`. This was generated cache cleanup only and left source changes unstaged.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 4 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `df -h .` showed free space recovered from 181 MiB to 95 GiB after deleting generated caches.

Remaining:

- Continue final audit across the cache-aware context-layout acceptance and composed full guided-flow acceptance. This slice closes the explicit Phase 7 artifact timeline gap but does not yet prove the whole pane flow end to end across every representative physics class.

### 2026-07-10: Generic Harness Adapter Non-Structural Result Fixtures Slice 1

Scope completed:

- Removed structural displacement/stress as the default FEA result identity from generic agent-harness runtime adapter tests.
- Updated the desktop runtime adapter field-selector preservation test to use `acoustic.pressure`, proving the generic `variables`/`variable` bridge preserves finite-element field selectors without relying on structural mechanics fields.
- Updated the direct runtime adapter execute-file fixture to return `thermal.temperature` and `temperature_view`, proving generic FEA execute-file delegation without stress-specific result identity.
- Updated the `ah-tools` generic execute-file and `set_outputs` smoke fixtures to use `thermal.temperature` instead of `structural.displacement`.
- Re-ran a targeted search confirming the remaining old load-condition names in these harness modules are negative clean-break assertions only.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-runtime-adapter-desktop variables_and_materialize_var_preserve_fea_field_selectors -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-runtime-adapter-direct delegates_unified_fea_and_geometry_runtime_surface -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime -- --nocapture` passed from `../runmat-private` with 3 focused runtime-tool tests.

Remaining:

- Continue the final phase-by-phase audit, especially the guided React pane and cache-aware context-layout acceptance. This slice removes another structural default from generic harness proof paths but does not claim full plan completion.

### 2026-07-10: Full Physics-Family Driving Condition Contract Slice 1

Scope completed:

- Broadened the Rust-owned `.fea` study document operation so non-structural driving/source values are preserved instead of being dropped by the typed authoring path.
- Added persisted optional scalar fields for flow, acoustic, electromagnetic, thermal, and coupled source definitions: pressure, velocity, mass flow, volumetric flow, temperature, heat flux, voltage, power, and frequency, while keeping compact summary values for agent context.
- Broadened the model-visible `finite_element_study_add_driving_condition` and `finite_element_study_update_driving_condition` schemas beyond structural force/pressure to include thermal sources, EM sources, acoustic sources, flow drivers, interface transfer, and coupled sources.
- Updated tool metadata so the model sees driving conditions as active-physics-family source/driver inputs, not as structural loads.
- Updated harness capability fixtures so FEA capabilities advertise the full supported public physics profile set rather than `linear_static_structural` only.
- Added regression coverage for an acoustic source payload with pressure/frequency through both the Rust `.fea` authoring path and the model-visible tool schema.

Tests/evidence:

- `cargo test -p runmat-runtime fea_document_authoring -- --nocapture` passed from `runmat-analysis` with 6 focused authoring tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context tools -- --nocapture` passed from `../runmat-private` with 14 focused metadata/schema tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_acoustic_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_cfd_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed from `../runmat-private`.

Remaining:

- Continue final full-plan audit. This slice closes a concrete structural-bias gap in the model-visible driving-condition contract, but the remaining completion bar is still the whole guided pane flow across the catalog, not only source authoring.

### 2026-07-10: Phase 9 Raw Topology Retirement Audit Slice 1

Scope completed:

- Audited the model-visible geometry/FEA tool boundary for retired low-density tools.
- Verified `geometry_inspect`, `geometry_view`, `fea_check_study`, and `fea_run_study` are absent from agent-harness model-visible tool catalogs and desktop agent bridge/runtime-provider implementation paths, with remaining hits limited to negative assertions and lower-level runtime inspect APIs.
- Verified the session geometry tools include the full target session lifecycle, including `geometry_clear_selection` and `geometry_close_session`.
- Re-ran the existing raw-topology regression proving geometry session outputs compact large STEP-like payloads and exclude topology, evaluator, raw byte, and oversized array markers.
- Re-ran context-provider regressions proving FEA current-turn context excludes raw topology/evaluator markers and tool guidance names the bounded geometry session tools instead of retired raw inspection tools.

Tests/evidence:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools geometry_session_tools_do_not_expose_raw_topology_payloads -- --nocapture` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context geometry -- --nocapture` passed from `../runmat-private` with 2 focused tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context fea_context -- --nocapture` passed from `../runmat-private` with 2 focused tests.
- `rg -n "geometry_inspect|geometry_view|fea_check_study|fea_run_study" ../runmat-private/agent-harness ../runmat-private/desktop/src/agent ../runmat-private/desktop/src/runtime/domain ../runmat-private/desktop/src/runtime/runtime-provider.tsx ../runmat-private/desktop/src/runtime/clients/browser ../runmat-private/desktop/src/runtime/clients/tauri/index.ts -g '*.rs' -g '*.ts' -g '*.tsx'` returned only negative assertions plus the lower-level `runtime_geometry_inspect` Tauri runtime API.

Remaining:

- Treat Phase 9 as substantially proven for model-visible agent context. Continue the final audit across all phases before marking the whole plan complete.

### 2026-07-10: Desktop Generic FEA Bridge Non-Structural Identity Slice 1

Scope completed:

- Hardened the desktop generic runtime bridge tests so FEA field summaries, field materialization, and execute-file changed-variable results no longer use structural `stress`/`displacement` as their default proof cases.
- Updated `agent-runtime-bridge` domain coverage to use acoustic and thermal field identities for lazy paged variables, materialization requests, preview payloads, and `.fea` execution results.
- Updated `runtime-provider` generic bridge coverage so live selected-run fields use `thermal.temperature` and replayed persisted fields use `acoustic.pressure`, including namespaced variable ids and bounded page materialization.
- Preserved the generic `variables`, `variable`, `figures`, and `show_figures` surfaces; no FEA-specific result tools were added.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 13 tests.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "exposes selected finite element fields|inspects replayed finite element figures" --reporter=dot` passed from `../runmat-private/desktop` with 2 focused tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue toward final composed user-flow acceptance and the requirement-by-requirement audit; this slice proves the desktop bridge generic result path with non-structural field identities but does not by itself prove the whole guided pane flow end to end.

### 2026-07-10: Profile-Aware Guided Workflow Routing Slice 1

Scope completed:

- Split physics-family workflow copy and setup requirements out of the large FEA agent context builder into `fea-physics-workflow.ts`.
- Routed incomplete study setup through profile-aware requirements instead of structural count fallbacks:
  - modal studies do not require boundary/driving-condition gates unless the study/runtime says otherwise;
  - structural, thermal, electromagnetic, acoustic, CFD, and coupled profiles route through their own boundary and driver/source gates;
  - unknown profile ids return to physics selection instead of silently using structural wording.
- Added catalog-wide guided-context regression coverage that iterates every public `FEA_SUPPORTED_PHYSICS_PROFILES` entry and verifies missing-boundary and missing-driving-condition workflow steps do not use structural copy for non-structural profiles.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 26 tests.
- `npm test -- src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/use-fea-agent-context-state.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 8 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue full composed acceptance through actual mesh/solve/postprocess flows; this slice proves guided setup routing across the catalog but does not by itself prove every profile can complete a solve/report loop.

### 2026-07-10: Physics Family Agnostic Wording Slice 1

Scope completed:

- Removed the remaining active direct-adapter test path that still exercised `update_load_condition`/`load_condition_id`; it now proves `update_driving_condition` with `driving_condition_id`.
- Updated solve-failure guidance so generic retry recommendations say boundary/driving-condition/output changes instead of treating “load” as the universal concept.
- Tightened the structural selected-region workflow copy so the model-facing action is a structural driver and driving-condition region, while preserving structural load wording only where it is truly structural-domain language.
- Updated target design and change-plan docs to describe material/boundary/driving-condition counts, driving-condition annotations, and driving-condition tools instead of generic load-condition tools.

Tests/evidence:

- `cargo test -p ah-runtime-adapter-direct delegates_unified_fea_and_geometry_runtime_surface -- --nocapture` passed from `../runmat-private/agent-harness`.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 25 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Keep full-catalog acceptance as the completion bar: structural load/stress paths are one supported profile family, not the generic FEA model.

### 2026-07-10: Browser Guided FEA Bridge Acceptance Slice 1

Scope completed:

- Strengthened the browser agent client worker-bridge coverage from a narrow render/constraint/check/execute path into a guided FEA bridge sequence.
- The browser client test now routes worker runtime bridge requests for:
  - `geometry_render`
  - `finite_element_study_operation` with `add_region`
  - `finite_element_study_operation` with `add_constraint`
  - `finite_element_study_operation` with `add_driving_condition`
  - `finite_element_study_operation` with `set_outputs`
  - `finite_element_study_operation` with `set_mesh`
  - `check`
  - `execute_file`
  - `select_run`
  - `variables`
  - `figures`
- Added assertions that the flow uses `driving_condition_id` and `add_driving_condition`, and does not fall back to `add_load_condition`.
- Proved that post-solve inspection still routes through generic selected-run, variables, and figures bridge operations.

Tests/evidence:

- `npm test -- src/agent/clients/browser/index.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 20 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final guided pane and mesh/solve/postprocess acceptance; this slice strengthens browser/worker bridge evidence but does not replace full composed user-flow coverage.

### 2026-07-10: Physics-Agnostic Driving Condition Contract Slice 1

Scope completed:

- Renamed the Rust-owned `.fea` document operation contract from `add/update/remove_load_condition` to `add/update/remove_driving_condition`.
- Renamed public operation inputs and summaries from `load_condition_id`, `load_conditions`, `FeaStudyLoadEntry`, and `loadType` to `driving_condition_id`, `driving_conditions`, `FeaStudyDrivingConditionEntry`, and `type`.
- Preserved the durable YAML `loads:` section as a file-format detail while exposing `driving_conditions` in operation diffs/results.
- Regenerated the public TypeScript FEA contracts and rebuilt `bindings/ts/dist` so private desktop typecheck consumes the new generated types.
- Removed the harness-side hidden translation from model-facing driving-condition tools back to load-condition runtime operations; the tools now dispatch `add/update/remove_driving_condition` directly.
- Updated desktop runtime bridge/view-model fixtures to consume `driving_conditions` and `type`.
- Tightened FEA run metadata so persisted dataset `analysisProfile` and `analysisRunKind` are validated against the generated supported profile/run-kind unions.
- Replaced stale design-plan references to load-condition tool names with driving-condition tool names.
- Added Rust readiness coverage proving the current public physics profile catalog uses profile-specific material, boundary/interface, and driving/source readiness gates instead of treating only structural studies as needing setup.

Tests/evidence:

- `cargo fmt --all --manifest-path Cargo.toml` passed from `runmat-analysis`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test -p runmat-runtime fea_document_authoring -- --nocapture` passed with 5 authoring tests.
- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts --reporter=dot` passed from `bindings/ts` with 46 tests.
- `cargo test -p ah-tools runtime -- --nocapture` passed from `../runmat-private/agent-harness` with 3 tests.
- `cargo test -p ah-harness agent_can_fork_open_select_and_edit_copied_fea_study -- --nocapture` passed from `../runmat-private/agent-harness`.
- `npm test -- src/runtime/runtime-provider.spec.tsx src/runtime/domain/agent-runtime-bridge.spec.ts src/app/components/fea/fea-study-review-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx src/app/components/agent/use-fea-agent-context-state.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 84 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test -p runmat-wasm --test fea_document_operation -- --list` reported 0 native tests because the file is `wasm32`-gated.

Remaining:

- Continue final guided-flow acceptance and audit remaining runtime/domain vocabulary that legitimately still uses solver/file-format “loads” versus model/user-facing “driving conditions”.

### 2026-07-10: Desktop Driving Condition Context Boundary Slice 1

Scope completed:

- Renamed the desktop FEA agent study overview field from `loadCount` to `drivingConditionCount`.
- Kept the low-level runtime summary mapping from `counts.load_conditions` contained inside `studyOverviewFromRuntimeDocumentResult`.
- Updated FEA welcome/status markdown so the setup section says “driving condition” instead of generic “load” or “load condition”.
- Added blocker normalization at the FEA agent context boundary so runtime diagnostics like “Missing load condition” become “Missing driving condition” before reaching model context or pane state.
- Updated the mesh-attention prompt to recommend “driving-condition target” changes instead of “load target” changes.
- Preserved explicitly structural copy where appropriate, such as “structural loads” and “structural load target”.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/use-fea-agent-context-state.spec.tsx src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 37 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue moving lower-level `.fea` document naming toward final terminology only where it crosses user/model boundaries; the durable YAML/runtime internals still use `loads`/`load_conditions`.

### 2026-07-10: Model-Facing Driving Condition Tool Rename Slice 1

Scope completed:

- Renamed the model-visible finite element driving-condition tools from load-condition names to physics-neutral driving-condition names:
  - `finite_element_study_add_driving_condition`
  - `finite_element_study_update_driving_condition`
  - `finite_element_study_remove_driving_condition`
- Updated the tool schemas to use `driving_condition_id` instead of `load_condition_id`.
- Kept the lower-level runtime document operation mapping to `add_load_condition`, `update_load_condition`, and `remove_load_condition` so the existing `.fea` `loads` section can continue to serialize through the current Rust document operation without exposing that wording to the model.
- Updated model tool metadata and FEA authoring guidance to describe driving conditions as forces, pressures, heat inputs, currents, sources, flow drivers, and excitations.
- Updated the desktop guided prompt for selected driving regions to call `finite_element_study_add_driving_condition`.
- Updated the composed fork/open/select/edit host-parity path so the agent uses the new public tool while the runtime still applies the existing document operation.
- Added clean-break assertions that the old `finite_element_study_*_load_condition` tool names are not registered as model-visible tools.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime -- --nocapture` passed with 3 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools -- --nocapture` passed with 19 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context tools -- --nocapture` passed with 14 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_fork_open_select_and_edit_copied_fea_study --test host_parity -- --nocapture` passed.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 25 tests.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue reducing remaining lower-level `.fea` document naming leaks where they reach user/model surfaces; this slice cleans the model-visible tool boundary but does not rename the durable YAML `loads` section or Rust internal document operations.

### 2026-07-10: Profile-Neutral Guided Workflow Step Contract Slice 1

Scope completed:

- Removed structural-biased model-facing workflow step IDs from the live desktop FEA agent context.
- Replaced `pick_fixed_areas` with `define_boundaries` for boundary-condition setup in `currentStep` and `completedSteps`.
- Replaced `pick_loaded_areas` with `define_driving_conditions` for driving-condition setup in `currentStep` and `completedSteps`.
- Preserved profile-specific user-facing labels and prompts through `physicsWorkflowCopy`, so structural studies can still say fixed/load while CFD, acoustic, electromagnetic, thermal, and coupled studies keep their family-specific language.
- Updated agent-harness FEA context fixture tests and composed host-parity guided-region request assertions so the model-frame contract uses the neutral step IDs.
- Verified the old structural step IDs no longer appear in live desktop agent or harness context Rust/TS sources.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 29 tests.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context fea_context -- --nocapture` passed with 2 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint --test host_parity -- --nocapture` passed.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue guided pane and typed study-operation acceptance across the full catalog; this slice fixes the model-facing workflow vocabulary but does not complete every setup path.

### 2026-07-10: Full Catalog Generic Runtime Surface Acceptance Slice 1

Scope completed:

- Expanded composed host-parity coverage from representative physics families to the full current public profile catalog shape.
- Added a reusable catalog-profile solve/postprocess model fixture so new profile acceptance cases do not require another copy-pasted model client per profile.
- Added transient structural acceptance using `transient.displacement`, `figure:run_fea_transient_structural:transient_displacement`, and `runtime-fea-transient-structural`.
- Added nonlinear structural acceptance using `nonlinear.plastic_strain`, `figure:run_fea_nonlinear_structural:plastic_strain`, and `runtime-fea-nonlinear-structural`.
- Added conjugate heat-transfer acceptance using `cht.interface_heat_flux`, `figure:run_fea_cht:interface_heat_flux`, and `runtime-fea-cht`.
- Added fluid-structure interaction acceptance using `fsi.coupling_residual`, `figure:run_fea_fsi:coupling_residual`, and `runtime-fea-fsi`.
- Wired all four modes through the same mock runtime execution, run selection, lazy variable, figure, and bounded materialization surfaces as the existing structural/modal/thermal/electromagnetic/acoustic/CFD/coupled paths.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness generic_runtime_surfaces --test host_parity -- --nocapture` passed with 14 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 22 tests.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue guided-pane and typed study-operation acceptance against the full catalog; this slice proves the generic runtime/result surface but not every user-facing setup path.

### 2026-07-10: Non-Structural Generic Report Acceptance Slice 1

Scope completed:

- Removed the remaining structural-only report acceptance fixture that treated bracket displacement and stress as the generic FEA report shape.
- Repointed the composed report harness flow to an acoustic replay run: `replay:/studies/resonator.fea:run_fea_acoustic`.
- Drove the same generic tool sequence for report generation: `select_run`, `variables`, `figures`, `show_figures`, `variable`, and `write`.
- Added a report-specific acoustic result identity with `acoustic.pressure`, `figure:run_fea_acoustic:pressure`, replay representation, and lazy-paged field materialization.
- Updated report assertions to reject `structural.displacement`, the old stress figure, embedded base64 image blobs, and inline artifact emission.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_write_fea_report_from_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 18 tests.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue full-catalog guided acceptance; this slice closes the report-path structural stress assumption but does not by itself prove every supported profile variant.

### 2026-07-10: Electro-Thermal Generic Runtime Surface Acceptance Slice 1

Scope completed:

- Added electro-thermal FEA solve/postprocess coverage to the composed harness after making `electro_thermal_coupled` a first-class profile.
- Extended the desktop runtime fixture with electro-thermal result identity: `electro_thermal.joule_heat`, `figure:run_fea_electro_thermal:joule_heat`, and `runtime-fea-electro-thermal`.
- Added an electro-thermal model client that drives the same generic tool sequence as other FEA families: `finite_element_study_set_mesh`, `check`, `execute`, `select_run`, `variables`, `figures`, `show_figures`, and `variable`.
- Reused the shared family-generic host-parity assertion so electro-thermal is proven through selected-run, lazy variable, figure, and bounded materialization surfaces without FEA-specific result tools.
- Verified the full host parity suite now covers structural, modal, electromagnetic, electro-thermal, thermal, thermo-mechanical coupled, CFD, acoustic, replay, report, fork/select, and geometry-region setup flows together.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_electro_thermal_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 18 tests.

Remaining:

- Continue full-family guided pane and report acceptance; electro-thermal now has both first-class profile/catalog coverage and composed generic runtime-surface parity.

### 2026-07-10: Electro-Thermal First-Class Profile Slice 1

Scope completed:

- Promoted electro-thermal from lower-level runtime/domain/result support into a first-class `electro_thermal_coupled` study profile.
- Wired the new profile through the Rust FEA contract enum, derived run kind, profile labels, `.fea` summary run-kind derivation, mesh refinement profile labeling, and refinement namespace applicability.
- Added electro-thermal model creation defaults that seed electrical material properties, an electro-thermal domain, a transient step, a grounded electrical boundary, and a current-driving condition.
- Regenerated TS FEA study document contracts so `FeaAnalysisProfile` includes `electro_thermal_coupled`.
- Added the profile to the public TS physics catalog as coupled physics with electro-thermal temperature, Joule heat, electric potential, and current-density default outputs.
- Updated the desktop guided coupled-physics starter copy to name electro-thermal alongside thermo-mechanical, CHT, and FSI.

Tests/evidence:

- `cargo fmt --all --manifest-path Cargo.toml` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime analysis_create_model_supports_electro_thermal_profile_template -- --nocapture` passed.
- `cargo test -p runmat-runtime --test operation_contracts analysis_create_model_contract_is_v1_and_maps_codes -- --nocapture` passed.
- `npm test -- src/index.spec.ts` passed from `runmat-analysis/bindings/ts` with 46 tests.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 25 tests.

Remaining:

- Continue full-family guided acceptance across solve/postprocess/report paths; electro-thermal composed generic runtime-surface parity has since been added by `Electro-Thermal Generic Runtime Surface Acceptance Slice 1`.

### 2026-07-10: Physics-Family Agnostic Agent Surface Slice 1

Scope completed:

- Audited the current supported study profile catalog against runtime evidence and confirmed the agent-facing surface should follow the public profile catalog: structural, modal, transient/nonlinear structural, thermal, thermo-mechanical coupled, electromagnetic, acoustic, CFD steady/transient, CHT, and FSI.
- At the time of this slice, electro-thermal existed lower in runtime/domain/result support but was not yet exposed as a first-class public study creation profile; this has since been addressed by `Electro-Thermal First-Class Profile Slice 1`.
- Removed remaining generic FEA pane wording that treated structural constraints/loads as the default setup vocabulary.
- Replaced generic user-facing setup language with materials, boundaries, sources, driving conditions, interfaces, outputs, and active-physics wording.
- Tightened the model-facing boundary-condition tool schema so examples start from the active physics family and include thermal, CFD, acoustic, electromagnetic, and structural examples instead of leading with fixed/displacement.
- Preserved explicitly structural wording only in the structural starter option, where constraints/loads are the correct physics-specific terms.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools -- --nocapture` passed with 19 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 35 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final audit for report generation and any runtime/replay paths that may still use structural outputs as generic assumptions.

### 2026-07-10: Physics-Aware Runtime Plot Default Slice 1

Scope completed:

- Audited runtime FEA plotting and found `fea.plot` default figure selection still preferred structural von Mises/stress fields globally when no field id was supplied.
- Replaced the structural-only preferred-field list with a deterministic scoring function that covers structural, modal, thermal, transient, nonlinear, electromagnetic, electro-thermal, thermo-mechanical, acoustic, CFD, CHT, FSI, and fluid field prefixes.
- Preserved structural behavior by keeping von Mises/stress ahead of displacement/residual fields when the available result set is structural.
- Ranked residual, iteration, orthogonality, and condition fields below primary result fields so default plots favor useful result views across physics families.
- Added runtime regression coverage for thermal, CFD, acoustic, and coupled default plot selection.

Tests/evidence:

- `cargo fmt --all --manifest-path Cargo.toml` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime fea_plot_default -- --nocapture` passed with the two filtered runtime plot-default tests.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final audit for report generation and any remaining runtime/replay code paths that may still name stress/displacement as generic result assumptions.

### 2026-07-10: Physics-Aware Result Pane Identity Slice 1

Scope completed:

- Audited result/replay/runtime surfaces for structural-only defaults and found the FEA results pane was not surfacing the run's persisted analysis identity or output summary.
- Extended the FEA results view model to derive `analysisProfile`, `physicsFamily`, `analysisRunKind`, and `availableOutputSummary` through the shared run metadata helper used by agent context/persistence.
- Added physics/profile/run-kind/output summary rows to the FEA results pane so live and replayed result surfaces are visibly tied to the selected profile/output set, not just a generic FEA run.
- Taught the shared output-summary helper to recognize normalized camel-case `fieldId` descriptors as well as persisted `field_id` descriptors.
- Added non-structural thermal result-pane regression coverage proving the pane renders profile/output identity without materializing field values or depending on structural fields.

Tests/evidence:

- `npm test -- src/runtime/domain/fea-results-view-model.spec.ts src/app/components/fea/fea-results-pane.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 5 tests.
- `npm test -- src/run/fea-run-persistence.spec.ts src/app/components/agent/fea-agent-state.spec.ts src/runtime/domain/fea-results-view-model.spec.ts src/app/components/fea/fea-results-pane.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 12 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final audit for report generation and runtime/replay code paths that may still name stress/displacement as generic result assumptions.

### 2026-07-10: Coupled Guided Workflow Copy Slice 1

Scope completed:

- Audited the guided FEA workflow copy after broadening runtime-surface parity and found thermo-mechanical coupled studies could fall through to structural fixed/load wording.
- Added an explicit coupled-physics copy branch for `coupled`, `thermo_mechanical`, `electro_thermal`, `fsi`, and `cht` profiles before single-domain structural/thermal/flow copy branches.
- Updated coupled guided setup wording to use boundary/interface regions and coupled driving conditions instead of fixed areas or load-condition targets.
- Removed unreachable `cht`/`fsi` checks from the CFD branch now that those profiles intentionally use coupled copy.
- Added regression coverage for thermo-mechanical coupled selected boundary and selected driver workflows.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 25 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 37 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final audit for report, replay, result rendering, and runtime paths that may still use structural/stress/displacement as generic defaults.

### 2026-07-10: CFD And Acoustic Generic Runtime Surface Acceptance Slice 1

Scope completed:

- Added CFD and acoustic FEA solve/postprocess harness paths to complete representative composed coverage for the currently documented physics-family set.
- Extended the desktop runtime fixture with CFD result identity: `fluid.velocity`, `figure:run_fea_cfd:velocity`, and `runtime-fea-cfd`.
- Extended the desktop runtime fixture with acoustic result identity: `acoustic.pressure`, `figure:run_fea_acoustic:pressure`, and `runtime-fea-acoustic`.
- Added CFD and acoustic model clients that drive the same generic tool sequence: `finite_element_study_set_mesh`, `check`, `execute`, `select_run`, `variables`, `figures`, `show_figures`, and `variable`.
- Reused the shared family-generic host-parity assertion so CFD/acoustic are proven through the same selected-run, lazy variable, figure, and bounded materialization surfaces as structural, modal, electromagnetic, thermal, and coupled studies.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_cfd_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_acoustic_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 17 tests.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final audit for guided pane, report, replay, context, and runtime paths that may still use structural/stress/displacement as generic fallbacks despite the now-broad composed runtime-surface test matrix.

### 2026-07-10: Thermal And Coupled Generic Runtime Surface Acceptance Slice 1

Scope completed:

- Added thermal and thermo-mechanical coupled FEA solve/postprocess harness paths to broaden Phase 8 acceptance beyond structural/modal/electromagnetic examples.
- Extended the desktop runtime fixture with thermal result identity: `thermal.temperature`, `figure:run_fea_thermal:temperature`, and `runtime-fea-thermal`.
- Extended the desktop runtime fixture with coupled result identity: `thermo_mechanical.coupling_residual`, `figure:run_fea_coupled:coupling_residual`, and `runtime-fea-coupled`.
- Added thermal and coupled model clients that drive the same generic tool sequence: `finite_element_study_set_mesh`, `check`, `execute`, `select_run`, `variables`, `figures`, `show_figures`, and `variable`.
- Added shared host-parity assertions proving thermal and coupled fields/figures flow through generic runtime inspection with lazy materialization and without leaking unrelated structural/modal/electromagnetic fields.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_thermal_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_coupled_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 15 tests.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue final audit for any guided pane, report, replay, or context code still using structural/stress/displacement as a generic fallback.

### 2026-07-10: Physics-Family Scope Audit Slice 1

Scope completed:

- Audited the current supported profile catalog and confirmed the target must cover structural static, modal, transient structural, nonlinear structural, thermal, electromagnetic, acoustic, CFD, and coupled physics profiles.
- Updated the target design to make profile-catalog-driven behavior the explicit contract across the agent, guided pane, runtime, selected run context, variables, figures, reports, and persisted artifacts.
- Updated the change plan so Phase 3 and Phase 8 acceptance cannot pass on structural stress/displacement assumptions alone.
- Clarified that boundary/driving-condition language, readiness gates, result fields, figures, and reports must come from the selected profile and requested outputs.
- Added context-layout requirements for `analysis_profile`, `physics_family`, run kind, output summaries, field descriptors, and figure refs as first-class bounded model-frame data.

Tests/evidence:

- Documentation/design update only; no code tests run for this slice.
- Verified existing code evidence includes TS-supported profile catalog, runtime/analysis field IDs for thermal, modal, acoustic, CFD, CHT, FSI, transient, nonlinear, thermo-mechanical, electro-thermal, and electromagnetic families, plus existing composed host parity for structural, modal, and electromagnetic generic runtime surfaces.

Remaining:

- Continue final audit for any guided pane, report, replay, or context code still using structural/stress/displacement as a generic fallback.

### 2026-07-10: Electromagnetic Generic Runtime Surface Acceptance Slice 1

Scope completed:

- Extended composed harness FEA acceptance beyond mechanics-family result shapes by adding an electromagnetic solve/postprocess path.
- Added electromagnetic result identity to the desktop runtime fixture: `em.magnetic_flux_density`, `figure:run_fea_em:magnetic_flux_density`, and `runtime-fea-em`.
- Added an electromagnetic FEA solve/postprocess model that drives the same generic tools as structural and modal flows: `finite_element_study_set_mesh`, `check`, `execute`, `select_run`, `variables`, `figures`, `show_figures`, and `variable`.
- Added host-parity assertions proving EM fields/figures flow through generic runtime inspection with lazy field materialization and without structural/modal field leakage.
- Re-ran full host parity to cover the structural, modal, electromagnetic, replay, report, fork/select, and geometry-region setup flows together.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_electromagnetic_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 13 tests.

Remaining:

- Continue Phase 8/final audit by verifying whether current composed evidence is sufficient for supported physics families or whether CFD/thermal/coupled need similar generic-runtime parity coverage; continue raw-topology/context and legacy bridge retirement audit.

### 2026-07-10: Modal Generic Runtime Surface Acceptance Slice 1

Scope completed:

- Audited composed harness FEA solve/postprocess coverage and found the existing generic runtime-surface acceptance was still structural-field/stress-figure specific.
- Extended the harness desktop transport fixture so finite element result identity can vary by physics/run mode instead of hardcoding `structural.displacement` and stress figures everywhere.
- Added a modal FEA solve/postprocess model that drives the same generic tools as the structural flow: `finite_element_study_set_mesh`, `check`, `execute`, `select_run`, `variables`, `figures`, `show_figures`, and `variable`.
- Added composed host-parity coverage proving modal result fields (`structural.mode_shapes`) and modal figures (`mode_shape_1`) flow through the same generic runtime surfaces with lazy field materialization and without inline artifact blobs.
- Re-ran the existing structural solve/postprocess parity path to prove the shared fixture refactor did not regress it.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_modal_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity -- --nocapture` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity -- --nocapture` passed with 12 tests.

Remaining:

- Continue Phase 8 by expanding composed generic-runtime acceptance to another non-structural family if audit evidence remains narrow, then continue final audit for raw-topology/context and legacy bridge retirement.

### 2026-07-10: Active Study Path Matching Slice 1

Scope completed:

- Tightened FEA context path matching so active-study selected runs and visual snapshots survive when one side is workspace-relative and the other side is absolute.
- Kept mismatch filtering strict by accepting only exact matches or directory-bound suffix matches, not basename-only matches.
- Applied the same normalized artifact-path matching to selected-run `.fea` paths and visual source paths.
- Added regression coverage proving an absolute selected-run `docPath` and absolute geometry `sourcePath` still drive the active relative study context when they refer to the same artifacts.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 24 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 36 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by proving the guided mesh/solve/postprocess flow over composed runtime surfaces across the supported physics set, then continue final audit for raw-topology/context and legacy bridge retirement.

### 2026-07-10: Active Study Visual-State Synchronization Slice 1

Scope completed:

- Added FEA context-boundary filtering for visual snapshots so graphical state must match the active geometry path or active study path before entering the model-visible FEA context.
- Removed stale result-render state when no active-study selected run is available, preventing visible overlays from unrelated runs from driving current study actions.
- Reused normalized artifact-path comparison at the context boundary instead of adding ad hoc checks in the pane or workflow rendering.
- Strengthened the mismatched-run regression so a completed `other.fea` run with an `other.step` result render is fully removed from `bracket.fea` context.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 23 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 35 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by proving the guided mesh/solve/postprocess flow over composed runtime surfaces across the supported physics set, then continue final audit for raw-topology/context and legacy bridge retirement.

### 2026-07-10: Active Study Selected-Run Synchronization Slice 1

Scope completed:

- Filtered selected FEA runs at the FEA context boundary so only runs whose `.fea` `docPath` matches the active study path can drive guided workflow state or enter the FEA turn snapshot.
- Prevented a completed, running, failed, or visible-result run from a different study from pushing the current study into result, mesh, solve, or visible-overlay states.
- Stripped stale result-render state from the FEA visual snapshot when the visible result belongs to a non-active selected run, while preserving ordinary geometry render state.
- Simplified downstream selected-run state checks so they operate on already-filtered active-study runs.
- Added regression coverage proving a completed `other.fea` run does not mark `bracket.fea` solved and does not appear in `context.selectedRun`.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 23 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 35 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by proving the guided mesh/solve/postprocess flow over composed runtime surfaces across the supported physics set, then continue final audit for raw-topology/context and legacy bridge retirement.

### 2026-07-10: Physics-Aware Guided Copy Slice 1

Scope completed:

- Centralized guided workflow copy for boundary/driving-condition steps in `fea-agent-context.ts`, keyed from the active study profile/run kind.
- Kept stable workflow step ids intact while making model-facing prompts and pane labels physics-aware for structural, modal, thermal, electromagnetic, acoustic, CFD, CHT, and FSI-style studies.
- Replaced non-structural "fixed area" / "load area" wording with appropriate boundary/source/driver language in guided prompts and selected-region actions.
- Exported the shared workflow step-label formatter to the status panel so visible pane headings use the same physics-aware copy as model prompts.
- Preserved structural-specific load wording for structural studies instead of flattening all physics into generic "values and units" text.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 26 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 34 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by proving the guided mesh/solve/postprocess flow over composed runtime surfaces across the supported physics set, then continue final audit for raw-topology/context and legacy bridge retirement.

### 2026-07-10: Physics-Agnostic Guided Readiness Slice 1

Scope completed:

- Fixed the guided FEA workflow step ordering so authoritative Rust-provided `readyToSolve` and selected-run terminal/active states win before fallback structural-style setup counts.
- Prevented ready non-structural studies from being pushed back into `pick_fixed_areas`, `pick_loaded_areas`, or `choose_outputs` solely because their valid physics setup does not look like a linear structural load/constraint workflow.
- Preserved the count-based prompts as fallback guidance for incomplete studies where readiness is not yet satisfied.
- Added regression coverage for a ready CFD study with no structural-style load/constraint/output counts and for an electromagnetic failed-mesh run that must enter mesh attention before setup-count fallbacks.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 20 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 31 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by proving the guided mesh/solve/postprocess flow over composed runtime surfaces across the supported physics set, then continue final audit for raw-topology/context and legacy bridge retirement.

### 2026-07-10: Physics-Agnostic Run Dataset Metadata Slice 1

Scope completed:

- Extended the runtime finite element run dataset contract with optional explicit analysis identity fields: `analysis_profile`, `physics_family`, `analysis_run_kind`, and `available_output_summary`.
- Regenerated the RunMat TypeScript FEA contracts and exported typed `FeaAnalysisProfile` / `FeaAnalysisRunKind` unions from the public TS package.
- Moved desktop selected-run physics/output derivation out of the agent component and into shared run-domain metadata helpers.
- Updated FEA run artifact persistence so every new dataset artifact records bounded physics identity and available output summary without materializing large fields.
- Kept replay compatibility by making the new dataset metadata optional and teaching selected-run context to use persisted dataset identity when live run metadata is unavailable.
- Added non-linear-static coverage with electromagnetic persistence metadata and CFD replay metadata so the path is not coupled to stress/displacement or linear structural assumptions.

Tests/evidence:

- `npm run generate:fea-contracts` passed from `bindings/ts`.
- `cargo fmt --all --manifest-path Cargo.toml` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime analysis_run_study_executes_linear_static_path -- --nocapture` passed from `runmat-analysis`.
- `npm run lint` passed from `bindings/ts`.
- `npm test -- src/run/fea-run-persistence.spec.ts src/app/components/agent/fea-agent-state.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 7 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue Phase 8 by driving the guided pane and postprocess/report acceptance across the full supported physics profile set, using the explicit run dataset identity as the replay/current-run source instead of structural-only assumptions.

### 2026-07-10: Selected Run Physics Identity Slice 1

Scope completed:

- Added `model_profile` to runtime `AnalysisStudyRunData` and persisted it into the finite element study-run evidence artifact.
- Regenerated the RunMat TypeScript FEA contracts.
- Extended the Rust-owned `FeaSelectedRunSnapshot` protocol with bounded selected-run analysis identity fields: `analysis_profile`, `physics_family`, `analysis_run_kind`, and `requested_output_summary`.
- Regenerated the desktop TypeScript protocol from `ah-protocol`.
- Mapped the new selected-run analysis identity fields through `ah-core` into `ah-context`, with model-frame and section-level assertions.
- Updated desktop selected-run snapshot construction to derive analysis profile/run-kind from run/result/dataset metadata, derive physics family from the supported profile catalog, and summarize requested/result outputs from bounded field descriptors.
- Updated result-inspection guidance and status-panel rendering so completed runs speak in terms of the selected run's actual physics family and output set instead of generic "active physics" text.

Tests/evidence:

- `npm run generate:fea-contracts` passed from `bindings/ts`.
- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path Cargo.toml` passed from `runmat-analysis`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test -p runmat-runtime analysis_run_study -- --nocapture` passed from `runmat-analysis` with 11 study-run tests.
- `cargo test -p runmat-runtime analysis_run_study_executes_linear_static_path -- --nocapture` passed from `runmat-analysis` with the new `model_profile` assertions.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 28 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm run lint` passed from `bindings/ts`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue Phase 8 by using the selected-run physics/output identity in broader postprocess/report acceptance and then continue GeometrySession/current-render tool coverage and raw-topology retirement.

### 2026-07-10: Mesh Artifact Provenance In Selected Run Context Slice 1

Scope completed:

- Extended the Rust-owned `FeaSelectedRunSnapshot` protocol with bounded mesh provenance refs: `mesh_id`, `mesh_artifact_ref`, `mesh_evidence_ref`, `refined_mesh_artifact_ref`, and `refined_mesh_evidence_ref`.
- Regenerated the desktop TypeScript protocol from `ah-protocol`.
- Mapped the new selected-run mesh provenance fields through `ah-core` into `ah-context`.
- Added model-frame and section-level assertions proving the refs appear in bounded JSON context while raw topology/evaluator data remains excluded.
- Updated desktop selected-run snapshot construction to extract mesh refs from direct run/result/dataset fields and artifact manifest entries without exposing raw artifact payloads.
- Updated mesh-attention next-action and prompt guidance so the agent can inspect bounded mesh evidence refs through generic run/artifact/check surfaces instead of guessing from diagnostics alone.
- Surfaced the bounded mesh refs in the FEA status panel selected-run card.

Tests/evidence:

- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 28 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed from both `runmat-analysis` and `../runmat-private`.

Remaining:

- Continue Phase 8 by proving mesh/solve/postprocess loops across the full supported physics family set, then continue GeometrySession/current-render tool coverage and raw-topology retirement.

### 2026-07-10: Physics-Family Authoring Coverage Slice 1

Scope completed:

- Removed the compact study authoring gate that rejected every non-linear-static structural profile.
- Kept the existing profile/run-kind compatibility check as the source of truth.
- Changed compact evidence authoring to reuse the existing profile-specific default model generation for all supported families, then bind the selected material, boundary, and load regions from mesh evidence without overwriting physics-specific boundary/load/step kinds.
- Preserved force-vector override behavior only for authored loads that are actually force loads.
- Added runtime coverage proving electromagnetic authoring uses the same mesh-evidence path while preserving vector-potential ground boundaries, coil-current loads, and electromagnetic steps.
- Updated target plan and user-experience notes so acceptance examples and guided first-run choices are framed around the active physics family and requested outputs instead of stress/displacement only.
- Fixed a runtime contract-test fixture that had not been updated for the Rust-owned `AnalysisStudySpec.outputs` field.

Tests/evidence:

- `cargo fmt --all --manifest-path Cargo.toml` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime analysis_author_study_preserves_non_structural_profile_defaults -- --nocapture` passed from `runmat-analysis`.

Remaining:

- Continue Phase 8 by connecting mesh attention to richer mesh-quality artifacts when available, then continue proving the guided pane across the supported structural, modal, thermal, flow, electromagnetic, acoustic, and coupled physics families.

### 2026-07-10: Mesh And Solve Issue Attention Slice 1

Scope completed:

- Extended the Rust-owned `FeaSelectedRunSnapshot` protocol with bounded `mesh_issue_summary` and `solve_issue_summary` fields.
- Regenerated the desktop TypeScript protocol from `ah-protocol`.
- Mapped mesh/solve issue summaries through `ah-core` into `ah-context`, with model-frame assertions proving the fields are present in bounded current-turn JSON context.
- Updated desktop selected-run snapshot construction to classify compact mesh and solve issue summaries from progress phase/message, FEA diagnostics, and FEA progress events without exposing raw diagnostic payloads.
- Tightened phase classification so explicit phases such as `mesh_prep` win over incidental message text such as “before solve.”
- Routed failed FEA runs with mesh issue summaries into `mesh_attention` before generic solve attention.
- Added a `Diagnose mesh` guided workflow action and made `solve_attention` prefer `solveIssueSummary` before generic diagnostics.

Tests/evidence:

- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 27 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- agent-harness/crates/ah-protocol/src/ops.rs agent-harness/crates/ah-context/src/request.rs agent-harness/crates/ah-core/src/engine.rs agent-harness/crates/ah-core/src/tests.rs agent-harness/crates/ah-context/src/projection.rs agent-harness/crates/ah-context/src/sections/attachments.rs agent-harness/crates/ah-context/src/sections/fea.rs desktop/src/agent/generated/protocol.ts desktop/src/app/components/agent/fea-agent-state.ts desktop/src/app/components/agent/fea-agent-state.spec.ts desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/app/components/agent/agent-turn-context.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by connecting mesh attention to richer mesh-quality artifacts when available, then resume GeometrySession/current-render tool coverage and raw-topology retirement.

### 2026-07-10: Selected Run Result Render State Slice 1

Scope completed:

- Extended the Rust-owned `FeaVisualStateSnapshot` protocol with `result_render_available` and bounded `visible_result_figure_ids`.
- Regenerated the desktop TypeScript protocol from `ah-protocol`.
- Mapped the new FEA visual result-render fields through `ah-core` into `ah-context`.
- Updated desktop FEA visual-state construction so generic figure monitor slots from the selected FEA run become explicit selected-run result-render state.
- Kept the result overlay loop on the generic runtime surface: visible result views are still driven by `figures` / `show_figures`, not a new FEA-specific result-display tool.
- Updated completed-run workflow guidance so a visible selected-run result view produces an `Inspect visible overlay` action instead of always prompting the model to show another result figure.
- Surfaced visible result figure ids in the FEA status panel.

Tests/evidence:

- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 22 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- agent-harness/crates/ah-protocol/src/ops.rs agent-harness/crates/ah-context/src/request.rs agent-harness/crates/ah-core/src/engine.rs agent-harness/crates/ah-core/src/tests.rs agent-harness/crates/ah-context/src/projection.rs agent-harness/crates/ah-context/src/sections/attachments.rs agent-harness/crates/ah-context/src/sections/fea.rs agent-harness/crates/ah-harness/tests/host_parity.rs desktop/src/agent/generated/protocol.ts desktop/src/app/components/agent/fea-agent-state.ts desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/app/components/agent/fea-agent-status-panel.tsx desktop/src/app/components/agent/fea-agent-status-panel.spec.tsx desktop/src/app/components/agent/agent-turn-context.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by tightening mesh/solve failure attention states and then move back to GeometrySession/current-render tool coverage and raw-topology retirement.

### 2026-07-10: Physics-Agnostic Selected Run Summaries Slice 1

Scope completed:

- Extended the Rust-owned `FeaSelectedRunSnapshot` protocol with bounded `mesh_summary` and `result_summary` fields, regenerated the desktop TypeScript protocol, and mapped the fields through `ah-core` into `ah-context`.
- Added model-frame assertions proving selected FEA run mesh/result summaries appear in bounded current-turn JSON context without exposing raw solver or result payloads.
- Updated desktop selected-run snapshot construction to derive compact mesh/result summaries from existing durable FEA run artifacts, `resultSummary`, run metadata, field descriptors, and figure counts.
- Surfaced selected-run mesh/result summaries in the FEA status panel.
- Made completed-run next actions use the bounded result summary when available.
- Broadened CAD-start workflow choices across the supported physics family set: structural, modal, thermal, CFD, electromagnetic, acoustic, and coupled physics.
- Removed structural-only result prompt wording from result-overlay and result-inspection actions; result guidance now points to the active physics and requested outputs.
- Broadened model-facing study tool metadata examples for constraints/boundaries and requested outputs across thermal, fluid, electromagnetic, acoustic, modal, coupled, and structural fields.

Tests/evidence:

- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-core --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 23 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- agent-harness/crates/ah-protocol/src/ops.rs agent-harness/crates/ah-context/src/request.rs agent-harness/crates/ah-core/src/engine.rs agent-harness/crates/ah-core/src/tests.rs agent-harness/crates/ah-context/src/projection.rs agent-harness/crates/ah-context/src/sections/attachments.rs agent-harness/crates/ah-context/src/sections/fea.rs agent-harness/crates/ah-context/src/tools/metadata.rs desktop/src/agent/generated/protocol.ts desktop/src/app/components/agent/fea-agent-state.ts desktop/src/app/components/agent/fea-agent-state.spec.ts desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/app/components/agent/fea-agent-status-panel.tsx desktop/src/app/components/agent/fea-agent-status-panel.spec.tsx desktop/src/app/components/agent/agent-turn-context.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by making mesh/solve/result state fully drive the graphical result overlay loop, then move back to GeometrySession/current-render tool coverage and raw-topology retirement.

### 2026-07-10: Selected FEA Run Diagnostics Snapshot Slice 1

Scope completed:

- Extended the Rust-owned `FeaSelectedRunSnapshot` protocol with compact run progress and diagnostic summary fields.
- Regenerated the desktop TypeScript protocol from `ah-protocol`.
- Mapped selected-run progress and diagnostic fields through `ah-core` into `ah-context`, with model-frame assertions proving they are visible in bounded JSON context.
- Updated desktop selected-run snapshot construction to summarize `ExecutionSession.progress`, `failureMessage`, and the first FEA diagnostic without exposing raw diagnostic payloads.
- Updated workflow next actions so running solves use progress text and failed solves use diagnostic summaries before falling back to generic guidance.
- Surfaced selected-run progress and diagnostic summary in the FEA status panel.

Tests/evidence:

- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 22 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- agent-harness/crates/ah-protocol/src/ops.rs agent-harness/crates/ah-context/src/request.rs agent-harness/crates/ah-core/src/engine.rs agent-harness/crates/ah-core/src/tests.rs agent-harness/crates/ah-context/src/projection.rs agent-harness/crates/ah-context/src/sections/attachments.rs agent-harness/crates/ah-context/src/sections/fea.rs desktop/src/agent/generated/protocol.ts desktop/src/app/components/agent/fea-agent-state.ts desktop/src/app/components/agent/fea-agent-state.spec.ts desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/app/components/agent/fea-agent-status-panel.tsx desktop/src/app/components/agent/agent-turn-context.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by tying these diagnostics to richer mesh-quality/solver-result summaries as those runtime artifacts become available, while preserving bounded model context.

### 2026-07-10: Result Overlay Guided Action Slice 1

Scope completed:

- Added a completed-run guided action for result overlays when the selected FEA run reports available figures.
- Kept result-overlay access on the generic runtime surface: the action tells the model to use `figures` and `show_figures`, and to avoid materializing full fields.
- Kept the action generated from `FeaSelectedRunSnapshot.figureCount`, so the UI and model context stay driven by selected run state rather than hardcoded markdown.
- Added desktop workflow coverage proving completed FEA runs expose the result-overlay action alongside inspect/report actions.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 18 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by connecting mesh/solve diagnostics and result-overlay display state more directly to the graphical surface once the runtime exposes richer mesh/solver diagnostic summaries.

### 2026-07-10: Typed FEA Workflow Attention State Slice 1

Scope completed:

- Added a Rust-owned `FeaWorkflowAttentionState` protocol enum and generated it into the desktop TypeScript protocol surface.
- Added `attention_state` / `attentionState` to `FeaWorkflowSnapshot` so guided mesh/solve/result state is structured data instead of prose embedded in `nextAction`.
- Mapped the new state through `ah-core` into `ah-context`, and asserted it appears in the model-visible FEA context JSON.
- Updated the desktop FEA workflow builder to emit explicit states for setup attention, ready-to-mesh, solving, solve attention, mesh attention, and results-ready flows.
- Surfaced the attention state in the FEA status panel and added desktop coverage for ready-to-mesh, running solve, failed solve, and completed-result transitions.
- Strengthened composed harness selected-region context assertions so the model frame must include `attention_state`.

Tests/evidence:

- `npm run gen:agent-harness-types` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/agent-turn-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 20 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- agent-harness/crates/ah-protocol/src/ops.rs agent-harness/crates/ah-protocol/src/lib.rs agent-harness/crates/ah-protocol/examples/export_ts.rs agent-harness/crates/ah-context/src/request.rs agent-harness/crates/ah-context/src/lib.rs agent-harness/crates/ah-context/src/projection.rs agent-harness/crates/ah-context/src/sections/attachments.rs agent-harness/crates/ah-context/src/sections/fea.rs agent-harness/crates/ah-core/src/engine.rs agent-harness/crates/ah-core/src/tests.rs agent-harness/crates/ah-harness/tests/host_parity.rs agent-harness/crates/ah-harness/tests/support/host.rs desktop/src/agent/generated/protocol.ts desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/app/components/agent/fea-agent-status-panel.tsx desktop/src/app/components/agent/fea-agent-status-panel.spec.tsx desktop/src/app/components/agent/agent-turn-context.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 8 by attaching concrete mesh/solve diagnostics and result-overlay controls to these typed attention states, rather than only showing the state label.

### 2026-07-10: Composed Guided FEA Workflow Context Slice 1

Scope completed:

- Strengthened the composed `ah-harness` selected-region FEA setup acceptance test so it submits a real FEA turn context with workflow, visual selection, readiness, open paths, and active study state.
- Added fixture-side request assertions proving the model frame contains the guided workflow snapshot, selected study/geometry paths, graphical selection selector/label, and current render availability before the model issues geometry and `.fea` mutation tools.
- Kept the composed action path unchanged after context projection: `geometry_open_session`, `geometry_select`, `geometry_create_region`, `finite_element_study_add_region`, `finite_element_study_add_constraint`, then generic `check`.
- Verified the broader host parity target still passes, so replay inspection, solve/postprocess, report writing, fork/edit, and selected-region setup continue to compose through the same harness/runtime surfaces.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint --test host_parity` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed with 11 tests.
- `git diff --check -- agent-harness/crates/ah-harness/tests/support/host.rs agent-harness/crates/ah-harness/tests/host_parity.rs` passed from `../runmat-private`.

Remaining:

- Continue closing Phase 8 by making the mesh/solve/postprocess guided state transitions richer than prompt text, especially solver/mesh failure attention states and result-overlay affordances.

### 2026-07-10: Workflow-Driven FEA Welcome Actions Slice 1

Scope completed:

- Made `fea-agent-context.ts` the single source for model/UI workflow choices in the FEA pane.
- Removed duplicated `runmat:prompt` shortcut generation from FEA welcome markdown; the markdown now stays descriptive, while actionable prompts come from `workflow.choices`.
- Added thermal and modal geometry-start workflow choices so the workflow snapshot preserves the useful startup actions that previously only existed in markdown copy.
- Let `FeaAgentStatusPanel` render expanded by default for the empty FEA welcome, so first-turn guided actions are visible before an agent thread exists.
- Wired the empty FEA welcome to render the same workflow action panel used by active FEA threads.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-state.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 18 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-state.ts desktop/src/app/components/agent/fea-agent-state.spec.ts desktop/src/app/components/agent/fea-agent-status-panel.tsx desktop/src/app/components/agent/fea-agent-status-panel.spec.tsx desktop/src/app/components/agent/agent-panel.tsx` passed from `../runmat-private`.

Remaining:

- Continue Phase 7/8 by driving these workflow choices into composed setup/edit/check/execute acceptance instead of only proving they render correctly.

### 2026-07-10: Composed Replay FEA Inspection Coverage Slice 1

Scope completed:

- Strengthened the composed `ah-harness` replayed FEA inspection test so it pins the generic runtime surface more tightly.
- The replay flow now asserts persisted result figures include finite-element result source, stable artifact figure id, preview artifact path, and scene artifact path.
- The replay flow now asserts FEA fields appear through generic `variables()` with stable field `variable_id`, `field_id`, page size/count, lazy paged materialization, and default materialization limit.
- Kept the model-visible sequence generic: `select_run`, `figures`, `show_figures`, `variables`, and `variable`, with no FEA-specific result tools.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-harness fea_replay_inspection_uses_generic_runtime_surfaces --test host_parity` passed.
- `git diff --check -- agent-harness/crates/ah-harness/tests/host_parity.rs` passed from `../runmat-private`.

Remaining:

- Continue with the larger mesh/solve/postprocess and GeometrySession/guided pane work, since the generic replay/run/variable/figure acceptance path is now materially better pinned.

### 2026-07-10: FEA Replay Field Workspace Rehydration Slice 1

Scope completed:

- Moved FEA field descriptor-to-workspace-entry construction into `run/fea-workspace-snapshot.ts` so live FEA runs and replayed FEA runs derive the same generic workspace variable entries.
- Updated live FEA run orchestration to use the shared workspace snapshot builder instead of a local helper.
- Updated FEA replay descriptor artifact hydration so loaded dataset/field descriptor artifacts also rebuild the replay session `workspaceSnapshot`.
- Added replay artifact coverage proving hydrated FEA descriptors become generic agent-visible `variables()` summaries with stable `variable_id`, `run_id`, `session_id`, `field_id`, lazy materialization, and paging metadata.
- Removed the duplicate FEA field descriptor id helper from FEA persistence and reused the shared descriptor helper.

Tests/evidence:

- `npm test -- src/replay/domain/fea-replay-artifacts.spec.ts src/replay/domain/replay-session-factory.spec.ts src/run/fea-run-orchestrator.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 8 tests.
- `npm test -- src/replay/domain/fea-replay-artifacts.spec.ts src/replay/domain/replay-session-factory.spec.ts src/run/fea-run-orchestrator.spec.ts src/run/fea-run-persistence.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 9 tests.
- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 62 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- desktop/src/run/fea-workspace-snapshot.ts desktop/src/run/fea-run-orchestrator.ts desktop/src/run/fea-run-persistence.ts desktop/src/replay/domain/fea-replay-artifacts.ts desktop/src/replay/domain/fea-replay-artifacts.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue Phase 2 acceptance by proving reloaded FEA result figures and field materialization stay fully inspectable through generic `select_run`, `variables`, `variable`, `figures`, and `show_figures` paths in the composed agent/runtime flow.

### 2026-07-10: Durable Figure Identity Remap Slice 1

Scope completed:

- Tightened execution-session figure remaps so `setFigures` derives handle mappings from all stable persisted figure identities, not just callers that manually set a scene-path remap.
- Extended figure remap normalization to record artifact figure ids, scene artifact paths, and preview artifact paths when a live handle is available.
- Added session-store assertions proving replay/live merge paths retain artifact-id remaps such as `sha256:*` alongside scene-path remaps.
- Verified the current FEA run persistence path records figure remaps for artifact figure id, scene path, and preview path, and keeps persisted FEA result figures in the normal run manifest/session artifact shape.

Tests/evidence:

- `npm test -- src/run/fea-run-persistence.spec.ts src/runtime/session-store.spec.ts --reporter=dot` passed from `../runmat-private/desktop` with 16 tests.
- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 62 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- desktop/src/run/fea-run-persistence.ts desktop/src/run/fea-run-persistence.spec.ts desktop/src/runtime/session-store.ts desktop/src/runtime/session-store.spec.ts` passed from `../runmat-private`.

Remaining:

- Continue proving durable FEA run datasets and result fields can be reloaded and inspected through generic run/variable/figure paths without a side-channel result API.

### 2026-07-10: Desktop Adapter FEA Variable Boundary Slice 1

Scope completed:

- Added `ah-runtime-adapter-desktop` coverage for the generic `variables` and `materialize_var` transport boundary.
- Proved FEA field summaries cross the Rust desktop adapter as generic variable summaries with stable `variable_id`, `run_id`, `session_id`, `field_id`, lazy materialization metadata, and paging metadata.
- Proved `materialize_var` forwards the full FEA field selector, including `variable_id`, `run_id`, `session_id`, `field_id`, `offset`, and `limit`, without degrading the request to a name-only selector.
- Re-ran the desktop React runtime provider bridge suite, which already covers selected and replayed FEA fields/figures through generic `variables`, `variable`, `figures`, and `show_figures` bridge commands.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-adapter-desktop --lib` passed with 1 test.
- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 62 tests.
- `git diff --check -- agent-harness/crates/ah-runtime-adapter-desktop/Cargo.toml agent-harness/crates/ah-runtime-adapter-desktop/src/adapter.rs agent-harness/crates/ah-tools/src/tests.rs` passed from `../runmat-private`.

Remaining:

- Continue Phase 2 by checking durable run/result artifact behavior for FEA figures and datasets, especially after reload.
- Clean stale design/plan terminology before using those docs as execution source for later slices.

### 2026-07-10: Agent Execute File Dispatch Slice 1

Scope completed:

- Added focused `ah-tools` coverage proving model-visible `execute({ target: "file", path })` dispatches to `RuntimeInterface::execute_file_with_options` rather than the inline/source execution path.
- Verified the file branch preserves `background`, returns normal run/session/runtime binding fields, and carries changed-variable metadata for FEA-style results.
- Re-ran the composed harness FEA solve/postprocess parity test that uses `execute` with `target: "file"` and then proceeds through generic `select_run`, `variables`, `figures`, `show_figures`, and `variable` surfaces.
- Kept this slice scoped to acceptance coverage because the implementation was already correctly routed; the risk was unpinned regression back to the old read-and-execute behavior.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools execute_file_tool_dispatches_to_file_runtime_path --lib` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools --lib` passed with 19 tests.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-harness agent_can_solve_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity` passed.

Remaining:

- Continue closing Phase 2 acceptance: make FEA field descriptors and materialization fully behave as generic variables, and keep result figures/artifacts durable through the normal run/dataset surface.
- Update stale design/plan references that still use the old load-condition `set` wording before relying on those docs for later slices.

### 2026-07-10: Shared Geometry Payload Contract Slice 1

Scope completed:

- Moved shared geometry host payload contracts into `runmat_runtime::geometry`: `GeometryPreviewBudgetPayload`, `GeometryPreviewBudgetPolicyPayload`, `GeometryPreviewTessellationProfilePayload`, `GeometryStatsPayload`, `GeometryInspectPayload`, and `GeometryPreviewPayload`.
- Removed duplicate WASM definitions of the same inspect/preview/budget payload structs and switched `runmat-wasm` to import the runtime-owned contracts.
- Removed duplicate Tauri definitions of the same payload structs and re-exported the runtime-owned contracts through the desktop runtime payload module for existing command/worker callers.
- Kept `GeometryPreviewSessionPayload` and `GeometryPreviewSessionProgressPayload` in the desktop Tauri layer because those are host UI streaming/session wrappers rather than runtime geometry contracts.
- Preserved the serialized API shape for existing TypeScript/browser/desktop callers while reducing cross-host drift.

Tests/evidence:

- `cargo fmt --all` passed in `runmat-analysis`.
- `cargo fmt --all --manifest-path ../runmat-private/desktop/src-tauri/Cargo.toml` passed.
- `cargo check -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host` passed.
- `cargo check --manifest-path ../runmat-private/desktop/src-tauri/Cargo.toml` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/geometry/geometry-preview-surface.spec.tsx src/app/components/fea/fea-study-surface.spec.tsx --reporter=dot` passed with 8 tests.
- `rg -n "struct GeometryPreviewBudgetPayload|enum GeometryPreviewBudgetPolicyPayload|struct GeometryPreviewTessellationProfilePayload|struct GeometryStatsPayload|struct GeometryInspectPayload|struct GeometryPreviewPayload" crates/runmat-wasm ../runmat-private/desktop/src-tauri crates/runmat-runtime/src/geometry/mod.rs -g '*.rs'` shows those definitions only in `crates/runmat-runtime/src/geometry/mod.rs`.
- `git diff --check` passed for touched files in both repositories.

Remaining:

- Continue the final audit for remaining host-only FEA/geometry envelopes and the larger plan acceptance tests around generic execution, check, run selection, variables, and figures.

### 2026-07-10: Typed FEA Progress Contract Slice 1

Scope completed:

- Added Rust-owned solver progress contracts to the generated TypeScript FEA contract surface: `FeaProgressPhase`, `FeaProgressStatus`, and `FeaProgressEvent`.
- Changed `FeaRunResult.progressEvents` from `unknown[]` to `FeaProgressEvent[]`.
- Changed runtime/WASM run payloads to preserve typed `FeaProgressEvent` values instead of converting progress through `serde_json::Value`.
- Re-exported progress contracts through `runmat_runtime::analysis` and switched the Tauri worker command channel, run payload, and progress collector to use typed progress events.
- Kept desktop `FeaEvent` as the UI event envelope, but changed browser/Tauri adapters so only the final UI event layer maps typed solver progress into sequence/timestamp/severity/kind fields.
- Typed desktop session-store persisted FEA `progressEvents` as `FeaProgressEvent[]`.
- Removed stale JSON progress parsing helpers and structural `Record<string, unknown>` progress parsing from the browser worker.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts` passed with 45 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt --all` passed in `runmat-analysis`.
- `cargo fmt --all --manifest-path ../runmat-private/desktop/src-tauri/Cargo.toml` passed.
- `cargo test -p runmat-runtime analysis_run_linear_static_emits_solver_and_artifact_progress_events --lib` passed.
- `cargo check -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host` passed.
- `cargo check --manifest-path ../runmat-private/desktop/src-tauri/Cargo.toml` passed.
- `rg -n "progress_tx: Option<tokio::sync::mpsc::UnboundedSender<serde_json::Value>>|Mutex<Vec<serde_json::Value>>|progress_events: Vec<JsonValue>|progressEvents\\?: unknown\\[\\]|progressEvents: unknown\\[\\]|progressFromFeaProgressRecord|json_number_as_f64" ../runmat-private/desktop/src-tauri ../runmat-private/desktop/src crates/runmat-wasm crates/runmat-runtime bindings/ts -g '*.rs' -g '*.ts' -g '*.tsx'` returned no matches.
- `git diff --check` passed for touched files in both repositories.

Remaining:

- Continue the final audit for geometry preview/debug helpers and any remaining host-only FEA payloads that should be generated or explicitly marked as UI envelopes.

### 2026-07-10: Load Condition Operation Clarity Slice 1

Scope completed:

- Replaced the ambiguous model-facing `finite_element_study_set_load_condition` tool with `finite_element_study_update_load_condition`.
- Renamed the Rust-owned `.fea` study document operation from `set_load_condition` to `update_load_condition`, then regenerated the TypeScript operation union from Rust.
- Changed load-condition update semantics from upsert to true replacement: add fails if the id exists, update fails if the id does not exist, remove fails if the id does not exist.
- Removed the now-unused upsert helper from FEA study document authoring.
- Updated harness tool registration, model-visible metadata, FEA/geometry guidance, direct runtime adapter tests, and TS operation-list assertions to use the new name.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts` passed with 45 tests.
- `cargo fmt --all` passed in `runmat-analysis`.
- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed with 4 tests.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools --lib` passed with 18 tests.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-adapter-direct --lib` passed with 3 tests.
- `rg -n "set_load_condition|finite_element_study_set_load_condition|SetLoadCondition" crates/runmat-runtime bindings/ts ../runmat-private/agent-harness ../runmat-private/desktop/src -g '*.rs' -g '*.ts' -g '*.tsx' -g '*.cjs' -g '*.d.ts'` returned no matches.
- `git diff --check` passed for touched files in both repositories.

Remaining:

- Continue the final audit for TS-owned event/progress contracts and geometry preview/debug helpers, separating real UI view models from Rust-owned runtime contracts.

### 2026-07-10: Binding Aggregate Results Removal Slice 1

Scope completed:

- Removed the public TS/WASM `feaResults` aggregate lookup surface so bindings no longer expose a duplicate result path beside run artifacts, field descriptors, and paged `feaField` materialization.
- Removed the Rust `AnalysisResultsLookupResult` binding payload contract and its TS generator mapping, keeping the lower-level runtime `analysis_results_by_run_id_op` available as the shared internal run/results abstraction.
- Regenerated FEA TS contracts and confirmed source/generated binding declarations no longer contain `FeaResultsResult`, `feaResults`, `AnalysisResultsLookupResult`, or the removed WASM payload helper.
- Kept the target direction explicit: FEA results should flow through run datasets, descriptors, variables/fields, figures, and paged materialization rather than a one-off aggregate API.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `cargo fmt --all` passed.
- `cargo test -p runmat-runtime analysis_field_page --lib` compiled `runmat-runtime`; the focused pattern matched 0 tests.
- `cargo test -p runmat-wasm --lib` passed; the crate currently has 0 lib unit tests.
- `cargo check -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host` passed.
- `rg -n "FeaResultsResult|feaResults|getFeaResults|runtime_fea_results|AnalysisResultsLookupResult|FeaResultsPayload|load_fea_results" bindings/ts crates/runmat-wasm crates/runmat-runtime ../runmat-private/desktop ../runmat-private/agent-harness -g '*.rs' -g '*.ts' -g '*.tsx' -g '*.d.ts'` returned no matches.

Remaining:

- Continue the final audit for any remaining side-channel host APIs, especially around geometry preview/debug helpers versus model-visible tool surfaces.

### 2026-07-10: Harness Turn Context Ingestion Slice 1

Scope completed:

- Added an `ah-core` submit-path regression that sends a real `AgentOp::UserTurn` with full FEA turn context through the harness engine.
- Captured the actual model request emitted by the engine and proved the submitted FEA study, workflow, visual selection, selected run, runtime figure monitor, and FEA guidance reach model-visible current-turn context.
- Guarded the same path against raw topology/evaluator leakage in the projected model request.
- Covered the chain from protocol `UserTurnContextSnapshot` through `apply_turn_context`, context assembly, projection, and model request construction.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-core user_turn_context_projects_fea_state_into_model_request -- --nocapture` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-core --lib` passed with 45 tests.

Remaining:

- Desktop submission, harness ingestion, and context projection are now covered as a chain. Continue auditing host adapter consistency and any remaining model-visible legacy geometry/result paths.

### 2026-07-10: Agent Turn Context Handoff Slice 1

Scope completed:

- Moved `AgentPanel` turn-context assembly into a focused `buildAgentTurnContext` helper so FEA, runtime selection, open paths, active file, and figure monitor state have a clear tested handoff point before submission to the harness.
- Added regression coverage proving FEA mode submits the computed `FeaTurnContextSnapshot` together with selected run and figure monitor state.
- Added regression coverage proving general-mode turns always submit `fea: null`, preventing stale FEA context from leaking when the user switches back to general agent work.
- Kept the helper narrow: it composes already-derived UI/runtime context and does not duplicate study parsing, workflow derivation, or geometry state logic.

Tests/evidence:

- `npm test -- src/app/components/agent/agent-turn-context.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/use-fea-agent-context-state.spec.tsx --reporter=dot` passed with 19 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- desktop/src/app/components/agent/agent-turn-context.ts desktop/src/app/components/agent/agent-turn-context.spec.ts desktop/src/app/components/agent/agent-panel.tsx` passed.

Remaining:

- The desktop turn-submission handoff is now directly covered. Continue the final audit for the harness-side `apply_turn_context` mapping and host adapter consistency so the submitted FEA state always reaches the model context and tools without side-channel behavior.

### 2026-07-10: FEA Context Projection Slice 1

Scope completed:

- Added composed `ah-context` coverage proving the assembled FEA snapshot is projected into the actual model request as current-turn state, not just retained as internal attachment metadata.
- The regression verifies model-visible workflow state, next action, visual region selection, selected run identity, and supported geometry extensions.
- The same regression guards the bounded-context rule by asserting raw topology/evaluator implementation terms do not leak into the projected FEA current-turn state.
- Preserved the existing cache-friendly layout: stable guidance remains in developer context, while FEA workflow/render/run state remains in the late dynamic current-turn message.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-context --lib` passed with 45 tests.

Remaining:

- The context projection path now has direct evidence for FEA workflow visibility. Continue the final audit for remaining brittle seams between the guided React pane state, harness turn context, and runtime/geometry tool surfaces.

### 2026-07-10: FEA Descriptor Contract Boundary Cleanup Slice 1

Scope completed:

- Tightened desktop FEA field descriptor handling so serialized field descriptors use the Rust-generated contract shape (`field_id`, `class_name`, `element_count`, `component_count`, `size_bytes`) rather than old camel-case aliases.
- Removed legacy descriptor alias acceptance from replay artifact hydration, replay session field descriptor reconstruction, FEA run event descriptor filtering, browser worker descriptor reads, result view-model normalization, and the generic agent variable bridge.
- Tightened the FEA descriptor test factory so tests must construct generated-contract descriptors instead of silently passing old descriptor property names.
- Preserved intentionally camel-case desktop view-state surfaces such as workspace `feaField.fieldId` and runtime command arguments/results, which are not the serialized FEA descriptor contract.

Tests/evidence:

- `npm test -- src/replay/domain/fea-replay-artifacts.spec.ts src/replay/domain/replay-session-factory.spec.ts src/runtime/domain/fea-results-view-model.spec.ts src/app/components/fea/fea-results-pane.spec.tsx src/run/fea-run-orchestrator.spec.ts src/run/fea-run-persistence.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/lanes/runtime-lane-manager.spec.ts --reporter=dot` passed with 27 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- The descriptor boundary no longer masks old TS-local shapes. Continue auditing any remaining TS-side FEA view models to ensure they derive display state from generated Rust/runtime contracts rather than becoming authoring or artifact authority.

### 2026-07-10: Direct Runtime Adapter Unified Surface Slice 1

Scope completed:

- Widened `ah-runtime-adapter-direct` so direct hosts can opt into the same unified runtime surface used by desktop and web: `runs`, `select_run`, `execute_file`, `variables`, `check`, `figures`, `show_figures`, `fea_capabilities`, typed finite element study operations, and `geometry_render`.
- Kept backward-compatible default capability errors for direct hosts that only implement the older minimal direct runtime interface.
- Delegated opted-in `.fea` file execution, generic checks, typed study operations, and geometry rendering through `DirectRuntimeAdapter` instead of forcing direct hosts through unavailable defaults.
- Added regression coverage for both default capability names and an opted-in unified FEA/geometry direct runtime.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-adapter-direct --lib` passed with 3 tests.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools --lib` passed with 18 tests.
- `git diff --check -- agent-harness/crates/ah-runtime-adapter-direct/Cargo.toml agent-harness/crates/ah-runtime-adapter-direct/src/lib.rs` passed.

Remaining:

- Direct-host runtime composition no longer lags the unified FEA/runtime surface structurally. Continue auditing host adapters and browser/native bridges for any remaining side-channel FEA or geometry behavior that bypasses the shared run/check/variable/figure abstractions.

### 2026-07-10: Selected FEA Study Rehydration Slice 1

Scope completed:

- Updated the FEA agent context hook so an existing selected `.fea` study can hydrate from the runtime filesystem even when its source is not the active editor buffer.
- Kept active editor contents as the fast path for the currently open study, but falls back to `runtimeClient.readFile(path)` for selected study files.
- Preserved the Rust-owned typed study summary path by feeding the loaded `.fea` source through `get_summary` instead of parsing study semantics in React.
- Added coverage proving a selected partial `.fea` study resumes with its study summary, readiness blockers, geometry reference, and workflow step rather than falling back to a generic review state.

Tests/evidence:

- `npm test -- src/app/components/agent/use-fea-agent-context-state.spec.tsx --reporter=dot` passed with 2 tests.
- `npm test -- src/app/components/agent/use-fea-agent-context-state.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed with 18 tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- FEA_GEOMETRY_AGENT_CHANGE_PROGRESS.md` passed.

Remaining:

- Selected study context can now rehydrate from durable `.fea` content. Continue proving the guided pane and context compiler consume the same workflow snapshot, and keep TS limited to view-model derivation rather than study-authoring authority.

### 2026-07-10: Desktop Aggregate FEA Results API Retirement Slice 1

Scope completed:

- Removed the desktop/Tauri aggregate `getFeaResults` / `feaResults` / `runtime_fea_results` path from the runtime client, browser worker, mock client, Tauri commands, Tauri worker, and test fixtures.
- Kept FEA result inspection on the shared field-backed variable path: completed FEA runs expose actual field entries, and field pages materialize through `getFeaField` from the generic `variable` tool.
- Removed synthetic FEA result marker variables from completed FEA run workspaces; the workspace now carries real field entries rather than `fea_run`, `fea_results`, or `fea_figures` placeholders.
- Fixed the native agent runtime bridge so it forwards generic `check` and `finite_element_study_operation` requests instead of leaving those transport variants unimplemented.
- Fixed native bridge variable materialization to forward the full selector payload, preserving `variable_id`, `run_id`, `session_id`, `field_id`, `offset`, and `limit` instead of collapsing the request to a name.

Tests/evidence:

- `cargo check --manifest-path ../runmat-private/desktop/src-tauri/Cargo.toml` passed.
- `npm test -- src/run/fea-run-orchestrator.spec.ts src/runtime/runtime-provider.spec.tsx src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/lanes/runtime-lane-manager.spec.ts src/runtime/hooks/runtimeHooks.spec.tsx src/runtime/hooks/useFileSystem.spec.tsx --reporter=dot` passed with 104 tests.
- `rg -n "FeaResultsPayload|getFeaResults|feaResults|runtime_fea_results|FeaResultsResult" ../runmat-private/desktop/src ../runmat-private/desktop/src-tauri/src` returns no matches.
- `git diff --check` passed for the touched desktop/Tauri bridge, runtime client, worker, orchestrator, and fixture files.

Remaining:

- The desktop model-visible aggregate result surface is retired. Continue auditing FEA-specific client method names that remain as adapter boundaries, especially `applyFeaStudyDocumentOperation`, to ensure they stay thin bindings to Rust-owned contracts rather than becoming TS source of truth again.

### 2026-07-10: FEA Workspace Marker Retirement Slice 1

Scope completed:

- Removed synthetic desktop workspace variables named `fea_run`, `fea_results`, and `fea_figures` from FEA run snapshots.
- Kept result metadata in the `ExecutionSession.fea` and run artifact surfaces, and kept result visualization in normal session figure refs.
- Kept actual FEA result fields as ordinary workspace entries with `feaField` run/field selectors, so Variables pane and agent materialization use the shared variable path.
- Added orchestrator coverage proving a completed FEA run workspace exposes the real field entry rather than broad marker variables.

Tests/evidence:

- `npm test -- src/run/fea-run-orchestrator.spec.ts src/runtime/hooks/useWorkspaceMaterializer.spec.ts --reporter=dot` passed for the orchestrator spec with 2 tests; Vitest found no separate materializer spec at that path.
- `npm test -- src/runtime/runtime-provider.spec.tsx src/runtime/domain/agent-runtime-bridge.spec.ts --reporter=dot` passed with 75 tests.
- `rg -n "fea_run|fea_results|fea_figures" ../runmat-private/desktop/src ../runmat-private/agent-harness/crates -g '!target'` now shows no desktop workspace marker construction; remaining matches are run-id/log strings, negative assertions, lower-level runtime client method names, and harness test fixture names.

Remaining:

- The desktop FEA workspace no longer carries ad-hoc result marker variables. Continue auditing lower-level `getFeaResults`/`getFeaField` client APIs and replay hydration to ensure they remain implementation details behind generic runtime variables/figures, not model-visible duplicate result tools.

### 2026-07-10: Variable Tool Metadata Cleanup Slice 1

Scope completed:

- Removed stale `variables` metadata entries for run/session/pattern/preview arguments that are not part of the current strict tool schema.
- Updated `variable` tool metadata to describe the actual paged field materialization inputs: `offset` and `limit`.
- Added a regression test that verifies the model-facing `variable` tool metadata names `offset`/`limit` and does not drift back to `preview_rows`/`preview_cols`.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-context --lib` passed with 44 tests.

Remaining:

- Tool schemas and metadata are aligned for the shared variable/FEA field inspection path. Continue the final audit for hidden compatibility shims that are not justified as UI view-model adaptation or persisted artifact hydration.

### 2026-07-10: Local FEA Execute File Dispatch Slice 1

Scope completed:

- Updated the CLI/local agent runtime so `execute_file` dispatches `.fea` documents through the finite element runtime path instead of reading YAML and executing it as MATLAB source.
- Added a first-class local FEA execution path for both study and sweep documents.
- Registered FEA result fields as lazy variable-pane entries scoped by real run IDs.
- Routed FEA field materialization through `analysis_results_by_run_id_op` with bounded paging, matching the shared run/variable inspection direction instead of adding a separate FEA-only surface.
- Cleared stale FEA field variables when a normal script run executes, so the selected local runtime state does not leak across run kinds.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-cli --lib` passed compile for the library target; it reported zero tests because `local_env.rs` is under the binary target.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-cli --bin runmat-agent local_env -- --nocapture` passed with 4 focused local runtime tests.

Remaining:

- CLI/local `.fea` execution now follows the shared FEA runtime path. Continue auditing for other agent-pane or harness paths that still bypass the shared run, variable, check, or artifact abstractions.

### 2026-07-10: Generated TS FEA Runtime API Contracts Slice 1

Scope completed:

- Moved FEA runtime API payload contracts out of wasm-local structs and into `crates/runmat-runtime/src/analysis/contracts.rs`.
- Added Rust-owned `AnalysisDocumentKind`, runtime capabilities, check result, run result, results lookup result, field page result, field request options, run dataset payload, field descriptors artifact payload, diagnostics artifact payload, and object artifact metadata contracts.
- Updated the wasm adapter to construct and serialize those runtime-owned contracts directly instead of returning local `JsonValue` descriptor envelopes.
- Extended `bindings/ts/scripts/generate-fea-contracts.cjs` to generate those runtime API/artifact contracts from Rust.
- Removed the duplicate hand-maintained TS `FeaCapabilities`, `FeaCheckResult`, `FeaRunResult`, `FeaResultsResult`, `FeaFieldRequestOptions`, `FeaFieldResult`, and artifact payload interfaces from `bindings/ts/src/index.ts`.
- Updated desktop test/client fixtures to satisfy the generated contract shape for guaranteed array fields.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/run/fea-run-orchestrator.spec.ts src/run/fea-run-persistence.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/domain/fea-results-view-model.spec.ts src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop` with 80 tests.
- `cargo test -p runmat-runtime operation_names_are_the_rust_owned_contract --lib` passed from `runmat-analysis`. The first attempt hit a transient `No space left on device` write failure in `target`; a retry passed.
- `cargo test -p runmat-wasm --lib` passed from `runmat-analysis`.
- `rg -n "export interface Fea(Capabilities|CheckResult|RunResult|RunDatasetPayload|FieldDescriptorsArtifactPayload|DiagnosticsArtifactPayload|ObjectArtifactMetadata|ResultsResult|FieldRequestOptions|FieldResult)|struct Fea(Capabilities|Check|Run|Results|Field).*Payload" bindings/ts/src crates/runmat-wasm/src/api/session.rs crates/runmat-runtime/src/analysis/contracts.rs` shows these TS interfaces are only defined in the generated contract module and the old wasm-local FEA payload structs are gone.

Remaining:

- The core FEA runtime API payloads are now Rust-owned and generated into TS. Continue the final audit for any desktop-specific parsing or presentation helpers that are accidentally acting as contract source rather than UI view-model code.

### 2026-07-10: Generated TS FEA Field Support Contracts Slice 1

Scope completed:

- Extended `bindings/ts/scripts/generate-fea-contracts.cjs` to read `AnalysisFieldKind`, `AnalysisFieldStorage`, `AnalysisFieldPagingDescriptor`, and `AnalysisFieldStorageRef` from Rust analysis contracts.
- Generated `FeaFieldKind`, `FeaFieldStorage`, `FeaFieldPagingDescriptor`, and `FeaFieldStorageRef` into `bindings/ts/src/generated/fea-study-document-contracts.ts`.
- Updated `bindings/ts/src/index.ts` to import/re-export those generated field support contracts.
- Removed the hand-maintained TS definitions for field kind, storage, paging descriptor, and storage ref.
- Left the higher-level `FeaFieldDescriptor` wrapper in `index.ts` for a deliberate follow-up, because desktop normalization and persisted artifact code still access historical camel-case aliases that should be cleaned in the same slice.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts -t "FEA study document contracts"` passed from `bindings/ts`.
- `cargo test -p runmat-runtime operation_names_are_the_rust_owned_contract --lib` passed from `runmat-analysis`.
- `rg -n "export type FeaFieldKind|export type FeaFieldStorage|export interface FeaFieldPagingDescriptor|export interface FeaFieldStorageRef" bindings/ts/src/index.ts bindings/ts/src/generated/fea-study-document-contracts.ts` shows these support contracts only defined in the generated contract module.
- `git diff --check -- bindings/ts/scripts/generate-fea-contracts.cjs bindings/ts/src/generated/fea-study-document-contracts.ts bindings/ts/src/index.ts bindings/ts/src/index.spec.ts FEA_GEOMETRY_AGENT_CHANGE_PROGRESS.md` passed.

Remaining:

- Field support contracts are generated from Rust. The higher-level `FeaFieldDescriptor`, `FeaRunResult`, `FeaResultsResult`, `FeaFieldResult`, and dataset payload wrappers remain the next contract-cleanup targets.

### 2026-07-10: Generated TS FEA Analysis Artifact Constants Slice 1

Scope completed:

- Extended `bindings/ts/scripts/generate-fea-contracts.cjs` to read runtime-owned FEA analysis constants from `crates/runmat-runtime/src/analysis/contracts.rs`.
- Generated the FEA run dataset schema versions, artifact kinds, and field paging defaults into `bindings/ts/src/generated/fea-study-document-contracts.ts`.
- Updated `bindings/ts/src/index.ts` to import and re-export those generated constants.
- Removed the hand-maintained copies of runtime-owned analysis constants from `bindings/ts/src/index.ts`.
- Left `FEA_RUN_KIND`, `FEA_RUN_CELL_ID`, and `FEA_RUN_MANIFEST_METADATA_SCHEMA_VERSION` in `index.ts` because those are run-history manifest concepts, not analysis-result contract constants.
- Added focused TS test coverage for the generated artifact constants.

Tests/evidence:

- `npm run generate:fea-contracts` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts -t "FEA study document contracts"` passed from `bindings/ts` with generated artifact constants and operation names covered.
- `npm run build:types` passed from `bindings/ts`.
- `cargo test -p runmat-runtime operation_names_are_the_rust_owned_contract --lib` passed from `runmat-analysis`.
- `rg -n "export const FEA_(RUN_DATASET_SCHEMA_VERSION|FIELD_DESCRIPTORS_SCHEMA_VERSION|DIAGNOSTICS_SCHEMA_VERSION|OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION|RUN_DATASET_KIND|DATASET_ARTIFACT_KIND|FIELD_DESCRIPTORS_ARTIFACT_KIND|DIAGNOSTICS_ARTIFACT_KIND|ARTIFACT_MANIFEST_KIND|FIELD_DEFAULT_PAGE_SIZE|FIELD_DEFAULT_MATERIALIZE_LIMIT)" bindings/ts/src/index.ts bindings/ts/src/generated/fea-study-document-contracts.ts` shows these constants only defined in the generated contract module.
- `git diff --check -- bindings/ts/scripts/generate-fea-contracts.cjs bindings/ts/src/generated/fea-study-document-contracts.ts bindings/ts/src/index.ts bindings/ts/src/index.spec.ts FEA_GEOMETRY_AGENT_CHANGE_PROGRESS.md` passed.

Remaining:

- Runtime-owned FEA analysis constants are no longer manually mirrored in TS. Continue auditing the remaining `FeaRunResult`, `FeaResultsResult`, `FeaFieldDescriptor`, and related payload interfaces, which still partially mirror wasm/runtime payload structs.

### 2026-07-10: Generated TS FEA Study Result Interfaces Slice 1

Scope completed:

- Extended `bindings/ts/scripts/generate-fea-contracts.cjs` so it parses the Rust `.fea` authoring structs in `fea_document_authoring.rs`.
- Generated `bindings/ts/src/generated/fea-study-document-contracts.ts` with the operation list, operation union, and 13 study-document result/summary interfaces.
- Renamed the generated module from operation-specific naming to contract naming so the module boundary reflects what it owns.
- Removed the hand-maintained `FeaStudy*` result/summary interface block from `bindings/ts/src/index.ts`.
- Updated the public `runmat` entrypoint to import/re-export the generated study-document operation/result types.
- Kept broad generated output ignored while allowing the specific source contract file to be tracked.

Tests/evidence:

- `npm run generate:fea-contracts` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts -t "FEA study document contracts"` passed from `bindings/ts`.
- `npm run build:types` passed from `bindings/ts`.
- `cargo test -p runmat-runtime operation_names_are_the_rust_owned_contract --lib` passed from `runmat-analysis`.
- `rg -n "export interface FeaStudy|export type FeaStudyDocumentOperation|FEA_STUDY_DOCUMENT_OPERATIONS" bindings/ts/src/index.ts bindings/ts/src/generated/fea-study-document-contracts.ts` shows the `FeaStudy*` interfaces only in the generated contract module, with `index.ts` re-exporting the generated contract.
- `git diff --check -- .gitignore bindings/ts/package.json bindings/ts/scripts/generate-fea-contracts.cjs bindings/ts/src/generated/fea-study-document-contracts.ts bindings/ts/src/index.ts bindings/ts/src/index.spec.ts FEA_GEOMETRY_AGENT_CHANGE_PROGRESS.md` passed.

Remaining:

- `.fea` study operation/result TS drift is now removed. Continue the final binding-contract audit for FEA run/result dataset shapes and any remaining desktop-only interpretation that should move to generated or Rust-owned contracts.

### 2026-07-10: Generated TS FEA Study Operation Contract Slice 1

Scope completed:

- Added `bindings/ts/scripts/generate-fea-contracts.cjs`, which reads the Rust-owned `FEA_STUDY_DOCUMENT_OPERATION_NAMES` constant from `fea_document_authoring.rs`.
- Generated `bindings/ts/src/generated/fea-study-document-operations.ts` with `FEA_STUDY_DOCUMENT_OPERATIONS` and the `FeaStudyDocumentOperation` union type.
- Changed the public `runmat` TS entrypoint to import/re-export the generated operation contract instead of hand-maintaining the union in `index.ts`.
- Added `npm run generate:fea-contracts` and wired it into `npm run build:types`.
- Adjusted `.gitignore` so this specific generated source contract is tracked while the broader generated builtins and dist outputs remain ignored.
- Added a focused package test proving the exported operation list is available from `runmat`.

Tests/evidence:

- `npm run generate:fea-contracts` passed from `bindings/ts`.
- `npm test -- src/index.spec.ts -t "FEA study document contracts"` passed from `bindings/ts`.
- `npm run build:types` passed from `bindings/ts`.
- `cargo test -p runmat-runtime operation_names_are_the_rust_owned_contract --lib` passed from `runmat-analysis`.
- `git diff --check -- .gitignore bindings/ts/package.json bindings/ts/scripts/generate-fea-contracts.cjs bindings/ts/src/generated/fea-study-document-operations.ts bindings/ts/src/index.ts bindings/ts/src/index.spec.ts` passed.

Remaining:

- Operation-name drift is now removed from the TS package. Larger `.fea` result/summary interface shapes are still manually mirrored in `bindings/ts/src/index.ts`; those should be handled by a broader Rust-to-TS schema generation path or a narrower generated contract module before final completion.

### 2026-07-10: Rust-Owned FEA Study Operation Contract Slice 1

Scope completed:

- Added `FeaStudyDocumentOperation` as the Rust-owned operation vocabulary for `.fea` study authoring.
- Added `FEA_STUDY_DOCUMENT_OPERATION_NAMES` as the exported authoritative operation-name list.
- Split the Rust authoring API into a typed entrypoint, `apply_fea_study_document_operation_typed`, and a string-parsing wrapper for wasm/JS edge compatibility.
- Removed raw string dispatch from the `.fea` authoring mutation path; unsupported operation handling now happens at the boundary parser.
- Re-exported the typed operation API from `runmat_runtime::analysis` so internal Rust callers do not need to go through stringly-typed dispatch.
- Added a contract test proving the exported names parse back to the Rust enum and reject unknown operation names.

Tests/evidence:

- `cargo fmt --all` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed from `runmat-analysis`.
- `cargo test -p runmat-wasm --lib` passed from `runmat-analysis` after the typed wrapper change, verifying the wasm crate still compiles against the string edge adapter.

Remaining:

- The runtime behavior and Rust authoring layer are now Rust-owned, but `bindings/ts/src/index.ts` still manually mirrors the `.fea` operation/result TypeScript shapes. The next cleanup should either introduce a real generated binding path for these exported Rust contracts or otherwise remove the duplicated TS source of truth.

### 2026-07-10: Generic FEA Report Artifact Acceptance Slice 1

Scope completed:

- Added host-parity coverage for a completed FEA run being inspected through generic runtime surfaces before report creation.
- Proved the report path uses the ordinary filesystem `write` tool rather than an FEA-specific report API or inline artifact event.
- Verified the report references the study, run, field, and figure IDs while keeping field data lazy-paged and excluding image/base64 payloads.
- Extended the mock desktop FEA result mode to support the report flow without changing the runtime tool surface.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_write_fea_report_from_generic_runtime_surfaces --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.

Remaining:

- The report/export user artifact path is now acceptance-covered through generic tools. The main remaining smell is the `.fea` document authoring contract still being partially surfaced as TS runtime-client types and worker dispatch; unwind that toward Rust-owned definitions/generated bindings next.

### 2026-07-10: Guided Results And Report Workflow Slice 1

Scope completed:

- Updated the guided FEA workflow state so ready-to-solve studies explicitly prompt the agent to set or confirm mesh settings with `finite_element_study_set_mesh` before generic `check` and `execute`.
- Updated the ready-to-solve prompt to route postprocessing through generic `select_run`, `variables`, `figures`, `show_figures`, and bounded `variable` inspection.
- Added an `inspect_results` workflow state for completed selected FEA runs.
- Added guided actions for inspecting completed result fields/figures and creating a concise Markdown report without embedding large field data.
- Kept this logic in the FEA context/workflow boundary consumed by both the pane and model context.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/app/components/agent/fea-agent-context.ts desktop/src/app/components/agent/fea-agent-context.spec.ts desktop/src/runtime/runtime-provider.spec.tsx agent-harness/crates/ah-harness/tests/host_parity.rs agent-harness/crates/ah-harness/tests/support/host.rs` passed.

Remaining:

- The guided pane now steers mesh, solve, postprocess, and report creation through the right generic surfaces. Continue final audit for whether the actual user-facing report generation needs a composed host-parity acceptance test or whether the existing generic `write`/Markdown report behavior is sufficient.

### 2026-07-10: Mesh Step In Composed Solve Loop Slice 1

Scope completed:

- Strengthened the desktop composed FEA bridge flow so geometry evidence now routes through typed region/constraint/load edits, `finite_element_study_operation` `set_mesh`, typed output selection, generic `check`, and generic `execute_file`.
- Made the desktop composed-flow readiness/check test double require mesh settings before solve, so removing the mesh step breaks the acceptance path.
- Strengthened the harness host-parity solve/postprocess model to call `finite_element_study_set_mesh` before `check` and `execute`.
- Updated host-parity expectations so the generic runtime solve loop includes `finite_element_study_set_mesh -> check -> execute -> select_run -> variables -> figures -> show_figures -> variable`.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx -t "runs a composed agent FEA flow" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.
- `git -C ../runmat-private diff --check -- desktop/src/runtime/runtime-provider.spec.tsx agent-harness/crates/ah-harness/tests/host_parity.rs agent-harness/crates/ah-harness/tests/support/host.rs` passed.

Remaining:

- This closes the missing mesh setup in the main composed solve/postprocess acceptance path. Continue final audit for user-facing report/export behavior and any remaining workflow-pane/runtime-state mismatch before claiming Phase 8 complete.

### 2026-07-10: Geometry Summary Compaction Module Slice 1

Scope completed:

- Moved bounded geometry summary reduction out of `agent-harness/crates/ah-tools/src/tools/geometry.rs`.
- Added `agent-harness/crates/ah-tools/src/tools/geometry/summary.rs` as the home for compact geometry summaries, bounded region/diagnostic samples, CAD/mesh summary shaping, and raw-topology shrink/drop policy.
- Moved raw-topology regression tests with the compaction code so model-context density and topology exclusion are tested at the correct boundary.
- Reduced `geometry.rs` from roughly 1130 lines to roughly 776 lines; it now focuses on tool handler orchestration while state and summary policy live in submodules.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests geometry` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests` passed from `../runmat-private`.
- `git -C ../runmat-private diff --check -- agent-harness/crates/ah-tools/src/tools/geometry.rs agent-harness/crates/ah-tools/src/tools/geometry/state.rs agent-harness/crates/ah-tools/src/tools/geometry/summary.rs` passed.

Remaining:

- `geometry.rs` is no longer a severe god file, but the remaining tool handlers can still be grouped later if another concrete responsibility boundary appears. The more important remaining work is now end-to-end workflow completion and final requirement audit, not more mechanical splitting.

### 2026-07-10: Workspace Tool Semantics Slice 1

Scope completed:

- Corrected desktop workspace bridge behavior so `open_path` makes a file visibly available as a pinned tab without changing the active selected workspace target.
- Kept `select_path` as the active-target operation and made it open/select files as pinned editor targets when needed.
- Updated model-facing `open_path` and `select_path` metadata to match the target design: open for user inspection, select for future agent context.
- Updated model-facing `execute` metadata and stable tool guidance so `execute({ target: "file", path })` is described as supporting executable artifacts such as `.m` scripts and `.fea` studies, not just `.m` files.
- Added context catalog coverage proving the open-vs-select distinction remains explicit.

Tests/evidence:

- `npm test -- src/app/components/shell/editor-panel.spec.ts src/agent/clients/browser/index.spec.ts -t "workspace bridge|agent workspace" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --tests` passed from `../runmat-private`.
- `git -C ../runmat-private diff --check -- desktop/src/app/components/shell/editor-agent-workspace-bridge.ts desktop/src/app/components/shell/editor-panel.tsx desktop/src/app/components/shell/editor-panel.spec.ts agent-harness/crates/ah-context/src/tools/metadata.rs agent-harness/crates/ah-context/src/tools/catalog_tests.rs agent-harness/crates/ah-context/src/sections/tools.rs` passed.

Remaining:

- This closes a concrete Phase 1 semantic mismatch. Continue auditing model-facing tool contracts for stale `.m`-only or pre-typed-FEA wording that could lead the agent away from the composed `.fea` flow.

### 2026-07-10: Geometry Tool State Module Slice 1

Scope completed:

- Split geometry session lifecycle and mutable state out of `agent-harness/crates/ah-tools/src/tools/geometry.rs`.
- Added `agent-harness/crates/ah-tools/src/tools/geometry/state.rs` to own session ids, snapshots, camera/view state, visibility state, section state, current selection, created regions, and compact region candidates.
- Updated geometry tool handlers to call state-store methods instead of directly constructing sessions or reaching into the session map.
- Kept summary compaction and individual tool handlers in `geometry.rs` for this slice, avoiding a broad mechanical rewrite.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests geometry` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests` passed from `../runmat-private`.
- `git -C ../runmat-private diff --check -- agent-harness/crates/ah-tools/src/tools/geometry.rs agent-harness/crates/ah-tools/src/tools/geometry/state.rs` passed.

Remaining:

- `geometry.rs` is still large at roughly 1130 lines. The next high-value split is likely summary compaction into its own module or grouping geometry tool handlers by session/query/selection responsibilities, but the state/data ownership smell is now removed.

### 2026-07-10: FEA Study View Model Domain Boundary Slice 1

Scope completed:

- Moved the FEA study editor model helpers out of the React component folder and into `desktop/src/runtime/domain/fea-study-view-model.ts`.
- Moved the corresponding helper tests into `desktop/src/runtime/domain/fea-study-view-model.spec.ts`.
- Updated the FEA study surface and review model to consume the runtime-domain projection instead of treating component code as the home for study document interpretation.
- Removed the old component-local `fea-study-view-model.ts` and spec files.

Tests/evidence:

- `npm test -- src/runtime/domain/fea-study-view-model.spec.ts src/app/components/fea/fea-study-review-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/runtime/domain/fea-study-view-model.ts desktop/src/runtime/domain/fea-study-view-model.spec.ts desktop/src/app/components/fea/fea-study-surface.tsx desktop/src/app/components/fea/fea-study-review-model.ts desktop/src/app/components/fea/fea-study-review-model.spec.ts desktop/src/app/components/fea/fea-study-view-model.ts desktop/src/app/components/fea/fea-study-view-model.spec.ts` passed.
- `rg` over `desktop/src` finds no remaining `study-document` path and only runtime-domain references to `fea-study-view-model`.

Remaining:

- The Rust/runtime binding is still the study document source of truth. This slice only moved TS-side view projection to the runtime-domain layer; future slices should avoid expanding this into a parallel schema or parser.

### 2026-07-10: Agent Show Figures Domain Boundary Slice 1

Scope completed:

- Moved `show_figures` request parsing, figure matching, monitor slot clamping, monitor image response shaping, and selected-session derivation out of `desktop/src/runtime/runtime-provider.tsx`.
- Added `executeAgentShowFigures` to `desktop/src/runtime/domain/agent-runtime-bridge.ts`.
- Kept the provider responsible for supplying projected runs, selected session id, monitor image capture, monitor slot state update, selection side effects, and logging.
- Added domain coverage for bounded monitor slots, selected run id derivation, monitor image capture, and invalid payload rejection.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "show_figures|inspects replayed finite element figures|composed agent FEA flow" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/runtime/domain/agent-runtime-bridge.ts desktop/src/runtime/domain/agent-runtime-bridge.spec.ts desktop/src/runtime/runtime-provider.tsx` passed.

Remaining:

- `runtime-provider.tsx` is still too large, but the bridge-specific figure policy now sits in a tested runtime-domain helper. Continue extracting only behavior with a real domain boundary and regression coverage.

### 2026-07-10: FEA Agent Context State Hook Slice 1

Scope completed:

- Moved FEA capability loading, active `.fea` summary hydration, visual-state projection, selected-run projection, and `buildFeaAgentContext` assembly out of `desktop/src/app/components/agent/agent-panel.tsx`.
- Added `desktop/src/app/components/agent/use-fea-agent-context-state.ts` as the FEA agent state boundary consumed by the shell panel.
- Updated the panel to use the hook's computed context for FEA mode selection, including runtime-provided document and geometry extensions.
- Added hook-level coverage proving runtime FEA capabilities and generated `.fea` document summary output hydrate the active model context.

Tests/evidence:

- `npm test -- src/app/components/agent/use-fea-agent-context-state.spec.tsx src/app/components/agent/agent-panel-body.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/app/components/agent/agent-panel.tsx desktop/src/app/components/agent/use-fea-agent-context-state.ts desktop/src/app/components/agent/use-fea-agent-context-state.spec.tsx` passed.

Remaining:

- The agent panel is still large, but it no longer owns FEA capability/summary async loading or context assembly. Continue extracting only clear shell-vs-domain seams rather than splitting UI event plumbing for its own sake.

### 2026-07-10: Runtime Provider Agent Bridge Dispatch Slice 1

Scope completed:

- Removed duplicated agent runtime command dispatch in `desktop/src/runtime/runtime-provider.tsx`.
- Introduced one provider-local `dispatchAgentRuntimeBridgeCommand` used by both the browser worker bridge callback and the Tauri `agent://runtime_request` listener.
- Kept generic runtime commands, typed FEA study operations, generic `check`, `execute_file`, geometry rendering, run selection, variables, figures, and `show_figures` on the same dispatch path.
- Preserved `runs({ limit })` behavior for both bridge entry points instead of having the limit handling only in one switch.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx -t "composed agent FEA flow|selects projected runs|routes runtime bridge" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "accepts Map payloads|exposes selected finite element fields|inspects replayed finite element figures|notifies the shell|selects runtime sessions|rejects run id payloads|composed agent FEA flow" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/runtime/runtime-provider.tsx` passed.

Remaining:

- This removes a concrete duplication in a large provider file. The provider is still large, so future high-value slices should continue moving domain behavior into `runtime/domain` or application services when the boundary is clear and testable.

### 2026-07-10: FEA Context Provider Boundary Slice 1

Scope completed:

- Extracted FEA context attachment projection out of the generic dynamic attachment builder into `agent-harness/crates/ah-context/src/sections/fea.rs`.
- Kept `build_dynamic_attachments` responsible for assembly/order only; the FEA-specific JSON shape, active-kind normalization, model guidance text, source id, and raw-topology regression now live with the FEA context provider.
- Added provider-level coverage for workflow, visual selection, selected run, and raw-topology exclusion.
- Kept an attachment-level regression that proves the generic dynamic attachment builder includes the FEA provider output.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --tests` passed from `../runmat-private`.
- `git -C ../runmat-private diff --check -- agent-harness/crates/ah-context/src/sections/attachments.rs agent-harness/crates/ah-context/src/sections/fea.rs agent-harness/crates/ah-context/src/sections/mod.rs` passed.
- `rg` over `agent-harness/crates/ah-context/src/sections` finds retired `geometry_inspect`, `geometry_view`, and `fea_run` names only in negative assertions.

Remaining:

- This is a focused Phase 6 boundary improvement, not the full long-term pluggable context-provider interface. The remaining audit should continue looking for large generic files that still own FEA/geometry domain projection or policy directly.

### 2026-07-10: Browser Worker FEA Operation Contract Slice 1

Scope completed:

- Tightened the browser worker FEA document-operation transport payload so `operation` uses the generated `runmat` `FeaStudyDocumentOperation` type instead of a raw `string`.
- Tightened the worker-side `applyFeaStudyDocumentOperation` session method shape to accept the same generated operation type.
- Tightened the runtime bridge domain spec's FEA operation test double to use the same generated operation type, so tests no longer model the operation vocabulary as an arbitrary string.
- Kept this as a thin transport/client boundary cleanup; the Rust/runtime binding remains the source of truth for the operation vocabulary.

Tests/evidence:

- `npm test -- src/agent/clients/browser/index.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/runtime/clients/browser/worker/transport.ts desktop/src/runtime/clients/browser/worker/worker.ts desktop/src/agent/clients/browser/index.spec.ts` passed.

Remaining:

- Continue auditing browser/worker runtime client plumbing for any other raw FEA schema shadows. Current lower-level `checkFeaStudy` and `runFeaStudy` names are runtime client methods, but they should stay thin adapters over generic agent-facing `check`/`execute` and runtime-owned results.

### 2026-07-10: Browser Runtime Bridge FEA Flow Coverage Slice 1

Scope completed:

- Added browser harness client regression coverage for worker `runtimeBridgeRequest` events carrying composed FEA runtime operations through the active desktop runtime bridge callback.
- Covered the bridge sequence `geometry_render -> finite_element_study_operation -> check -> execute_file` from worker-side requests to main-thread runtime handling.
- Verified the exact operation names and payloads passed across the bridge, including typed study mutation, generic validation, and generic file execution.
- Verified runtime bridge responses are delivered back to the worker for study edit, check, and execute results.
- Strengthened browser/agent-level coverage for the composed FEA flow without adding duplicate FEA result tools or a separate postprocessing surface.

Tests/evidence:

- `npm test -- src/agent/clients/browser/index.spec.ts -t "routes composed FEA runtime bridge requests" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/agent/clients/browser/index.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/agent/clients/browser/index.spec.ts` passed.

Remaining:

- Continue the final audit against remaining browser/worker runtime client names such as `checkFeaStudy`, `runFeaStudy`, and `feaStudyDocumentOperation`. These are currently lower-level runtime/WASM client bindings rather than component-owned source-of-truth models, but the final state should keep them thin and generated/runtime-owned where possible.

### 2026-07-10: FEA Results View-Model Domain Boundary Slice 1

Scope completed:

- Moved FEA result/run payload interpretation out of the React component area and into `desktop/src/runtime/domain/fea-results-view-model.ts`.
- Moved the corresponding tests from the FEA component folder into `desktop/src/runtime/domain/fea-results-view-model.spec.ts`.
- Updated the FEA results pane to consume the runtime-domain view model and formatting helpers instead of owning result normalization next to UI code.
- Deleted the old component-local `fea-results-data.ts` and `fea-results-data.spec.ts` files so the result view model has a single domain home.

Tests/evidence:

- `npm test -- src/runtime/domain/fea-results-view-model.spec.ts src/app/components/fea/fea-results-pane.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git -C ../runmat-private diff --check -- desktop/src/app/components/fea/fea-results-pane.tsx desktop/src/runtime/domain/fea-results-view-model.ts desktop/src/runtime/domain/fea-results-view-model.spec.ts desktop/src/app/components/fea/fea-results-data.ts desktop/src/app/components/fea/fea-results-data.spec.ts` passed.

Remaining:

- This removes one of the last component-layer result payload normalizers. The final audit should now focus on any remaining non-generic FEA result paths, lower-level geometry bridge names that are still exposed beyond runtime internals, and whether the guided pane has enough composed browser/desktop coverage to satisfy the full Phase 7/8 user-flow acceptance.

### 2026-07-10: Live FEA Solve/Postprocess Generic Runtime Acceptance Slice 1

Scope completed:

- Added a composed host-parity acceptance flow where the scripted model uses only generic runtime tools to check, solve, select, and inspect a live `.fea` run.
- The asserted tool sequence is:
  `check -> execute -> select_run -> variables -> figures -> show_figures -> variable`.
- Extended the desktop runtime adapter test double with a live finite element solve mode that returns a normal execution result, live FEA run summary, finite element field variable, stable result figure id, and bounded field materialization.
- Verified `execute({ target: "file", path })` returns live run/session/runtime binding metadata for the `.fea` solve path.
- Verified `select_run`, `variables`, `figures`, `show_figures`, and `variable` expose live FEA outputs through the same run, variable, figure, and paged materialization surfaces used for other runtime runs.
- Corrected the generic `execute` tool schema so the file branch requires `path` and the inline branch requires `code`; unused branch fields no longer need to be passed as explicit `null`.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_solve_fea_and_postprocess_with_generic_runtime_surfaces --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context -p ah-tools --tests` passed from `../runmat-private`.

Remaining:

- The main live solve/postprocess acceptance gap is now covered at the harness/tool boundary. Remaining work should be driven by final requirement audit, with special attention to any stale TS-side result payload normalization or lingering non-generic result paths.

### 2026-07-10: FEA Physics Catalog Source-of-Truth Cleanup Slice 1

Scope completed:

- Added a `runmat` binding-edge FEA physics profile catalog with profile id, label, family, target, user value, and default output fields.
- Switched the desktop FEA study review tree to consume the `runmat` catalog instead of owning a separate hand-written profile list.
- Removed desktop substring heuristics for default result fields; default output rows now come from the exported profile catalog.
- Switched the guided FEA pane's supported-family summary to derive from the same catalog family list rather than a separate hard-coded sentence.
- Rebuilt the ignored local `bindings/ts/dist` artifacts so the desktop compiler sees the new exports through its existing `runmat` path mapping.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `npm test -- src/app/components/fea/fea-study-review-model.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check -- bindings/ts/src/index.ts bindings/ts/dist/index.d.ts bindings/ts/dist/index.js` passed in `runmat-analysis`.
- `git -C ../runmat-private diff --check -- desktop/src/app/components/fea/fea-study-review-model.ts desktop/src/app/components/agent/fea-agent-state.ts` passed.

Remaining:

- The major remaining workflow gap is still full mesh/solve/postprocess acceptance through generic `check`, `execute`, `select_run`, `variables`, `figures`, and paged field inspection. The source-of-truth cleanup here removes a remaining desktop-owned domain catalog before that solve loop is hardened.

### 2026-07-10: Selected Geometry Region To Typed Constraint Acceptance Slice 1

Scope completed:

- Added a composed host-parity acceptance path where the scripted agent opens a geometry session, selects a rendered face/region, creates a named geometry-session region, writes that region into the `.fea` study with `finite_element_study_add_region`, adds a fixed constraint with `finite_element_study_add_constraint`, and validates with generic `check`.
- The test proves the selected-region guided setup path uses session-oriented geometry tools plus typed study operations, not raw `.fea` patching or invented selectors.
- The sequence asserts the model-visible tool order:
  `geometry_open_session -> geometry_select -> geometry_create_region -> finite_element_study_add_region -> finite_element_study_add_constraint -> check`.
- The test also asserts the geometry-created region returns `finite_element_study_add_region_input`, the selector is `region:face_mount`, and subsequent study/check outputs target `studies/bracket.fea`.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_turn_selected_geometry_region_into_typed_fea_constraint --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests geometry` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context --tests` passed from `../runmat-private`.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Remaining:

- This closes a key composed proof for Phase 4/7 region-to-study setup. The remaining end-to-end workflow gap is the full mesh/solve/postprocess loop driven from guided setup through generic `execute`, `variables`, `figures`, and report/report-like outputs.

### 2026-07-10: FEA Guided Pane Selected-Region Actions Slice 1

Scope completed:

- Made FEA workflow derivation consume the current visual state, including the selected rendered region selector from the editor/geometry surface.
- Added selected-region guided actions for `pick_fixed_areas` and `pick_loaded_areas`.
- The fixed-area action tells the agent to use the exact current selector, create a durable named region with `finite_element_study_add_region` if needed, add the fixed constraint with `finite_element_study_add_constraint`, then run `check`.
- The load-area action uses the exact current selector as the load-condition target, asks for load type/magnitude/direction before editing, then routes through `finite_element_study_add_region`, `finite_element_study_add_load_condition`, and `check`.
- Guarded these actions so they appear only when the visual state contains a grounded `selectionSelector`; unresolved visual selections do not produce typed edit prompts.
- Added a guided-pane UI regression proving the selected-region action renders and dispatches the typed-tool prompt.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/fea-agent-status-panel.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Remaining:

- This moves the guided pane from display-only visual selection toward actionable FEA setup. The deeper Phase 7 acceptance flow still needs direct composed coverage that a selected region action results in typed `.fea` edits and validation through the agent/harness loop.

### 2026-07-10: Geometry Render Bridge Naming Cleanup Slice 1

Scope completed:

- Renamed the internal agent runtime geometry bridge from `geometry_render_view` to `geometry_render` so the bridge, model-visible tool, desktop runtime provider, Tauri bridge, web host, and browser harness transport all use the same session-render vocabulary.
- Renamed the Rust runtime-interface request/result types from `GeometryViewRequest` / `GeometryRenderViewResult` to `GeometryRenderRequest` / `GeometryRenderResult`.
- Updated desktop runtime-domain helpers from `executeAgentGeometryRenderView` / `AgentGeometryRenderViewResult` to `executeAgentGeometryRender` / `AgentGeometryRenderResult`.
- Removed exact legacy `geometry_view` capability/context labels from implementation paths; remaining `geometry_view` references are negative assertions that prove the old model-visible tool is absent.
- Added a regression that unsupported geometry rendering reports the current `geometry_render` capability name rather than the retired `geometry_view` name.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts -t "renders bounded geometry view images" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "renders geometry views for agent geometry session tools" --reporter=dot` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-runtime-interface -p ah-runtime-adapter-desktop -p ah-tools geometry --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-web-host -p ah-cli --tests geometry` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity geometry` compiled the host-parity test target successfully; the filter matched zero tests.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools unsupported_geometry_render_uses_current_capability_name --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools geometry_session_tools_do_not_expose_raw_topology_payloads --tests` passed from `../runmat-private`.
- `rg` found no stale `geometry_render_view`, `GeometryRenderView`, `GeometryViewRequest`, `GeometryRenderViewResult`, `AgentGeometryRenderView`, `executeAgentGeometryRenderView`, or `normalizeGeometryView` implementation symbols under agent harness, desktop runtime, or Tauri agent bridge sources.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Remaining:

- Continue final audit for lower-level runtime geometry inspection APIs that may remain legitimate UI/runtime helpers; this slice removes the old agent bridge vocabulary and exact legacy capability labels from the implementation path.

### 2026-07-10: FEA Editor-to-Agent Visual Selection Slice 1

Scope completed:

- Added a UI-only `FeaEditorSelectionSnapshot` for the current rendered FEA study selection; it carries study path, geometry source, selected region id/selector, and render availability without becoming a new TS document source of truth.
- Wired `FeaStudySurface` to publish the selected review-row region when that region is grounded in the rendered/previewed geometry.
- Stored the visual selection in `EditorProvider`, cleared it on active editor path changes, exposed it through `useEditorContext`, and fed it into the FEA agent visual snapshot.
- Filtered editor selections against the active `.fea`/geometry path so stale selections from other tabs do not enter model context.
- Updated the FEA guided status panel to show the selected graphical region when available.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/fea/fea-study-surface.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/agent/agent-panel-body.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/app/components/fea/fea-study-surface.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Remaining:

- Continue the final audit for any remaining document-authority drift in desktop creation/edit flows, but the model can now receive current FEA editor graphical selection without a duplicate TS `.fea` model.

### 2026-07-10: FEA Guided Pane Visual State Scoping Slice 1

Scope completed:

- Tightened FEA agent visual-state projection so current render availability comes only from figure monitor slots that belong to the selected FEA run when a selected run is present.
- Prevented unrelated script/notebook figure monitors from being advertised to the FEA context and guided pane as the current FEA graphical state.
- Preserved generic runtime behavior: the runtime can still monitor arbitrary figures; only the FEA agent context projection filters to the relevant selected run.
- Added coverage for both cases: unrelated monitor slots do not mark FEA visual state current, while monitor slots from the selected FEA run do.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts src/app/components/agent/agent-panel-body.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final audit around guided-pane state fidelity. The pane now avoids one class of transient UI contamination, but deeper geometry-session selection state still needs true graphical-selection integration before the full Phase 7 acceptance flow can be claimed.

### 2026-07-10: Geometry Raw-Topology Retirement Regression Slice 1

Scope completed:

- Tightened model-visible geometry session compaction so `stats`, region samples, and mesh samples use curated summary fields instead of arbitrary shrunk runtime JSON.
- Added a public tool-boundary regression where the runtime returns a large STEP-like summary with raw topology, surface evaluators, CAD evaluators, raw source bytes, large assembly data, large node arrays, and raw mesh vertices/elements.
- Verified both `geometry_open_session` and `geometry_render` expose bounded snapshots without raw topology markers or oversized payloads.
- Removed the last non-test desktop `geometry_view` naming leftover by renaming the local agent helper/log event to `geometry_render`.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools geometry_session_tools_do_not_expose_raw_topology_payloads` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context -p ah-tools --tests` passed from `../runmat-private`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "geometry_render|geometry render|composed agent FEA flow" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `rg` over agent tools/context and desktop agent/runtime-provider surfaces finds `geometry_inspect`/`geometry_view` only in negative assertions.

Remaining:

- Continue final audit for any broader Phase 9 billing/cost regression opportunity, but the main raw-topology model-visible payload path now has production compaction and regression coverage.

### 2026-07-10: FEA Desktop Source-of-Truth Cleanup Slice 1

Scope completed:

- Removed the desktop-local FEA study document result normalizer and its schema-mirroring test.
- Updated FEA study surface and agent-state consumers to use the generated `runmat` `FeaStudyDocumentOperationResult` contract directly from the runtime client.
- Removed the desktop agent-runtime bridge's duplicated FEA study operation allowlist; the bridge now validates that an operation is present and leaves unsupported-operation rejection to the Rust-owned authoring runtime contract.
- Removed TS-side structural-profile readiness inference from the FEA review tree. Empty materials, boundaries, and loads no longer become required because of a desktop profile string heuristic; authoritative readiness remains the Rust-backed validation/readiness blockers.

Tests/evidence:

- `npm test -- src/app/components/fea/fea-study-view-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx src/app/components/agent/fea-agent-context.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/fea/fea-study-review-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue the final audit for smaller drift points. The remaining local physics/profile list is UI display copy for the review tree, not document parsing or readiness source of truth.

### 2026-07-10: FEA Fork Workspace Acceptance Flow Slice 1

Scope completed:

- Added harness-level composed coverage for the Phase 1 workspace/filesystem primitive acceptance path.
- The scripted model now copies an existing `.fea` file with `copy`, opens the copy with `open_path`, selects the copy with `select_path`, then continues FEA authoring by calling `finite_element_study_add_load_condition` against the copied path.
- The test uses the real filesystem tool over a temporary project root, a recording workspace interface, and the desktop runtime adapter test double, so it proves cross-tool composition rather than isolated tool behavior.
- Verified `open_path` and `select_path` are distinct UI focus operations: opening the copied file does not mark it selected, while selecting it does.
- Verified the original `.fea` source bytes are copied before typed study edits target the copied path.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness agent_can_fork_open_select_and_edit_copied_fea_study --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.

Remaining:

- Continue adding composed acceptance coverage for the later geometry-session and guided-pane paths where current evidence is still mostly unit/domain-level.

### 2026-07-10: Agent Runtime Legacy Bridge Removal Slice 1

Scope completed:

- Removed legacy FEA-specific run/check and one-shot geometry inspect hooks from the harness `RuntimeInterface`.
- Removed `FeaCheckStudy`, `FeaRunStudy`, and `GeometryInspect` from the desktop runtime adapter transport enum; agent-side execution now goes through generic `execute_file`, generic `check`, typed study operation, and session-oriented `geometry_render_view`.
- Removed the matching native Tauri agent runtime bridge aliases, so native and browser agent paths no longer retain hidden compatibility commands for `fea_check_study`, `fea_run_study`, or `geometry_inspect`.
- Moved the local CLI finite element validation implementation onto the generic `check` method and returned the generic preflight envelope with checker kind, phases, pass/safe status, blockers, warnings, diagnostics, and evidence refs.
- Updated runtime/tool/host test doubles to use generic `check` and removed obsolete transport cases.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-runtime-interface -p ah-runtime-adapter-desktop -p ah-tools -p ah-context -p ah-web-host --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.
- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.
- `rg` over the agent harness, desktop agent bridge, Tauri agent bridge, and runtime provider now finds no live `fea_check_study`, `fea_run_study`, or agent-bridge `geometry_inspect` implementation paths; only lower-level runtime inspect APIs and negative catalog assertions remain.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Notes:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-cli --tests` and `cargo check --manifest-path desktop/src-tauri/Cargo.toml` were interrupted after long silent compile/link phases. Neither produced a failure before interruption. The completed focused harness, provider, and typecheck coverage above are the evidence for this slice.

Remaining:

- Continue final audit for remaining low-level APIs that are still legitimate runtime/UI capabilities versus model-visible or agent-bridge legacy paths.

### 2026-07-10: Agent Runtime Bridge and Generated Contract Boundary Slice 1

Scope completed:

- Moved generic agent source execution out of `runtime-provider.tsx` and into `desktop/src/runtime/domain/agent-runtime-bridge.ts`.
- Added `executeAgentRuntimeSource` so replay workspace forking, runtime execution, response shaping, foreground notifications, and optional script history persistence now have a runtime-domain home.
- Kept the provider responsible for dependency wiring only: selected session state, workspace artifact reads, run callback, session/runtime stores, persistence hooks, and shell notifications.
- Added direct domain coverage for replay-forked source execution through the generic runtime bridge.
- Removed remaining local desktop duplicates of generated FEA capabilities/check contracts from `runtime/clients/client.ts`.
- Removed local desktop duplicates of generated geometry preview/inspection/scene contracts from `runtime/clients/client.ts`, leaving only the desktop-specific preview-session wrapper local.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx -t "agent runtime bridge|composed agent FEA flow|runtime bridge execution starts" --reporter=dot` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx --reporter=dot` passed from `../runmat-private/desktop`.

Remaining:

- Continue final audit for any other frontend-owned Rust/runtime payload contracts. The main client boundary now imports generated FEA and geometry payload types from `runmat` instead of shadowing them in desktop.

### 2026-07-10: Execute Tool Target-Specific Schema Slice 1

Scope completed:

- Extended the shared `runmat-tool-schema` representation to preserve root `oneOf` branches instead of dropping them during schema normalization.
- Updated strict model-tool normalization to recursively normalize and validate `oneOf` variants.
- Updated the generic `execute` tool schema so the model-visible contract has distinct inline and file branches: inline requires non-null `code`, file requires non-null `path`, and both require `target`.
- Reverted earlier raw FEA authoring-schema optional-field strictness now that strict normalization handles provider requirements without making runtime tool invocation cumbersome.
- Added regression coverage for strict `oneOf` normalization and the real runtime `execute` tool schema.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context strict_schema_preserves_and_normalizes_one_of_variants` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools runtime_tools_smoke` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context -p ah-tools --tests` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.

Remaining:

- Continue final audit for any execution-schema/provider compatibility edge cases. Runtime validation and model-visible schema now agree on inline/file target requirements.

### 2026-07-10: Agent FEA Execute Bridge Domain Split Slice 1

Scope completed:

- Moved agent-triggered `.fea` file execution semantics out of `runtime-provider.tsx` and into `desktop/src/runtime/domain/agent-runtime-bridge.ts`.
- Added `executeAgentFeaStudyFile` as the domain bridge for generic `execute({ target: "file", path })` dispatch when the target is a finite element study.
- Added `createAgentExecuteResultFromSession` so FEA execution responses are shaped from `ExecutionSession` in the runtime domain instead of duplicated inside the provider.
- Kept `runtime-provider.tsx` responsible for dependency wiring only: runtime client, trace/log refs, persistence filesystem hooks, session upsert, runtime execution lookup, and shell notifications.
- Added direct domain coverage proving `.fea` execution checks, runs, persists artifacts, emits runtime notifications, and returns normal agent execution response fields.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "composed agent FEA flow"` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx -t "agent runtime bridge|composed agent FEA flow"` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Remaining:

- Continue final audit for broad provider responsibilities. The `.fea` execute path now has a clearer runtime-domain home, but generic script execute response shaping still lives in `runtime-provider.tsx`.

### 2026-07-10: Harness-Level FEA Replay Inspection Flow Slice 1

Scope completed:

- Added harness-level coverage for inspecting a replayed finite element run through the generic runtime surfaces rather than FEA-only result tools.
- Scripted the model flow through `select_run`, `figures`, `show_figures`, `variables`, and `variable` against a replayed `.fea` run.
- Extended the desktop transport test double with FEA replay run metadata, artifact-backed result figure metadata, and lazy paged finite element field metadata.
- Fixed strict model-tool schema normalization at the root: nested objects and array item objects are now recursively normalized for strict provider requirements.
- Closed geometry session input schemas with `deny_unknown_fields` so generated tool schemas remain explicit and provider-valid.
- Made FEA authoring schemas explicit about nullable slots for strict model calls.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-context strict_schema_normalizes_nested_objects_and_array_items` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness fea_replay_inspection_uses_generic_runtime_surfaces --test host_parity` passed from `../runmat-private`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-harness --test host_parity` passed from `../runmat-private`.

Remaining:

- Continue final audit for remaining source-of-truth/module-boundary smells. The key newly covered path is that persisted FEA results can be selected and inspected like normal runtime runs without inline artifact blobs.

### 2026-07-10: FEA Study Runtime Result Boundary Slice 1

Scope completed:

- Added `desktop/src/runtime/domain/fea-study-runtime-result.ts` as the shared boundary adapter for Rust/WASM FEA study document operation results.
- Removed the loose `summary`-only cast from the FEA study view model; malformed Rust-owned study summaries now fail closed instead of fabricating UI state.
- Switched the FEA study surface and agent FEA overview projection onto the same runtime-domain normalizer.
- Kept TypeScript as a consumer/projection layer: durable `.fea` document semantics remain in Rust, while UI code works from normalized runtime contract data.
- Added direct tests for valid Rust-shaped payloads and malformed payload rejection.

Tests/evidence:

- `npm test -- src/runtime/domain/fea-study-runtime-result.spec.ts src/app/components/fea/fea-study-view-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx src/app/components/agent/fea-agent-context.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.

Remaining:

- Continue final audit for any remaining duplicated FEA source-of-truth parsing. The old TypeScript `fea-study-document` files are deleted; the remaining FEA UI files should stay presentation/model-projection only.

### 2026-07-10: Runtime Agent Bridge Domain Split Slice 1

Scope completed:

- Moved agent bridge payload/path normalization out of `runtime-provider.tsx` and into `desktop/src/runtime/domain/agent-runtime-bridge.ts`.
- Moved finite-element study document operation execution out of `runtime-provider.tsx`, including `.fea` path validation, source read, Rust/WASM-backed study operation dispatch, and write-back behavior.
- Moved geometry render-view request normalization and preview/image/disposal handling into the runtime domain bridge helper.
- Kept `runtime-provider.tsx` responsible for live client access, shell/runtime orchestration, and logging callbacks rather than owning FEA/geometry bridge semantics.
- Added direct domain tests for FEA study operation dispatch and bounded geometry render-view behavior.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "finite element|composed agent FEA|generic agent check"` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx -t "finite element|composed agent FEA|generic agent check|agent runtime bridge"` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Notes:

- A full `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx` run surfaced an existing broad provider-suite failure in `does not leak previous live workspace into a different selected file run` where the selected doc path was `/notebook.md` instead of `/basic.m`. The focused FEA bridge/provider tests for this slice passed.

Remaining:

- Continue final audit. `runtime-provider.tsx` is still broad overall, but the FEA/geometry agent bridge semantics now live in the runtime domain helper.

### 2026-07-10: FEA Study Surface Review Model Split Slice 1

Scope completed:

- Extracted FEA study review-tree construction, physics option projection, status aggregation, validation issue projection, setup-region highlights, and setup-region annotations out of `fea-study-surface.tsx`.
- Added `desktop/src/app/components/fea/fea-study-review-model.ts` as the focused presentation-model home for FEA review derivation.
- Kept `fea-study-surface.tsx` focused on runtime orchestration and rendering: document summary fetch, geometry preview lifecycle, scene export, canvas presentation, resize state, and table rendering.
- Reduced `fea-study-surface.tsx` from roughly 1,900 lines to roughly 1,060 lines without changing the Rust-owned `.fea` document source of truth.
- Added direct tests for the extracted review model so review-tree and region overlay derivation are covered outside the React surface.

Tests/evidence:

- `npm test -- src/app/components/fea/fea-study-review-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Continue final audit. The next high-signal boundary target remains `desktop/src/runtime/runtime-provider.tsx`, which still owns a broad bridge surface and should be checked for FEA-specific dispatch logic that belongs in `runtime/domain` helpers.

### 2026-07-10: Agent Pane And Legacy Geometry Surface Cleanup Slice 1

Scope completed:

- Moved FEA agent welcome/status interpretation, visual-state snapshot creation, selected-run snapshot creation, and runtime study-summary projection out of the generic `agent-panel.tsx` into focused FEA agent modules.
- Moved the collapsible FEA status/workflow panel out of `agent-panel.tsx`, leaving the generic agent pane to wire the FEA boundary component rather than own FEA workflow rendering internals.
- Removed the optional legacy agent-facing geometry tool exposure path for `geometry_inspect` and `geometry_view`.
- Removed the obsolete legacy-geometry exposure config flag and deleted the old tool metadata for those removed agent tools.
- Preserved the lower-level runtime geometry bridge used by session-oriented tools; the removed surface is only the old direct agent API.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-context -p ah-web-host --tests` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts` passed from `../runmat-private/desktop`.
- `git diff --check` passed in both `../runmat-private` and `runmat-analysis`.

Remaining:

- Continue the final completion audit, paying particular attention to remaining large UI/runtime files and whether they still own domain logic rather than projection/orchestration.

### 2026-07-10: Raw Topology Context Regression Slice 1

Scope completed:

- Tightened geometry summary compaction so `geometrySummary` and CAD summaries are allowlisted instead of generically shrinking arbitrary nested payloads.
- Preserved compact identity/count/status fields: geometry id, revision, units, source identity, source kind, material evidence samples, mesh samples, mapping summary, and CAD region status.
- Added regression coverage proving raw topology, evaluator payloads, control points, CAD evaluator arrays, raw bytes, and large assembly trees do not appear in model-visible compact geometry output.
- Strengthened FEA context attachment coverage to assert the current-state block does not include raw topology/evaluator keys.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.

Remaining:

- Run the final completion audit before marking the goal complete.

### 2026-07-10: Guided Pane Context Synchronization Slice 1

Scope completed:

- Added protocol-level `FeaVisualStateSnapshot` and `FeaSelectedRunSnapshot`.
- Carried FEA visual/run snapshots through desktop generated protocol types, desktop FEA context construction, the agent turn context, `ah-core` context mapping, and `ah-context` dynamic attachments.
- Extended the model-visible `fea_context` JSON so workflow state, graphical/render availability, and selected FEA run/result counts are synchronized in one authoritative current-state block.
- Updated the guided FEA panel to show compact graphical state and selected run status from the same context snapshot sent to the model.
- Kept the projection compact and host-owned: TypeScript summarizes existing runtime/pane state but does not own durable study/result contracts.

Tests/evidence:

- `cargo run --manifest-path agent-harness/Cargo.toml -p ah-protocol --example export_ts --features ts -- desktop/src/agent/generated/protocol.ts` passed.
- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-protocol -p ah-context --tests` passed.
- `cargo check --manifest-path agent-harness/Cargo.toml -p ah-core` passed.

Notes:

- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-protocol -p ah-context -p ah-core --tests` compiled but the broad `ah-core` behavioral tests failed in existing engine lifecycle/security/subagent cases unrelated to this context mapping slice; the focused protocol/context tests and `ah-core` compile check are the evidence for this change.

Remaining:

- Add or strengthen raw-topology/context regression coverage for Phase 9.
- Run the final completion audit before marking the goal complete.

### 2026-07-10: Geometry Session Tool Coverage Slice 1

Scope completed:

- Added model-visible session-oriented geometry tools for `geometry_set_visibility`, `geometry_section`, `geometry_measure`, and `geometry_find_features`.
- Kept the new operations in the harness `GeometrySession` boundary, so they mutate bounded current state and do not require raw topology in context.
- Extended `GeometrySession` snapshots with visibility and section state.
- `geometry_find_features` derives bounded setup-relevant feature candidates from compact region evidence.
- `geometry_measure` records targeted measurement requests against stable selectors/current selection and returns explicit bounded metadata instead of topology dumps.
- Updated model-visible metadata and FEA tool guidance so the model knows when to use camera, visibility, section, feature, measure, query, select, and region tools.
- Extended the runtime tool smoke path to prove the agent can open geometry, render, set camera, isolate, section, find a feature, select it, measure it, create a region, and carry that selector into typed `.fea` editing.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.

Remaining:

- Continue auditing guided pane/graphical synchronization against the target UX.
- Add stronger raw-topology/context regression coverage if the current bounded-output tests are not sufficient for Phase 9 completion.

### 2026-07-10: Composed Agent FEA Flow And Artifact Contract Slice 1

Scope completed:

- Added a runtime-provider bridge flow covering geometry evidence -> typed `.fea` study edits -> generic `check` -> `execute_file` solve kickoff.
- The flow keeps one in-memory `.fea` document across agent bridge calls, so check/execute observe the prior typed region, constraint, load, and output edits.
- Verified geometry region evidence can drive a study selector and that execute dispatches through the FEA orchestrator into a normal `fea-study` execution session.
- Added shared `runmat` FEA artifact schema-version constants and payload interfaces for run datasets, field descriptor artifacts, diagnostics artifacts, and object artifact metadata.
- Updated desktop FEA persistence to consume the shared artifact contract instead of inventing the dataset kind/schema payload shape in `fea-run-persistence.ts`.
- Regenerated the local `runmat` TS binding declarations consumed by desktop.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `cargo fmt --all` passed from `runmat-analysis`.
- `cargo test -p runmat-runtime fea_document --lib` passed from `runmat-analysis`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "composed agent FEA flow"` passed from `../runmat-private/desktop`.
- `npm test -- src/run/fea-run-persistence.spec.ts src/run/fea-run-manifest.spec.ts` passed from `../runmat-private/desktop`.

Remaining:

- Run the target-design/change-plan completion audit and either close the remaining gaps or mark the goal complete only if the evidence supports it.
- Keep any remaining TS-side FEA code limited to host integration, projections, and rendering rather than durable study/result contract authority.

### 2026-07-10: Guided FEA Workflow Snapshot Slice 1

Scope completed:

- Added protocol-level `FeaWorkflowSnapshot` and `FeaWorkflowChoiceSnapshot` so workflow state can travel with `FeaTurnContextSnapshot`.
- Regenerated desktop agent protocol types from `ah-protocol`.
- Derived workflow state in `fea-agent-context.ts` from selected path, active study summary, readiness, and runtime-provided FEA capabilities.
- Added workflow state to the model-visible FEA context attachment instead of hardcoded workflow prose inside `ah-context`.
- Replaced the expanded FEA status markdown drawer with a compact guided workflow surface showing current step, objective, next action, validation gates, completed steps, and prompt buttons.
- Kept workflow logic at the FEA context boundary; the panel now consumes the snapshot rather than owning study/setup semantics.

Tests/evidence:

- `cargo fmt --all --manifest-path agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path agent-harness/Cargo.toml -p ah-protocol -p ah-context --tests` passed.
- `npm test -- src/app/components/agent/fea-agent-context.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining:

- Add one true browser/agent-level flow over the web host/desktop worker using the workflow snapshot, typed study edits, `check`, and solve kickoff.
- Continue trimming legacy markdown-only FEA welcome content as the guided pane shell becomes the primary interaction surface.

### 2026-07-10: Runtime-Owned Study Outputs Slice 1

Scope completed:

- Added `AnalysisOutputRequest` and `AnalysisStudySpec.outputs` as runtime-owned output intent.
- Extended `.fea` loading to accept `outputs:` entries with `id`, `field`/`field_id`/`name`, `location`/`target`, and `kind`/`type`.
- Added the Rust document-authoring `set_outputs` operation, replacing the complete output list while preserving dotted field ids like `structural.displacement`.
- Exposed `set_outputs` through the public TS operation union, desktop bridge normalization, harness `finite_element_study_set_outputs` tool/schema, and FEA context metadata/guidance.
- Added desktop bridge and WASM session coverage for the typed output operation.

Tests/evidence:

- `cargo test -p runmat-runtime fea_document --lib` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.
- `npm run build:types` passed from `bindings/ts`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "dispatches finite element study document operations"` passed from `../runmat-private/desktop`.
- `env TMPDIR=/private/tmp wasm-pack test --node crates/runmat-wasm --test fea_document_operation` passed after regenerating the WASM builtin registry.

Remaining:

- Build the guided pane workflow state around selected path/study/run.
- Add browser/agent-level flow coverage for geometry selection -> typed study edits -> check -> run.

### 2026-07-10: Runtime-Owned Field Descriptor Contract Slice 1

Scope completed:

- Moved FEA field paging/storage metadata into `runmat-runtime` by adding `AnalysisFieldPagingDescriptor` and `AnalysisFieldStorageRef` directly to `AnalysisFieldDescriptor`.
- Re-exported FEA field paging defaults and artifact-kind constants from the runtime analysis API instead of leaving them as private desktop vocabulary.
- Updated the public `runmat` TS contract to expose FEA field descriptors, field materialization, run artifact constants, and the `feaField` handle method.
- Removed `desktop/src/run/fea-field-storage.ts`; desktop persistence now persists runtime-provided field descriptors instead of reconstructing paging/storage policy.
- Changed `fea-run-manifest.ts` to re-export FEA constants from `runmat` rather than defining them locally.
- Hardened desktop/replay/runtime bridge readers to accept both camelCase and snake_case descriptor keys while normalizing UI view models at the boundary.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `cargo test -p runmat-runtime fea_document --lib` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/run/fea-run-persistence.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/app/components/fea/fea-results-pane.spec.tsx src/replay/domain/fea-replay-artifacts.spec.ts src/replay/domain/replay-session-factory.spec.ts` passed from `../runmat-private/desktop`.
- `env TMPDIR=/private/tmp wasm-pack test --node crates/runmat-wasm --test fea_document_operation` passed.

Remaining:

- Wire artifact-kind constants all the way from Rust-generated bindings instead of manually mirroring them in the TS package source.
- Keep replay/persistence TS code limited to host integration and UI projection as the pane workflow grows.

### 2026-07-09: Runtime Unification Slice 1

Scope completed:

- Agent `execute_file` in the desktop bridge now dispatches `.fea` files to the existing finite element study orchestrator instead of reading the file and executing its contents as source.
- Agent-facing run summaries now include `run_kind`, so `select_run` is explicitly run-kind agnostic across script/notebook/FEA/REPL sessions.
- Agent-facing variable summaries now include stable identity fields: `variable_id`, `kind`, `run_id`, `session_id`, and `field_id`.
- Generic `variable` accepts `variable_id`, `name`, or `field_id`, with optional run/session scoping.
- Desktop runtime bridge exposes FEA result fields through generic `variables` and materializes them through generic `variable` by routing field-backed entries to `getFeaField`.
- Added `desktop/src/runtime/domain/agent-runtime-bridge.ts` so run/field identity and bridge payload shaping have a domain home instead of growing inside `runtime-provider.tsx`.

Tests/evidence:

- `cargo test -p ah-tools runtime_tools_smoke --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-adapter-direct -p ah-core -p ah-harness --tests` passed.
- `npm test -- runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-cli` was started but interrupted after a long silent link/build phase; no failure was observed before interruption.

Remaining from runtime unification:

- Persist FEA result figures as normal run figure artifacts with stable artifact ids.
- Define/persist the FEA run dataset manifest that relates study, geometry, mesh, fields, figures, diagnostics, logs, and reports.
- Broaden selector/materialization tests after dataset/figure artifacts are wired.

### 2026-07-09: Filesystem Primitive Slice 1 (`copy`)

Scope completed:

- Added a general `copy` tool to the harness filesystem tool set.
- The tool copies files only, refuses overwrite by default, supports explicit `overwrite`, and can create missing parent directories when `create_parent_dirs` is true.
- Added agent metadata for `copy` with clear source/destination/overwrite/create-parent semantics.
- Added test coverage for copying into a new parent directory, refusing accidental overwrite, explicit overwrite, and reading the copied file back.

Tests/evidence:

- `cargo test -p ah-tools filesystem_tools_mutate_within_scope --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.

Remaining from workspace primitive work:

- None for the first primitive pass. Follow-up hardening can add richer UI assertions if the editor surface gains a lighter integration test harness.

### 2026-07-09: Workspace Primitive Slice 1 (`open_path`, `select_path`)

Scope completed:

- Added `ah-workspace-interface` as the domain boundary for workspace/editor state mutations.
- Added general `open_path` and `select_path` tools to `ah-tools`; these are not FEA-specific and do not live under runtime.
- Registered workspace tools in the browser web host alongside project filesystem/search tools.
- Added a dedicated browser worker/main-thread workspace bridge, separate from filesystem and runtime bridges.
- Installed the desktop workspace bridge from `EditorProvider`, mapping `open_path` to pinned file selection and `select_path` to preview selection; directory selection stays non-editor-opening.
- Extracted the editor-side bridge decision logic into `editor-agent-workspace-bridge.ts` so behavior is testable without importing the full editor/runtime surface.

Tests/evidence:

- `cargo test -p ah-tools --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test -p ah-web-host --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test -p ah-context tools --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `npm test -- src/agent/clients/browser/index.spec.ts` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/shell/editor-panel.spec.ts` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from workspace primitives:

- Consider adding a full React integration assertion once the editor/shell test harness can mount the real `EditorProvider` without importing runtime wasm public assets.

### 2026-07-09: FEA Run Artifact Persistence Slice 1

Scope completed:

- Added `desktop/src/run/fea-run-persistence.ts` as the first FEA-specific persistence boundary that writes completed FEA runs into the normal run artifact layout.
- Completed FEA runs now get a v2 run manifest at `.artifacts/runs/<runId>.json` with `runKind: "fea-study"` and a `__fea__` cell.
- The FEA manifest cell now references content-addressed artifacts for a compact finite element dataset, workspace snapshot, field descriptors, diagnostics, artifact manifest, stdout/stderr, trace/log channels, and persisted figure scene/preview bytes when available.
- Both desktop Run-button execution and agent `execute_file` for `.fea` now pass persistence dependencies into the shared FEA orchestrator path instead of leaving agent runs metadata-only.
- `run-history` accepts/preserves `runKind` and FEA metadata, and generic output indexing can surface FEA workspace, dataset, and figure entries.
- The shared Rust `runmat-runtime-artifacts` manifest model now preserves `runKind` and `fea` metadata, so the Rust index/reader side can carry FEA run manifests without a separate artifact system.

Tests/evidence:

- `npm test -- src/run/fea-run-persistence.spec.ts src/run/fea-run-orchestrator.spec.ts src/run/run-history.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt --manifest-path ../runmat-private/shared/runmat-runtime-artifacts/Cargo.toml` completed.
- `cargo test --manifest-path ../runmat-private/shared/runmat-runtime-artifacts/Cargo.toml` passed.

Remaining from FEA artifact persistence:

- Harden large-field persistence around chunk/page descriptors and lazy materialization policy.
- Add end-to-end coverage for persisted FEA figures/fields after reload through the same run/variable/figure surfaces used by script and notebook runs.

### 2026-07-09: FEA Terminal Persistence And Replay Hydration Slice 1

Scope completed:

- FEA terminal sessions now persist through the same run manifest path whether they complete successfully, fail validation, or fail during runtime execution.
- Validation-failed `.fea` checks no longer remain metadata-only: they now write a failed `fea-study` run manifest with stdout, diagnostics, dataset metadata, and trace/log refs when available.
- Added `desktop/src/run/fea-run-manifest.ts` as the shared domain home for the FEA run cell id and replay metadata shaping, instead of having replay import persistence internals.
- Replay session creation now restores `runKind: "fea-study"` and attaches FEA manifest references for dataset, field descriptors, diagnostics, artifact manifests, and kept artifact refs.
- Replay hydration extracts FEA field identity from the persisted workspace artifact body before workspace normalization strips unknown entry metadata.
- Replay refresh fingerprints now include `runKind` and `fea` metadata so project replay indexing does not skip manifest-only FEA metadata changes.

Tests/evidence:

- `npm test -- src/run/fea-run-orchestrator.spec.ts src/replay/domain/replay-session-factory.spec.ts src/run/fea-run-persistence.spec.ts src/run/run-history.spec.ts` passed from `../runmat-private/desktop`.
- `npm test -- src/replay/session-refresh.spec.ts src/replay/domain/replay-session-factory.spec.ts src/replay/replay-orchestrator.spec.ts src/replay/use-project-replay-index.spec.tsx src/run/fea-run-orchestrator.spec.ts src/run/fea-run-persistence.spec.ts` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from terminal persistence/replay hydration:

- Hydrate the referenced FEA dataset/field descriptor artifacts themselves when the UI or agent needs them, without eager-loading large field payloads.
- Add end-to-end coverage for selecting a replayed FEA run and inspecting fields/figures through generic agent/runtime pane surfaces.

### 2026-07-09: FEA Field Paging And Generic Variable Slice 1

Scope completed:

- Added `desktop/src/run/fea-field-storage.ts` as the shared descriptor boundary for finite element field paging, value-count inference, page sizing, and default bounded materialization.
- Persisted FEA dataset and field descriptor artifacts now include explicit paging metadata: total scalar-component count, page size, page count, and default materialization limit.
- The persisted dataset payload now records a lazy field paging policy so large result fields are clearly treated as paged data instead of eagerly loaded blobs.
- Agent-facing variable summaries now expose paging/materialization metadata for FEA fields through the same generic `variables` surface used for runtime workspace values.
- The generic `variable` tool now accepts `offset` and `limit`; FEA field-backed variables use those selectors to request bounded field pages from the desktop runtime bridge.
- Replayed FEA workspaces now reattach `feaField` bindings from the raw persisted workspace artifact body after generic workspace normalization, so replayed fields still materialize through the generic variable path.
- Harness runtime contracts and tool metadata now include paging metadata plus `offset`/`limit` selector inputs, with descriptions that keep FEA fields aligned with the generic variable inspection model.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/run/fea-run-persistence.spec.ts src/replay/domain/replay-session-factory.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools -p ah-harness -p ah-cli --tests` passed.
- `cargo fmt` passed from `../runmat-private/agent-harness`.

Remaining from field paging/materialization:

- Hydrate referenced FEA dataset and field descriptor artifacts on demand for replay/runtime UI consumers instead of only carrying refs and compact descriptors.
- Add end-to-end coverage for selecting a replayed FEA run and inspecting a paged field through the agent pane and runtime variable surface.
- Decide how result field previews should present both numeric table pages and optional rendered field views without creating an FEA-only duplicate inspector.

### 2026-07-09: FEA Replay Descriptor Hydration Slice 1

Scope completed:

- Added `desktop/src/replay/domain/fea-replay-artifacts.ts` as the replay-domain boundary for hydrating compact FEA descriptor artifacts.
- Replay session loading now reads persisted FEA dataset and field descriptor JSON artifacts from both run-manifest scans and runs-index scans.
- Hydration recovers compact dataset metadata, result summary, diagnostics, artifact manifest, and merged field descriptors without loading field value payloads.
- `ExecutionSession.fea` now has explicit artifact-ref fields for dataset, field descriptors, diagnostics, artifact manifests, and run manifests instead of relying on casts.
- Live persisted sessions now use `fea.dataset` for the compact dataset payload and `fea.datasetManifest` for the artifact ref, matching replay sessions.
- Agent variable summaries now prefer hydrated field descriptors when summarizing FEA fields, so replayed workspace rows can stay compact while summaries still recover field class, dtype, shape, byte size, and page counts.

Tests/evidence:

- `npm test -- src/replay/domain/fea-replay-artifacts.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/run/fea-run-persistence.spec.ts src/replay/domain/replay-session-factory.spec.ts` passed from `../runmat-private/desktop`.
- `npm test -- src/replay/session-refresh.spec.ts src/replay/domain/replay-session-factory.spec.ts src/replay/domain/fea-replay-artifacts.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx src/run/fea-run-persistence.spec.ts src/run/fea-run-orchestrator.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from replay descriptor hydration:

- Add a browser/agent-level flow test for selecting a replayed FEA run and inspecting a paged field via the generic `variables`/`variable` tools.
- Unify FEA result figure inspection with the existing figure inventory/show surface, including stable artifact ids for replayed result figures.
- Decide the UI shape for result field previews that can show both numeric pages and optional rendered field views.

### 2026-07-09: FEA Figure Surface Unification Slice 1

Scope completed:

- Moved agent figure summary shaping into `desktop/src/runtime/domain/agent-runtime-bridge.ts`.
- Generic `figures` and `show_figures` results now include run id, run kind, live/replay representation, artifact figure id, and a `source` marker that identifies finite element result figures without introducing an FEA-only figure tool.
- Persisted/replayed FEA figures continue to use stable artifact-backed figure ids, while live figures retain handle-backed ids.
- The Rust runtime interface now accepts the enriched figure metadata so the harness can carry it through to tool results and context images.
- Cleaned up `figures`/`show_figures` tool metadata so the model is told to use `select_run` for a different run and to pass stable ids from `figures` into `show_figures`.

Tests/evidence:

- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-interface -p ah-tools -p ah-context --tests` passed.
- `cargo fmt` passed from `../runmat-private/agent-harness`.

Remaining from figure surface unification:

- Add an end-to-end agent flow test that selects a replayed FEA run, lists figures, shows a persisted result figure, lists variables, and materializes a paged field.
- Decide how the runtime pane presents finite element field previews when both numeric pages and rendered field views are available.

### 2026-07-09: FEA Replay Generic Inspection Coverage Slice 1

Scope completed:

- Added desktop bridge coverage for a replayed FEA run using only generic agent/runtime commands.
- The test exercises `figures`, `show_figures`, `variables`, and `variable` against a replayed finite element session.
- Verified replayed FEA result figures expose stable artifact-backed figure ids, run id, run kind, replay representation, and finite element result source through the generic figure surface.
- Verified `show_figures` can request a replayed FEA result figure by stable id and produce a monitor image when a live/replayed handle is available.
- Verified replayed FEA fields use hydrated descriptors for class, shape, and total count, and that `variable` passes explicit `offset`/`limit` through to bounded field materialization.

Tests/evidence:

- `npm test -- src/runtime/runtime-provider.spec.tsx src/runtime/domain/agent-runtime-bridge.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from generic inspection coverage:

- Add an actual harness-level/browser-host flow once a representative `.fea` replay artifact fixture exists.
- Move from inspection plumbing into typed study checking/editing and guided FEA pane state transitions.

### 2026-07-09: Generic Check Slice 1

Scope completed:

- Added a generic `check` runtime tool and runtime-interface contract across the agent harness.
- Removed model-visible `fea_check` tool registration, metadata, and context guidance; FEA static validation now routes through the generic `check` surface.
- Added desktop/web bridge support for a generic `check` command.
- Desktop `.fea` checks now dispatch to the finite element validator and normalize raw `FeaCheckResult` payloads into a generic preflight result with checker kind, artifact kind, phases, pass/safe status, blockers, warnings, diagnostics, evidence artifact refs, and raw details.
- Kept lower-level `fea_check_study` as an internal adapter/runtime method while existing desktop runtime bridge code migrates; it is no longer the model-facing tool.

Tests/evidence:

- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-interface -p ah-runtime-adapter-desktop -p ah-tools -p ah-context -p ah-web-host -p ah-harness --tests` passed.
- `npm test -- src/runtime/domain/agent-runtime-bridge.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt` passed from `../runmat-private/agent-harness`.

Remaining from generic check:

- Add static check dispatch for `.m`/notebook files once a non-executing checker exists.
- Start typed `.fea` document summary/create/edit tools.
- Retire low-density geometry tools from the model-visible catalog after session-aware geometry tools land.

### 2026-07-09: Typed FEA Study Operations Slice 1

Scope completed:

- Added an explicit model-visible typed study tool set:
  - `finite_element_study_get_summary`
  - `finite_element_study_create`
  - `finite_element_study_add_region`
  - `finite_element_study_add_material`
  - `finite_element_study_assign_material`
  - `finite_element_study_add_constraint`
  - `finite_element_study_add_load_condition`
  - `finite_element_study_set_load_condition`
- Added one internal runtime bridge dispatch, `finite_element_study_operation`, so model-visible tools stay explicit without duplicating transport plumbing.
- Desktop bridge now reads/writes `.fea` files through the runtime filesystem and returns compact study summaries, counts, and changed-section diffs.
- Extended the existing `.fea` document helper into a typed operation boundary for create, summary, region, material, material assignment, constraint, and load-condition edits.
- Removed `fea_run` from model-visible tool registration and prompt guidance; `.fea` execution should go through generic `execute({ target: "file", path })`.
- Updated FEA prompt guidance to use typed `finite_element_study_*` edits, generic `check`, and generic `execute`.
- Did not add `finite_element_study_set_outputs` yet because the current Rust `.fea` loader does not accept a top-level `outputs` block; exposing that tool now would write invalid study files.

Tests/evidence:

- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-runtime-interface -p ah-runtime-adapter-desktop -p ah-tools -p ah-context -p ah-web-host -p ah-harness --tests` passed.
- `npm test -- src/app/components/fea/fea-study-document.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt` passed from `../runmat-private/agent-harness`.

Remaining from typed study operations:

- Add update/remove operations for regions, materials, assignments, constraints, mesh settings, and outputs after the runtime loader accepts the corresponding document sections.
- Add a full agent flow test: copy or create `.fea`, select it, apply typed edits, run `check`, then `execute`.
- Replace old geometry tools with session-oriented geometry tools before relying on region creation from CAD selections.

### 2026-07-10: FEA Study Document Domain Split Slice 1

Scope completed:

- Moved the typed FEA study document layer out of `desktop/src/app/components/fea` into `desktop/src/fea/study-document`.
- Split the old UI-adjacent document helper into focused domain files:
  - `types.ts` for summaries, editor model rows, operation results, and typed operation payloads.
  - `yaml.ts` for bounded YAML scanning/quoting helpers used by the document operations.
  - `source.ts` for new `.fea` source generation.
  - `parser.ts` for summaries, editor model parsing, readiness checks, selector resolution, and path resolution.
  - `operations.ts` for typed study document mutations and diff reporting.
  - `index.ts` for the public domain-module export.
- Updated the FEA surface, agent context builder, editor panel, and runtime provider to import from `@/fea/study-document`.
- Moved the study document tests to the new domain module and removed the old component-level helper/spec.

Tests/evidence:

- `npm test -- src/fea/study-document/study-document.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from the domain split:

- Continue shrinking UI/runtime files where FEA-specific orchestration still sits in broad providers.
- Move the remaining TS parser/readiness display helpers to Rust-provided summaries so UI context and static readiness do not duplicate `.fea` schema semantics.
- Add update/remove/mesh/output operations only in the Rust document authoring module where the current `.fea` loader and schema can accept the written document shape.
- Start the geometry session tool slice so region creation can be driven from model-controlled geometry state rather than raw one-shot inspection tools.

### 2026-07-10: Rust-Owned FEA Study Document Authoring Slice 1

Scope completed:

- Added `crates/runmat-runtime/src/analysis/fea_document_authoring.rs` as the Rust source of truth for durable `.fea` document creation, summary, and typed mutation.
- Implemented the already model-visible operation set in Rust:
  - `create`
  - `get_summary`
  - `add_region`
  - `add_material`
  - `assign_material`
  - `add_constraint`
  - `add_load_condition`
  - `set_load_condition`
- Exposed the Rust operation through the WASM session as `applyFeaStudyDocumentOperation`.
- Routed the browser runtime worker/client and desktop runtime-provider agent bridge through the Rust operation. The bridge still performs file read/write, but no longer owns `.fea` parse/create/mutate semantics.
- Routed editor-panel “create .fea from geometry” through the runtime operation instead of TS source generation.
- Removed TS `.fea` source creation and mutation helpers from `desktop/src/fea/study-document`; the remaining TS module is now limited to current UI display parsing/readiness until that gets replaced by Rust summaries.

Tests/evidence:

- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.
- `cargo test -p runmat-wasm --lib apply_fea_study_document_operation_js` passed.
- `npm test -- src/fea/study-document/study-document.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path Cargo.toml` passed.

Remaining from Rust-owned document authoring:

- Replace TS UI parser/readiness helpers with Rust-produced study summaries and readiness/check output.
- Generate or share TS-facing operation/result types from Rust contracts instead of hand-maintaining loose `unknown`/local unions.
- Add schema-backed update/remove/mesh/output operations in Rust only, then expose model-visible tools from that contract.

### 2026-07-10: Rust-Owned FEA Study Summary/Readiness Slice 1

Scope completed:

- Rust `get_summary` now returns the semantic study readiness block alongside the parsed `.fea` summary, rows, counts, and diff data.
- Agent context hydration now consumes Rust-provided readiness instead of recalculating readiness from counts in `AgentPanel`.
- `FeaStudySurface` no longer parses `.fea` YAML in TypeScript. It requests the Rust document summary through `applyFeaStudyDocumentOperation("get_summary", ...)` and projects that result into the existing review tree model.
- Removed the TS YAML/parser/readiness helpers from `desktop/src/fea/study-document`; that module now only contains TS-facing types plus view-level path/region helpers.
- FEA surface validation still augments Rust semantic readiness with the existing check/schema result state, so schema/check failures remain visible without duplicating `.fea` semantic readiness.

Tests/evidence:

- `cargo fmt --all --manifest-path Cargo.toml` passed.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/fea/study-document/study-document.spec.ts src/app/components/agent/fea-agent-context.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.
- `cargo test -p runmat-wasm --lib apply_fea_study_document_operation_js` completed but matched zero tests, so it did not validate the WASM binding behavior.

Remaining from Rust-owned summary/readiness:

- Add a real WASM/session-level test for `applyFeaStudyDocumentOperation`.
- Generate or share TS-facing operation/result types from Rust contracts instead of hand-maintaining the browser bridge types.
- Add schema-backed update/remove/mesh/output operations in Rust only, then expose model-visible tools from that contract.
- Consider moving the remaining view-only structural-profile section-status hint into an explicit Rust summary field if the UI starts needing more profile-specific display policy.

### 2026-07-10: Geometry Session Tool Foundation Slice 1

Scope completed:

- Added `ah-tools/src/tools/geometry.rs` as the model-facing geometry session tool boundary instead of continuing to grow geometry logic inside the broad runtime tool module.
- Added normal FEA-mode geometry session tools:
  - `geometry_open_session`
  - `geometry_get_state`
  - `geometry_set_camera`
  - `geometry_render`
  - `geometry_close_session`
- Geometry sessions keep bounded current state in the harness tool layer and use existing runtime preview/render methods as lower-level adapters.
- `geometry_render` attaches model-visible images from the current session state while keeping the JSON payload compact.
- Geometry summaries now bound region arrays, diagnostic arrays, nested JSON depth, array lengths, and long strings before entering tool output/context.
- Removed the old `geometry_inspect` / `geometry_view` tool implementation block from the broad runtime tool module; a later cleanup slice removed the remaining optional legacy registration path entirely.
- Updated FEA tool guidance and FEA context attachments to direct the model toward geometry sessions instead of raw inspect/view calls.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-web-host --tests` passed.

Remaining from geometry session foundation:

- Add browser/desktop flow coverage that opens a real geometry session, renders an image, changes camera, creates a region, and edits a `.fea` file.
- Retire legacy geometry bridge operations entirely once session tools no longer need the old lower-level adapter names.

### 2026-07-10: Geometry Session Selection And Region Slice 1

Scope completed:

- Added bounded geometry-session region candidate querying with `geometry_query`.
- Added explicit current selection state with `geometry_select` and `geometry_clear_selection`.
- Added `geometry_create_region`, which creates a named session region from the current selection or explicit selector.
- `geometry_create_region` returns `finite_element_study_add_region_input`, making the intended handoff to typed `.fea` study editing explicit without combining UI selection and file mutation into one opaque tool.
- Geometry session snapshots now include current selection and created regions.
- FEA tool guidance now tells the model to use `geometry_query`, `geometry_select`, and `geometry_create_region` before writing geometry-bound study regions.

Tests/evidence:

- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-web-host --tests` passed.

Remaining from geometry session selection/region:

- Add real browser/desktop flow coverage for session render -> select/create region -> `finite_element_study_add_region`.
- Add measured/probed selection helpers once the graphical surface exposes hit-testing or picked region ids to the harness.
- Retire the lower-level legacy geometry bridge operation names after the desktop/runtime bridge exposes first-class session operations instead of adapter calls.

### 2026-07-10: FEA Study Document TS Source-Of-Truth Removal Slice 1

Scope completed:

- Removed the misleading `desktop/src/fea/study-document` module entirely; TypeScript no longer has a folder that looks like the source of truth for `.fea` document semantics.
- Moved FEA study summary operation/result DTOs into the runtime client boundary where they describe the Rust/WASM bridge payload.
- Moved view-only study projection helpers into `desktop/src/app/components/fea/fea-study-view-model.ts`, next to the FEA surface that consumes them.
- Kept path/region selector helpers as display/preview helpers only; create/summary/mutation/readiness remain Rust-owned through `applyFeaStudyDocumentOperation`.
- Updated FEA surface tests to mock the Rust runtime summary boundary explicitly instead of relying on hidden TS parsing assumptions.

Tests/evidence:

- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/app/components/fea/fea-study-view-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.

Remaining from TS source-of-truth removal:

- Generate or share TS-facing operation/result types from Rust contracts instead of hand-maintaining the browser bridge DTOs.
- Move any remaining profile-specific display policy into explicit Rust summary fields if the UI needs more than generic view projection.

### 2026-07-10: WASM FEA Document Operation Coverage Slice 1

Scope completed:

- Added `crates/runmat-wasm/tests/fea_document_operation.rs` as a real wasm32 session-level test for `applyFeaStudyDocumentOperation`.
- The test initializes the WASM runtime, creates a `.fea` study document through the exported session method, then summarizes the generated source through the same method.
- Fixed `scripts/regenerate-wasm-registry.sh` to generate the checked-in registry for the actual default `runmat-wasm` runtime feature set (`plot-web`) instead of stamping `occt-wasm-host`, which made normal wasm tests fail registry validation.
- Regenerated `crates/runmat-runtime/src/builtins/generated_wasm_registry.rs` for the corrected default wasm feature set.

Tests/evidence:

- `scripts/regenerate-wasm-registry.sh` completed and wrote 1506 registry entries / 665 builtins.
- `wasm-pack test --node crates/runmat-wasm --test fea_document_operation` passed: 1 wasm test passed.
- `cargo fmt --all --manifest-path Cargo.toml` passed.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.

Remaining from WASM document operation coverage:

- Add generated/shared TS-facing contracts for the Rust-owned operation payloads.
- Add schema-backed update/remove/mesh/output operations in Rust, then expose them through the typed tool set once the `.fea` schema accepts those sections.

### 2026-07-10: FEA Study Document Shared Contract Slice 1

Scope completed:

- Added a typed Rust-owned `.fea` study document operation result contract in `runmat-runtime` instead of returning anonymous JSON from document authoring.
- Exported the matching TypeScript-facing FEA study document operation/result types from the public `runmat` TS package.
- Added `applyFeaStudyDocumentOperation(...)` to the public TS session handle and web session wrapper so desktop bridge types consume the public runtime contract instead of hand-maintained local DTOs.
- Switched the desktop runtime client boundary to import/re-export FEA study document contract types from `runmat`.
- Kept `desktop/src/app/components/fea/fea-study-view-model.ts` as view projection only; it no longer owns the `.fea` document contract.
- Normalized WASM builtin registry generation so both shell and Node generators stamp the default web helper registry, while the runtime build fingerprint ignores `occt-wasm-host` because that feature changes geometry host wiring rather than builtin helper registration.

Tests/evidence:

- `npm run build:types` passed from `bindings/ts`.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `cargo fmt --all --manifest-path Cargo.toml` passed.
- `node scripts/regenerate-wasm-registry.mjs` completed and wrote 1506 registry entries / 665 builtins.
- `cargo check -p runmat-wasm --target wasm32-unknown-unknown --features occt-wasm-host` passed.
- `env TMPDIR=/private/tmp wasm-pack test --node crates/runmat-wasm --test fea_document_operation` passed.
- `npm test -- src/app/components/fea/fea-study-view-model.spec.ts src/app/components/fea/fea-study-surface.spec.tsx src/app/components/agent/fea-agent-context.spec.ts src/runtime/runtime-provider.spec.tsx` passed from `../runmat-private/desktop`.

Remaining from shared contract work:

- Add first-class runtime study outputs before exposing output editing; current `.fea` summary can display outputs, but `AnalysisStudySpec` does not own them yet.
- Add browser/desktop flow coverage that opens geometry, creates a region from session state, and writes that region into a `.fea` study.
- Shape the model-guided FEA pane around durable selected path/study/run state instead of transient UI assumptions.

### 2026-07-10: FEA Study Edit Operations Slice 1

Scope completed:

- Added Rust-owned update/remove operations for supported `.fea` document sections:
  - `update_region`
  - `remove_region`
  - `update_material`
  - `update_constraint`
  - `remove_constraint`
  - `remove_load_condition`
- Added Rust-owned `set_mesh` authoring for the existing loader-backed `mesh` block.
- Tightened add/update semantics in the authoring layer: add operations now fail on duplicate ids, update operations require existing ids, and set-load remains the explicit upsert operation.
- Exposed the new operation names through the public `runmat` TS contract.
- Exposed model-visible agent tools:
  - `finite_element_study_update_region`
  - `finite_element_study_remove_region`
  - `finite_element_study_update_material`
  - `finite_element_study_update_constraint`
  - `finite_element_study_remove_constraint`
  - `finite_element_study_remove_load_condition`
  - `finite_element_study_set_mesh`
- Updated agent tool metadata and FEA guidance so the model sees clear distinctions between add, update, remove, set-load, and set-mesh.
- Did not expose `finite_element_study_set_outputs`: the current `.fea` summarizer can display an `outputs` block, but the runtime `AnalysisStudySpec` has no first-class outputs field yet. Writing output settings before the runtime owns them would preserve a bad split.

Tests/evidence:

- `cargo fmt --all --manifest-path Cargo.toml` passed.
- `cargo fmt --all --manifest-path ../runmat-private/agent-harness/Cargo.toml` passed.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.
- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.
- `npm run build:types` passed from `bindings/ts`.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `env TMPDIR=/private/tmp wasm-pack test --node crates/runmat-wasm --test fea_document_operation` passed.

Remaining from study edit operations:

- Add first-class output request/selection semantics to the runtime study contract if output editing is required for the guided workflow.
- Add end-to-end browser/desktop coverage that uses geometry session selection to create a region, writes it through typed study tools, runs `check`, then proceeds to mesh/execute.

### 2026-07-10: Geometry-To-Study Bridge Coverage Slice 1

Scope completed:

- Added composed `ah-tools` coverage for `geometry_open` -> render/query/select/create-region -> `finite_element_study_add_region` using the returned `finite_element_study_add_region_input`.
- Fixed the desktop runtime bridge FEA operation normalizer to admit the Rust/public-TS operation set now exposed to the harness: update/remove operations and `set_mesh`.
- Added desktop bridge coverage for `finite_element_study_operation` with `set_mesh`, proving runtime filesystem read/write still goes through the Rust-owned `.fea` operation boundary.
- Added desktop bridge coverage for `geometry_render_view`, proving the agent bridge can use `previewGeometry` plus `renderGeometrySceneImage` and return a model-visible image payload.

Tests/evidence:

- `cargo test --manifest-path ../runmat-private/agent-harness/Cargo.toml -p ah-tools -p ah-context --tests` passed.
- `cargo test -p runmat-runtime fea_document_authoring --lib` passed.
- `npm run typecheck` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "dispatches finite element study document operations|renders geometry views"` passed from `../runmat-private/desktop`.
- `npm test -- src/runtime/runtime-provider.spec.tsx -t "does not leak previous live workspace"` passed from `../runmat-private/desktop` after a full-suite order/timing failure on that unrelated existing test.

Remaining from bridge coverage:

- Add one true browser/agent-level flow over the web host/desktop worker once the guided pane state is shaped enough to make that test representative.
- Keep replacing lower-level bridge assumptions with runtime-owned session/study contracts as they become visible.

### 2026-07-10: FEA Agent Capability Ownership Slice 1

Scope completed:

- Removed hardcoded FEA path classification ownership from the agent pane by routing `buildFeaAgentContext` and `isFeaAgentPath` through runtime-provided FEA capabilities.
- The pane now fetches `getFeaCapabilities()` and uses its supported document/geometry extensions for study detection, geometry detection, context snapshots, and automatic FEA/general mode selection.
- Kept a small fallback capability set only for startup and unavailable-runtime states; runtime capabilities win once loaded.
- Added coverage showing runtime-provided extensions drive classification and model context, including an extension outside the fallback list.

Tests/evidence:

- `npm test -- src/app/components/agent/fea-agent-context.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from capability ownership:

- Continue pushing pane workflow state toward runtime-owned selected path/study/run context rather than UI-only inference.

### 2026-07-10: FEA Run Artifact Contract Ownership Slice 1

Scope completed:

- Promoted FEA run-manifest metadata from untyped JSON in `runmat-runtime-artifacts` to a typed shared Rust contract.
- Added shared Rust constants for FEA run kind, FEA cell id, manifest metadata schema version, and FEA artifact kinds.
- Added shared Rust coverage proving the FEA metadata serializes to the same camelCase manifest shape the desktop reads/writes.
- Centralized desktop FEA run identity, artifact kind names, artifact ref ids, and metadata construction in `desktop/src/run/fea-run-manifest.ts`.
- Updated FEA persistence, replay hydration, replay session creation, generic agent bridge run-kind branching, and run orchestration telemetry to use the centralized FEA run/artifact contract instead of local schema literals.
- Added focused desktop tests for the FEA run-manifest helper.

Tests/evidence:

- `cargo fmt --manifest-path ../runmat-private/shared/runmat-runtime-artifacts/Cargo.toml` passed.
- `cargo test --manifest-path ../runmat-private/shared/runmat-runtime-artifacts/Cargo.toml` passed.
- `npm test -- src/run/fea-run-manifest.spec.ts src/run/fea-run-persistence.spec.ts src/replay/domain/replay-session-factory.spec.ts src/replay/domain/fea-replay-artifacts.spec.ts src/runtime/domain/agent-runtime-bridge.spec.ts` passed from `../runmat-private/desktop`.
- `npm run typecheck` passed from `../runmat-private/desktop`.

Remaining from artifact contract ownership:

- Promote the nested FEA dataset and field-descriptor payload schemas themselves when they become runtime-query API contracts rather than desktop packaging payloads.
- Keep FEA result querying on generic variables/figures/dataset APIs; avoid adding FEA-specific result tools unless a proven access pattern cannot fit the generic surfaces.

## Next Slice

Continue toward the target FEA workflow:

- Treat shared scene/context selection attachments as a named completion gate:
  audit existing STEP/CAD, `.fea`, mesh, and result selection sources; fill any
  missing product-to-chip or chip-to-model gaps; and add composed acceptance for
  selection chip -> typed study edit -> generic `check`/`execute`.
- Add stronger raw-topology/context regression coverage if the current bounded-output tests do not prove Phase 9.
- Run the requirement-by-requirement completion audit against `FEA_GEOMETRY_AGENT_CHANGE_PLAN.md`, `FEA_GEOMETRY_AGENT_TARGET_DESIGN.md`, `FEA_GEOMETRY_AGENT_USER_EXPERIENCE.md`, and `FEA_GEOMETRY_AGENT_CONTEXT_LAYOUT.md`.
- Only mark complete after the audit proves fork/resume, typed authoring, generic run/result inspection, guided pane state, and bounded context are all covered by implementation and tests.
