# Next Steps

## Direction

We are reorienting away from incremental VM stack-shape patches and toward making the semantic HIR + MIR own more of the program shape before bytecode lowering.

The recent ratchets are valuable coverage, but several fixes exposed the same architectural issue: VM dispatch is inferring assignment/indexing semantics from stack layout and runtime `Value` shapes. That is brittle and makes fallback removal harder.

Going forward, prefer changes that make semantic intent explicit in MIR/bytecode over adding new runtime heuristics.

## Principles

- Keep the current ratchet tests as safety coverage.
- Avoid new `StoreIndex` / `StoreSlice` stack-sniffing branches unless they unblock a critical regression.
- Move index and assignment classification earlier, ideally into MIR lowering.
- Keep runtime helpers focused on executing explicit plans, not discovering source semantics.
- Replace compiler-internal builtin string calls with typed VM/runtime ABI operations.

## Indexing And Assignment Plan

### 1. Classify Assignment Places In MIR

MIR should distinguish these assignment forms explicitly:

- scalar paren assignment, e.g. `A(2) = 9`
- vector/range/logical paren assignment, e.g. `A(2:3) = 9`, `A(mask) = 9`
- deletion through empty RHS, e.g. `A(2:3) = []`
- cell paren replacement, e.g. `C(2) = {9}`
- cell paren deletion, e.g. `C(2:3) = []`
- brace content assignment, e.g. `C{2} = 9`
- member/indexed store-back, e.g. `s.a(2) = 9`, `C{1}.x = 9`

The VM should not have to infer these categories from stack layout.

### 2. Normalize Slice Lowering

Cases like `A(2:3) = ...` should lower as slice assignment intentionally, not as `StoreIndex(1)` with a tensor-valued index that runtime later reinterprets.

Target outcome:

- scalar numeric index lowers to scalar store
- colon/range/logical/vector indices lower to slice store
- empty RHS is encoded as deletion semantics, not just a generic empty tensor that runtime guesses from

After this, simplify or remove the recent `StoreIndex` branches that detect non-scalar index values.

### 3. Centralize Cell Indexing Semantics

Create one cell indexing/assignment plan API covering:

- MATLAB column-major linear indexing
- subscript indexing
- colon selection
- range/vector/logical selection
- comma-list expansion order
- paren replacement versus brace content assignment
- deletion
- RHS scalar expansion and shape checks
- GC write barriers

Then replace direct `ca.data[...]` indexing outside that API.

This should let VM dispatch call a small number of explicit cell operations instead of maintaining parallel logic in `StoreIndex`, `StoreSlice`, call expansion, display, and object helper paths.

### 4. Simplify Bytecode Dispatch

Once MIR lowering emits precise operation kinds:

- `StoreIndex` should handle scalar assignment only.
- `StoreSlice` should handle planned selections.
- cell assignment should route through the centralized cell plan path.
- object/handle `subsref` and `subsasgn` should receive structured index descriptors.

The dispatch layer should execute known operations, not classify source syntax.

## Hard-Coded Internal Builtins

There are compiler/VM paths that encode runtime behavior through hard-coded builtin names. Some names represent public builtin calls and are fine at the public dispatch boundary. Others are compiler-internal services and should become typed VM/runtime ABI operations.

### Known Internal String Hooks

- `make_handle`
  - No current production references remain in VM/HIR lowering.
  - Function-handle lowering now emits typed `CreateFunctionHandle` instructions.

- `feval`
  - Semantic lowering and bytecode use typed dynamic-call instructions, and VM runtime forwarding now calls `runmat_runtime::call_feval_async` instead of owning the builtin string.
  - MIR lowering now asks `HirCallableRef::is_feval_builtin_like()` instead of owning an ad hoc `name == "feval"` branch; the remaining legacy VM expression branch uses the shared `FEVAL_BUILTIN_NAME` classifier constant.

- `call_method`
  - VM object plumbing now forwards through `runmat_runtime::call_method_async` instead of owning the builtin string.
  - Prefer typed object dispatch operations.

- `__make_cell`
  - No current production references remain in VM/runtime/builtins; index packaging now uses direct `CellArray` construction helpers.
  - Prefer structured index descriptors.

- object protocol strings: `"subsref"`, `"subsasgn"`, `"()"`, `"{}"`, `"."`
  - These are MATLAB protocol concepts, but the VM should not manually assemble ad hoc argument lists for them.
  - Member dispatch now routes the `"."` selector through `ObjectIndexKind::Member`, matching paren/brace selector classification.
  - The remaining object `subsasgn` paren fallback now gets the `"()"` selector from `ObjectIndexKind::Paren` instead of assembling the string inline.
  - Member fallback method checks now get `subsref` / `subsasgn` names from `ObjectIndexOp` instead of inline strings.
  - VM object `subsasgn` fallback qualified-name construction now uses `ObjectIndexOp::Subsasgn.protocol_name()`.
  - Legacy expression lowering checks dynamic member indexing against `PAREN_SELECTOR_NAME` instead of an inline `"()"` allocation.
  - Paren, brace, and member selectors are centralized as `PAREN_SELECTOR_NAME`, `BRACE_SELECTOR_NAME`, and `MEMBER_SELECTOR_NAME`.
  - Runtime object protocol handlers now use `OBJECT_INDEX_PAREN`, `OBJECT_INDEX_BRACE`, and `OBJECT_INDEX_MEMBER` for selector matching.
  - Runtime object protocol dispatch/class registration now uses `OBJECT_SUBSREF_METHOD` and `OBJECT_SUBSASGN_METHOD` outside macro attributes.
  - Prefer `ObjectSubsref` / `ObjectSubsasgn` style operations carrying structured index data.

- `classref`
  - Legacy VM class-reference lowering now uses `CLASS_REF_CONSTRUCTOR_NAME` instead of repeating the sentinel string in compiler pattern matches.
  - Prefer a typed class-reference HIR/MIR form over source-call pattern matching.

- `nargin` / `nargout`
  - VM call-count handling now classifies these through a VM intrinsic counter call category instead of inline string branches.

- `rethrow`
  - VM implicit rethrow handling now classifies this through a VM intrinsic exception call category instead of an inline string branch.

- `await` / `spawn`
  - Semantic HIR lowering now uses shared `AWAIT_EXTENSION_NAME` and `SPAWN_EXTENSION_NAME` constants for RunMat extension call classification.

- `__register_test_classes`
  - Legacy HIR lowering now uses `TEST_CLASS_REGISTRATION_BUILTIN_NAME` for the test-support static-class registration hook.

- `~`
  - Semantic and legacy HIR multi-assign lowering now use `DISCARD_OUTPUT_NAME` for discard-output classification.

### Replacement Strategy

1. Inventory all hard-coded builtin/runtime names in compiler and VM dispatch.
2. Classify each as one of:
   - public user-facing builtin call
   - compiler/runtime internal service
   - object protocol operation
3. Replace internal services with typed instructions or ABI calls.
4. Keep public builtin calls string/name-based only at the builtin dispatch boundary, or migrate to builtin IDs where available.
5. Add ratchet tests or bytecode assertions that semantic lowering no longer emits internal string-builtin call patterns for these cases.

## Recommended Immediate Slice

The fast semantic-call cleanup slices are complete. The next changes should avoid adding VM heuristics and instead close one of the remaining designed ABI gaps.

Concrete next targets:

- Continue widening Turbine value-lane coverage beyond semantic cell expansion; semantic expanded calls can now cross the JIT boundary with tagged variables and `TurbineArgSpec`, while generic `feval`/external expansion and object expansion still need semantic descriptor work.
- Treat unresolved/external callback identities that are not present in a semantic registry as unresolved calls; the VM no longer recompiles legacy HIR as a dynamic fallback.
- Continue moving object/index call sites to structured descriptors; VM object protocol dispatch now has an `ObjectIndexDescriptor` serialization boundary instead of ad hoc method argument assembly in the public helper functions.

Avoid spending more time on scalar/range assignment band-aids: semantic MIR/bytecode now lowers common range, vector, logical, cell, and member store-back cases through typed slice/index instructions, and `StoreIndex` is narrowed to scalar indices.

## Semantic Compiler Design Work

The new semantic compiler is now carrying enough source intent that the remaining work should be designed as compiler-product cleanup, not as VM fallback patching. The main theme is to stop treating legacy HIR as the durable intermediate representation and make semantic HIR, MIR, layout, and bytecode the only compiler product the runtime needs.

### 1. Semantic Function Callback ABI

Current state:

- Top-level semantic compilation produces semantic function bytecode and typed call instructions for direct calls, function handles, `feval`, multi-output calls, and expanded arguments.
- Runtime and Turbine callback paths resolve callback names through `SemanticFunctionRegistry` to stable session `FunctionId`s, then invoke `SemanticFunctionBytecode` by semantic identity when the current bytecode/session product contains the callee.
- `RunMatSession` persists a semantic function registry across successful interactive inputs and remaps per-input function IDs into session-unique IDs before execution.
- `RunMatSession` no longer stores or seeds previous-input functions through legacy HIR statements; previous-input function resolution is registry-backed.
- Semantic function registry entries retain source IDs so callback diagnostics and replacement logic can stop depending on legacy HIR statements.
- `SemanticFunctionRegistry` indexes functions by defining source so session-owned replacement/removal can operate on semantic metadata rather than legacy HIR statements.
- Replacing a function from an older source retires the older source's semantic function group from the session registry.
- Redefining a function in the session semantic registry replaces the prior registry entry for that name.
- Direct calls to functions defined in previous interactive inputs resolve through the session semantic registry and lower to `CallSemanticFunction`.
- Direct multi-output calls to functions defined in previous interactive inputs resolve through the session semantic registry and lower to `CallSemanticFunctionMulti`.
- Direct cell-expansion calls to functions defined in previous interactive inputs resolve through the session semantic registry and lower to `CallSemanticFunctionExpandMulti`.
- Direct cell-expansion multi-output calls to functions defined in previous interactive inputs resolve through the session semantic registry and lower to `CallSemanticFunctionExpandMultiOutput`.
- Function-handle calls to functions defined in previous interactive inputs execute through `Value::SemanticFunctionHandle` identities.
- Local zero-capture named function handles now lower directly to `CreateSemanticFunctionHandle`; anonymous and captured function handles remain closures because they need capture/layout metadata.
- Function-handle literals for functions defined in previous interactive inputs now bind to `CreateSemanticFunctionHandle` after session registry attachment, so `@f` carries semantic identity instead of a name-only `FunctionHandle` when the target is known.
- `feval` multi-output calls through previous-input semantic function handles invoke semantic identities.
- Expanded `feval` calls through previous-input semantic function handles invoke semantic identities.
- Expanded multi-output `feval` calls through previous-input semantic function handles invoke semantic identities.
- Multi-output `feval` dispatch now checks the semantic registry and errors on unresolved user-function names instead of invoking a named legacy fallback.
- `feval` closure dispatch resolves closure names through the semantic registry when an embedded semantic function id is unavailable.
- Runtime `feval` now invokes embedded `Closure.semantic_function` and `Value::SemanticFunctionHandle` identities directly before name fallback.
- Runtime `feval`, `cellfun`, and `arrayfun` now route semantic callback attempts through `SemanticCallableRequest`, the runtime-facing descriptor bridge for function identity/name, prepared args, requested outputs, and callback kind.
- Runtime `feval`, `cellfun`, and `arrayfun` now ask a VM-installed semantic name resolver before name-only user-function or builtin fallback, so runtime callback strings/`FunctionHandle(name)` values can still resolve to session semantic functions when the active VM bytecode registry knows the name.
- Local and previous-input user-function calls inside `end` expressions now carry `EndExpr::SemanticCall` identities instead of relying on name recovery.
- Compiler-produced `feval('name', ...)` callees for local/session semantic functions now bind to `CreateSemanticFunctionHandle` before reaching runtime `feval`, including multi-output and expanded-argument forms.
- `cellfun` and `arrayfun` string callback literals for local/session semantic functions now bind to `CreateSemanticFunctionHandle` bytecode before reaching runtime builtins.
- Named user-call bytecode dispatch now checks the semantic registry before builtin fallback, then errors if unresolved.
- Turbine direct `CallSemanticFunction` bytecode now compiles through a semantic host callback by `FunctionId` instead of falling back to the interpreter.
- Turbine named `CallFunction(name)` bytecode now resolves the active `SemanticFunctionRegistry` at compile time and lowers semantic-known names directly by `FunctionId`, before considering legacy-shaped callback definitions.
- Turbine direct and named semantic multi-output calls now compile through a semantic host callback that writes numeric outputs into JIT result slots; semantic expanded calls now compile through tagged `TurbineValue[]` variables plus `TurbineArgSpec`, including true cell `expand_all` arguments. Generic `feval`/external expansion and object expansion remain outside this semantic JIT path.
- Multi-output `feval` no longer enters legacy user fallback after semantic resolution; unresolved names now error instead of reconstructing legacy HIR bytecode.
- The VM legacy user fallback compiler path has been removed; `compile_legacy_named_user_dispatch_fallback`, `compile_legacy_user_dispatch_fallback`, `PreparedLegacyUserCall`, and `CompiledLegacyUserDispatch` no longer exist.
- `LegacyUserFunction`, `Bytecode.functions`, and `ExecutionContext.functions` have been removed from VM bytecode/runtime metadata; callable behavior now goes through semantic function registries or explicit unresolved-call errors.
- The old VM legacy HIR compiler modules have been deleted; production bytecode compilation is semantic HIR/MIR/layout driven.
- VM callable execution now has an initial `CallableDescriptor` ABI that carries callee identity, prepared arguments, requested output count, fallback policy, display name, call kind, and optional source/span metadata for `feval`, direct semantic bytecode calls, expanded semantic calls, registry-resolved named calls, and semantic user-function calls inside `end` expressions.

Target state:

- A runtime user-function callback receives or resolves a stable semantic function identity, not a legacy statement body.
- The callback invokes already compiled semantic function bytecode from `Bytecode.semantic_function_registry` or a session-owned semantic function registry.
- Function invocation accepts one canonical call frame shape:
  - callee identity: `FunctionId`, function-handle value, or dynamic callable value
  - captured values: closure capture slots, already ordered by `VmAssemblyLayout`
  - input values: runtime arguments after expansion
  - requested outputs: exact/at-least/unknown output policy
  - source/callsite metadata: span/source IDs for diagnostics
- The same ABI is used by direct calls, local functions, anonymous closures, `feval`, runtime callbacks, and Turbine callbacks.

Design implication:

- Legacy user-function bytecode metadata is gone; unresolved or external function names must become semantic descriptors before execution or fail as unresolved calls.
- The current `CallableDescriptor` is the first runtime boundary for replacing remaining name-shaped call metadata; the long-term descriptor should grow `FunctionId`/`DefPath`, output policy, and layout/capture data where available.

Current unresolved-call inventory:

- Raw fallback compiler boundary: removed. There is no VM path that rebuilds a compatibility HIR program and recompiles it during dispatch.
- Multi-output `feval`: `handle_feval_user_multi_output` asks `SemanticFunctionRegistry` / `runmat_runtime::user_functions::try_call_semantic_function`; unresolved names now raise `UndefinedFunction`.
- Named/expanded user calls: `handle_prepared_user_function_call` checks the semantic registry, then builtin dispatch, then raises `UndefinedFunction`. Primary MIR lowering emits `CallSemanticFunction*` for semantic function callees.
- End-expression callbacks: local/session user-function calls carry `EndExpr::SemanticCall` and execute through `CallableDescriptor`; unresolved names or values without semantic identity now fail instead of entering legacy recompilation.
- Turbine external callback behavior: direct semantic calls and semantic-known named calls lower by `FunctionId`; unresolved named callbacks now remain outside JIT semantic execution and no longer invoke isolated or VM legacy user-function fallback.
- Runtime callback builtins: `feval`, `cellfun`, and `arrayfun` invoke semantic callbacks through `SemanticCallableRequest`; the request still resolves name-only handles through the active VM semantic resolver before builtin/name fallback.

Classification:

- Semantic-first, replaceable next: runtime producers that can be upgraded to semantic handles before crossing the VM/runtime ABI, plus non-semantic expanded JIT call shapes after their callees are represented by semantic descriptors instead of names.
- Semantic-first, blocked by callable identity shape: runtime-created strings and plain `Value::FunctionHandle(name)` values where no active semantic resolver is installed or the resolver cannot map the name to a stable semantic identity; compiler/session-produced handles now carry `Value::SemanticFunctionHandle` identity.
- External compatibility boundary: unresolved/external callback names that are not semantic-resolved now fail as unresolved calls; they no longer execute through interpreter legacy recompilation paths or Turbine isolated legacy callback execution.
- Dead/duplicate raw fallback call sites: removed; searches for the legacy fallback compiler helpers should return no production hits.

Completed implementation slices:

- Added a semantic user-function invoker path that executes `SemanticFunctionBytecode` by `FunctionId` with captures, args, and requested outputs.
- Routed VM/runtime callback paths and Turbine direct/named single-output and multi-output call paths through semantic registry identity when the current bytecode product knows the callee.
- Removed the centralized legacy fallback; unresolved/external dynamic functions must become semantic descriptors or remain unresolved.

### 2. Collapse Legacy HIR Compatibility Seams

Observed older-HIR artifacts worth collapsing:

- The compatibility AST/lowering seam in `runmat-hir` has been removed; `runmat_hir` now exposes semantic lowering only.
- The legacy-shaped user-function record and `runmat_vm::legacy::*` compatibility namespace have been removed. Downstream callers should use semantic function bytecode/registry APIs.
- VM execution internals such as `CallFrame` and `ExecutionContext` are no longer root-level `runmat_vm` exports or bytecode prelude exports; call-stack diagnostics use runtime call-frame types instead.
- Legacy bytecode compilation is no longer exposed through VM public modules or used by VM dispatch.
- `RunMatSession` now persists workspace bindings as stable ABI keys plus current VM slots; HIR lowering receives a derived slot map only at compile time.
- `LoweringResult` now carries semantic assembly/index products without legacy HIR projection maps or placeholder inference fields.
- LSP analysis uses semantic bindings plus shape facts rather than legacy variable maps.
- Parser compatibility now has an explicit `runmat` mode; `CompatibilityMode::RunMatExtended` no longer has to collapse to parser `matlab` mode at the boundary.
- MIR function summaries now reuse canonical `runmat_hir::FunctionAbi`; the duplicate `FunctionAbiSummary` compatibility wrapper has been removed.
- MIR spawn-safety summarization now returns `SpawnSafetyFact` directly; the `SpawnSafetySummary` wrapper record has been removed.
- MIR source maps and analysis module summaries no longer duplicate per-module `compatibility_mode` metadata; this compatibility policy signal now stays in HIR/lowering context rather than being threaded through MIR products without consumers.
- HIR products no longer duplicate lowering-context compatibility policy: `HirAssembly.compatibility_mode` and `LoweringResult.compatibility_mode` have been removed as write-only metadata, while strict-mode behavior remains enforced from `LoweringContext` during lowering.
- Dead MIR incremental product metadata surface has been removed: the unconsumed `ProductCacheKey`/`CacheProduct` API and `incremental` module (including unused variants such as `HirModule`, `MirBody`, `ClassMetadata`, `AnalysisFacts`) were test-only and had no production callers.
- Dead MIR module-summary compatibility surface has been removed: `AnalysisStore.modules` and `ModuleSummary` aggregation were unconsumed outside MIR tests, so module-level effect aggregation metadata is no longer synthesized separately from per-function summaries.
- Dead MIR source-map compatibility metadata has been removed: `MirSourceMap.source_unit` and its lowering plumbing were unconsumed outside MIR tests once module-summary aggregation was deleted.
- Dead MIR source-map compatibility metadata has been further trimmed: `MirSourceMap.module` was also unconsumed outside MIR tests and has been removed with corresponding lowering/test cleanup.
- Dead MIR source-map compatibility metadata has been fully trimmed to active analysis inputs: write-only `MirSourceMap.function` and `MirSourceMap.enclosing_class` were removed after confirming no production consumers.
- Dead MIR source/local record compatibility metadata has been trimmed to active analysis fields: `MirSourceRecord` now carries only expression linkage, and `MirLocalSource` now carries only local+expression linkage after removing unconsumed statement/span/binding metadata.
- MIR local-to-expression provenance is now carried directly on `MirLocal.expr`, and duplicate `MirSourceMap.locals` / `MirLocalSource` plumbing has been removed while preserving expression-fact projection in analysis.
- Dead MIR analysis product metadata has been trimmed further: `AnalysisStore.liveness` is removed from stored analysis outputs (while standalone `analyze_liveness` remains available), because no production consumers read the stored map.
- Dead MIR analysis product metadata has been trimmed further: `AnalysisStore.bindings` and `BindingFact` are removed from stored analysis outputs because no production consumers read the binding-fact map (local facts remain on `mir_locals`).
- Dead MIR analysis product metadata has been trimmed further: `AnalysisStore.functions` is no longer stored in analysis outputs; function-summary propagation remains internal to `analyze_assembly` for spawn-safety classification and diagnostics.
- Dead MIR analysis product metadata has been trimmed further: `AnalysisStore.spawn_boundaries` is no longer stored in analysis outputs; spawn-boundary classification now uses direct summary-driven analysis in tests while spawn-safety diagnostics remain emitted by `analyze_assembly`.
- Dead MIR analysis product metadata has been trimmed further: `AnalysisStore.expressions` and `ExprFact` are removed from stored analysis outputs because no production consumers read expression-fact maps; output typing and summary projection continue to use `mir_locals`.
- Dead MIR source-map statement provenance has been removed: `MirBody.source_map`, `MirSourceMap`, and `MirSourceRecord` were production-unconsumed metadata, so MIR lowering and tests now rely on CFG/local semantics directly without statement-record side channels.
- Dead MIR local expression provenance has been removed: `MirLocal.expr` and expression-tagging temp-local plumbing were production-unconsumed metadata after expression-fact store removal, so MIR locals now retain only binding/kind/span identity.
- Dead MIR local-fact metadata has been trimmed further: `MirLocalFact` no longer stores `empty_array_role`, `expansion`, `operator`, or duplicate `tensor_element_domain` payloads because those fields had no production consumers once expression-fact projection was removed; local facts retain canonical type/shape/value-flow/async/init data used by summaries and static analysis.
- Dead MIR summary metadata has been trimmed further: `FunctionSummary.spawn_safety` was removed as a write-only field with no production readers, while spawn-boundary classification continues to run through `analyze_spawn_boundaries_with_summaries` diagnostics.
- Dead MIR spawn-safety helper surface has been trimmed further: `summarize_spawn_safety` was removed after dropping the write-only summary field, keeping spawn-safety semantics anchored on explicit boundary analysis and diagnostics.
- `analyze_assembly` no longer runs transitive `propagate_function_summaries` for diagnostics because the active spawn-safety path consumes direct capture facts from per-function summaries; this removes an unconsumed production merge pass while preserving standalone summary-propagation APIs/tests.
- Dead MIR summary propagation surface has now been removed entirely: `propagate_function_summaries` and its dedicated transitive-effect tests were test-only after production callsite removal, so summary behavior now reflects direct per-body semantics unless an explicit propagation consumer is reintroduced.
- Dead MIR spawn-safety pre-classification surface has been removed: `analyze_spawn_boundaries` and its unclassified smoke test were test-only after the diagnostics path moved to summary-classified boundary analysis.
- Dead MIR liveness analysis surface has been removed entirely: standalone `analyze_liveness`/`LivenessFacts` had no production consumers after store cleanup, so the module and liveness-only tests are deleted.
- Dead MIR summary handle-tracking metadata has been trimmed further: `FunctionSummary.function_handles` and its scan plumbing were test-only with no production readers, so summary scanning now tracks only capture/effect/call facts consumed by active analyses.
- Dead MIR summary mutation mirror has been removed: `FunctionSummary.place_mutations` was test-only duplication of MIR statement data, so summary records no longer echo `MirStmtKind::PlaceMutation` entries.
- Dead MIR requested-output summary mirror has been removed: `FunctionSummary.requested_output_sensitive` was test-only and duplicated information derivable from call records + output facts, so the field and builder helper are deleted.
- Dead MIR output-flow summary mirrors have been removed: `FunctionSummary.output_value_flows` and `output_async_values` had no production readers and duplicated derivable analysis information, so summary outputs now carry canonical type/shape facts only.
- Dead MIR output-shape summary mirror has been removed: `FunctionSummary.output_shapes` had no production readers and duplicated shape information derivable from local/output facts.
- Dead MIR call-dispatch summary mirror has been removed: `CallSummary.dispatch`/`NominalDispatchHook` had no production readers and duplicated call-shape facts already present on MIR call records.
- Dead MIR workspace-binding write mirrors have been removed: `FunctionSummary.writes_globals` and `writes_persistents` were test-only binding-set metadata with no production readers; workspace-effect classification continues to run from explicit `WorkspaceEffect` markers.
- Dead MIR summary identity mirror has been removed: `FunctionSummary.function` duplicated function identity already provided by the summary map key/callsite and had no production readers.
- Dead MIR unknown-call barrier summary mirror has been removed: `FunctionSummary.may_call_unknown` was test-only metadata with no production readers, and unresolved-call coverage now asserts directly on `summary.calls` callee identities.
- Dead MIR workspace-effect summary scan helper has been removed: `scan_workspace_effect` was a no-op after dropping workspace binding write mirrors, so summary collection now records explicit `WorkspaceEffect` markers directly.
- Dead MIR ABI summary mirror has been removed: `FunctionSummary.abi` duplicated `MirBody.abi` and had no production readers; ABI coverage now asserts directly on lowered body ABI.
- Dead MIR effect summary mirror has been removed: `FunctionSummary.effects` was production-unconsumed metadata duplicating explicit MIR `WorkspaceEffect` / `EnvironmentEffect` statements and async terminator/call facts; tests now assert these markers directly on MIR bodies.
- Dead MIR analysis effects wrapper has been removed: `EffectSummary` in analysis facts had no remaining production consumers once summary effect mirroring was deleted.
- Dead MIR output summary mirror has been removed: `FunctionSummary.outputs` and the associated output-join helper pipeline were test-only metadata with no production readers; summary construction now tracks only capture/call facts needed by active spawn-safety analysis.
- Dead MIR summary-store coupling has been removed: `summarize_body` no longer depends on `AnalysisStore`/`MirLocalKey` inputs after output summary mirror removal, reducing the API to direct MIR-body summarization.
- Dead MIR call summary mirrors have been removed: `FunctionSummary.calls` and `CallSummary` were production-unconsumed metadata duplicating direct MIR call nodes; call-shape tests now assert against lowered `MirRvalue::Call` values via test helpers.
- Dead MIR summary wrapper surface has been removed: `FunctionSummary` was replaced with explicit `CaptureFacts`, and the API now uses `analyze_capture_facts` plus `analyze_spawn_boundaries_with_capture_facts` to reflect the remaining production use (spawn-safety capture classification) without summary-shaped compatibility naming.
- Dead MIR capture-summary public API surface has been trimmed further: `CaptureFacts`/`analyze_capture_facts` are now crate-internal implementation details, while public callers use `analyze_assembly_spawn_boundaries` for spawn-boundary classification.
- Dead MIR per-body spawn-safety classifier API surface has been trimmed further: `analyze_spawn_boundaries_with_capture_facts` is now internal-only, and analysis/tests consume the assembly-level classifier entrypoint.
- Dead MIR `analysis::summaries` compatibility seam has been removed entirely: capture-fact scanning is now inlined into `spawn_safety` implementation, `analysis/summaries.rs` is deleted, and `analysis/mod.rs` no longer declares a `summaries` module.
- Dead MIR analysis fact compatibility enums have been removed: `FusibilityFact`, `ParallelSafetyFact`, and `AccelEligibilityFact` in `analysis/facts.rs` had no production consumers and no longer ship as unused analysis API surface.
- Dead MIR spawn-safety diagnostic export surface has been trimmed: `diagnose_spawn_safety` is no longer publicly exported and is now module-internal (`pub(super)`), with diagnostics flowing through `analyze_assembly`.
- Dead MIR uninitialized-read diagnostic helper export surface has been trimmed: `diagnose_uninitialized_reads` is now module-internal (`pub(super)`), and lowering tests assert these diagnostics through `analyze_assembly` outputs instead of calling the helper directly.
- Dead MIR semantic-misuse diagnostic helper export surface has been trimmed: `diagnose_semantic_misuse` is now module-internal (`pub(super)`), keeping semantic misuse diagnostics flowing through `analyze_assembly` rather than as a standalone public helper.
- Dead MIR spawn-boundary classifier export surface has been trimmed: `analyze_assembly_spawn_boundaries` is now module-internal (`pub(super)`), `analysis/mod.rs` no longer re-exports `spawn_safety::*`, and lowering tests now assert spawn-safety behavior through `analyze_assembly` diagnostics/messages plus explicit MIR spawn-shape checks.
- Dead MIR lowering compatibility exports have been trimmed: `lowering::lower_function` is removed from the public API, and `MirLoweringContext` is now crate-internal with only `lower_assembly` exported for external MIR lowering entrypoints.
- Dead MIR lowering context field surface has been trimmed: `MirLoweringContext.binding_locals` is now private to `lowering::ctx`, keeping binding-local map mutation/lookup behind context methods instead of field-level access.

Target cleanup direction:

- Keep source compatibility behavior, but represent it through semantic assembly, analysis facts, and workspace ABI records.
- Continue using the session workspace binding table as the durable workspace ABI source, deriving transient VM slot maps only for compiler/runtime execution boundaries.
- Keep replacing remaining HIR compatibility seams with semantic descriptors or remove them from public compatibility surfaces.
- Move tests that only need compiler behavior from hand-built legacy HIR to semantic source fixtures or semantic MIR fixtures.
- Do not reintroduce legacy dispatch helpers for unresolved/external callbacks; those identities need registry-backed semantic descriptors.

### 3. Normalize Call Shapes

Current call lowering has converged, but the bytecode still has many sibling instructions:

- direct semantic function calls
- dynamic `feval` calls
- builtin calls
- function expansion calls
- method/member-index calls
- multi-output variants

This is acceptable while removing fallbacks, but the semantic model should eventually expose one call descriptor family before bytecode selection.

Target MIR concept:

- `MirCall` should carry callee kind, syntax kind, argument expansion specs, requested output policy, object/member dispatch policy, and effect facts.
- Bytecode lowering may still emit specialized instructions for performance, but it should not rediscover method-vs-builtin-vs-feval behavior from names.

Old-HIR cleanup opportunity:

- Legacy lowering encoded many call distinctions as string names (`feval`, method names, internal builtins). The new system can keep those distinctions as typed MIR call facts instead.

### 4. Member And Object Protocol ABI

Current state:

- Typed member bytecode now covers member load/store, dynamic member load/store, method/member-index calls, and expanded member-index calls.
- Object protocol calls now route through an `ObjectIndexDescriptor` serialization boundary in the VM call layer; remaining work is to move more call sites to build descriptors directly instead of prebuilding selector cells before the helper call.
- Object expansion paths in the main VM dispatch loop now build `ObjectIndexDescriptor` directly for paren/brace `subsref` instead of calling the legacy-shaped object index helper.
- Builtin, `feval`, and user-function expand-multi object paths now build `ObjectIndexDescriptor` directly for brace `subsref` expansion.
- VM indexing dispatch and slice read/write object paths now construct `ObjectIndexDescriptor` directly; the legacy-shaped `call_object_index_method` wrapper has been removed.
- `ObjectIndexSelector` can now carry raw index values and serialize protocol cells at the descriptor boundary, removing the generic object selector-cell helper and the older `subsref` selector-cell builders.
- `ObjectIndexSelector` now distinguishes empty expansion selectors and scalar-index selectors from arbitrary raw value selectors, so obvious object brace/paren paths no longer downcast scalar indices into generic value lists before descriptor serialization.
- Object protocol dispatch sites now use typed `ObjectIndexDescriptor` constructors for paren/brace `subsref` and `subsasgn`, so call sites no longer assemble operation/kind pairs manually.
- `ObjectIndexDescriptor` internals and selector/op enums are now crate-scoped, keeping object protocol serialization behind the VM call-layer ABI instead of exposing field-wise construction.
- Store-slice object assignment now uses the descriptor-backed `object_subsasgn_paren` path directly instead of falling back to manually assembled `Class.subsasgn` arguments.
- Member load/store object fallback checks now use descriptor-layer helpers for `subsref`/`subsasgn` membership and dispatch instead of importing object protocol operation names into resolver code.
- Object operator and handle-method dispatch now go through typed VM call-layer helpers instead of exposing raw `call_method` argument-vector assembly to dispatch modules.
- `CallableDescriptor` construction now routes through internal semantic, `feval` forwarding, and name-only fallback constructors, narrowing direct access to callable target fields.
- Runtime semantic callback bridge construction now goes through `SemanticCallableRequest` constructors instead of open field assembly at `feval`/`cellfun`/`arrayfun` call sites.
- Semantic HIR now has a shared `CallableIdentity` and explicit `CallableFallbackPolicy` vocabulary over existing `DefPath`, builtin, semantic-function, method, dynamic-name, and external-name identities.
- MIR calls and function-handle operands now carry `CallableIdentity`, while VM lowering still maps known identities onto the existing bytecode instructions.
- `CallableDescriptor` and runtime `SemanticCallableRequest` now accept resolved `CallableIdentity` instead of open `{function, name}` request fields, while preserving current semantic and name-fallback behavior.
- VM callback coverage now asserts unresolved external function handles and `cellfun` callbacks fail with `RunMat:UndefinedFunction` rather than entering legacy fallback execution.
- `runmat-hir` tests now assert semantic assembly/index shapes directly; compatibility-lowering helper coverage has been removed with the compatibility seam.

Target state:

- Object and handle-object indexing should receive a structured index descriptor rather than ad hoc cells and protocol strings assembled at each call site.
- The descriptor should represent paren, brace, dot/member, colon, range, end, and comma-list expansion directly.
- `IndexKind::Dot` should be eliminated from MIR indexing if it is not semantically constructed, or explicitly mapped to member MIR if a parser/lowering path can produce it.

Collapse opportunity:

- The old design blurred `obj.name(...)` among function call, method call, member read, and member indexing until late VM dispatch. The semantic design should keep ambiguity only where MATLAB requires runtime dispatch; all statically known member/index shapes should lower to typed MIR/bytecode.

### 5. Indexing And Assignment Normalization

Current state:

- Most common scalar, range, vector, logical, cell, and member store-back cases now lower through semantic bytecode.
- Some VM dispatch still classifies index/value shapes for compatibility.

Remaining design work:

- Decide whether logical indices remain explicit `MirIndexComponent::Logical` or normalize to ordinary expression operands producing logical arrays before MIR bytecode lowering.
- Decide whether colon/end in cell/multi-assign contexts should be handled by the same selector plan used for tensor slice operations.
- Model deletion explicitly in MIR instead of recognizing empty RHS in runtime stores.
- Collapse duplicate cell selection logic across `IndexCell`, `IndexCellExpand`, call expansion, and `StoreIndexCell` into one selector/plan API.

Target outcome:

- VM instructions execute explicit index plans.
- Runtime helper paths validate and apply plans.
- No VM branch needs to infer whether a tensor index means scalar indexing, slice indexing, logical indexing, or deletion.

Accepted resolution (A/A/A/A/A):

- Use one canonical selector-plan representation across tensor/cell/object indexing and assignment paths.
- Model deletion explicitly in MIR/bytecode (no runtime empty-RHS inference as the semantic source of truth).
- Classify scalar-vs-slice assignment in MIR/lowering only; VM executes explicit operation kinds.
- Remove remaining transitional `IndexKind::Dot` compatibility branches unless a sourced parser/HIR path requires them.
- Enforce selector-plan invariants at compile/lowering boundaries; runtime validation remains execution-safety focused.

### 6. Varargout And Multi-Output Semantics

Current state:

- Multi-output direct, dynamic, and expanded calls are typed in bytecode.
- MIR output targets are fixed per-assignment (`Place` or `Discard`) and compile-time-checked against requested output count.

Design follow-up:

- If varargout target expansion is intended in the future, define the target-list ABI explicitly:
  - fixed outputs before expansion
  - varargout capture cell/output-list target
  - discard targets
  - requested output count propagation to callee

Collapse opportunity:

- Legacy multi-assign handling mixed output shaping, stack unpacking, and assignment target handling. Semantic MIR should own the output target list and requested output policy before bytecode.

### 7. Async, Future, And Spawn Semantics

Current state:

- MIR contains `Future`, `Spawn`, and `Await` shapes, but bytecode lowering is incomplete.
- These are RunMat extensions and should not be rushed as VM op patches.

Design needed before implementation:

- Runtime `Future`/task value representation.
- Suspension/resume ABI for `Await` terminators.
- Workspace/export behavior for top-level await versus function-local await.
- Cancellation, diagnostics, and call-stack metadata across suspension.
- Compatibility policy: MATLAB-strict should reject unsupported async syntax before MIR/VM.

Accepted resolution (A/A/A/A/A):

- Runtime value shape: represent futures as task identities (`Future(TaskId)`-style), with task state in runtime-managed registry structures.
- Suspension model: interpreter/runtime suspension records (frame/resume-point based), not CPS/state-machine lowering.
- Top-level await contract: host-facing suspension outcome and explicit resume, not implicit blocking semantics.
- Cancellation semantics: cooperative cancellation at safe yield points/await boundaries.
- Diagnostics: stitch logical async call stacks (spawn/await/resume metadata), not active-frame-only reports.

### 8. Struct And Object Aggregate Semantics

Current state:

- Tensor and cell aggregates lower directly.
- `MirAggregateKind::Struct` and `ObjectArray` are still bytecode gaps.

Design options:

- Lower struct/object aggregate literals into typed construction bytecode.
- Lower them into empty construction plus member stores if that better matches MATLAB semantics.
- Route through public builtins only if the source syntax is actually a builtin call, not as compiler-internal construction.

Preferred direction:

- Add typed aggregate construction instructions once semantic HIR has a clear source form for struct/object literals.

Accepted resolution:

- Use typed aggregate bytecode construction operations as the canonical lowering target (not public-builtin indirection for syntax literals).
- Preserve strict source-order evaluation for aggregate element/field expressions.
- Handle duplicate-field semantics by explicit compatibility policy decision (documented behavior; no implicit fallback heuristics).
- Keep syntax-literal construction as compiler-internal typed ABI, while user-authored builtin calls stay on public call-dispatch paths.
- Surface invalid aggregate forms as semantic/lowering errors whenever determinable before runtime.

### 9. Compatibility Mode Cleanup

Current state:

- Compatibility policy, parser mode, and RunMat extension flags are still partly entangled.

Target state:

- Parser mode answers syntax acceptance questions.
- Semantic compatibility mode answers behavior policy questions.
- Runtime/session mode answers host policy questions such as top-level await and workspace export.

Collapse opportunity:

- Replace broad variants such as `RunMatExtended` where they mask multiple independent policy bits.

Accepted resolution (A/A/A/A/A):

- Parser mode owns syntax acceptance exclusively.
- Lowering context owns explicit semantic behavior policy bits.
- Session/request host policy owns execution-time host behavior (for example top-level await/export semantics).
- Request/ABI policy surface keeps only fields with active enforced behavior; remove dead placeholders.
- Keep distinct `RunMat` vs `Matlab` labels only where they map to concrete behavior differences; otherwise treat as alias-equivalent policy mapping.

## Remaining Gap Classification

Treat current MIR bytecode gap markers as follows:

- `control-flow terminator`: design gap for async/await or future terminators, not a small VM patch unless a concrete source reproducer exists.
- `varargout expansion`: parser/HIR does not currently construct this shape; keep as future ABI design work rather than a live bytecode gap.
- `slice index`: comparison-derived logical tensor masks and call-result index variables now lower through semantic slice bytecode for read/write; remaining gaps are selector-plan normalization for range/end/colon in non-tensor and cell contexts.
- `dot assignment` / `dot indexing`: static member read/write source now ratchets through semantic member MIR; remaining `IndexKind::Dot` branches appear transitional and should be verified for removal or mapped explicitly if a source reproducer reaches them.
- `indexed member store-back`: struct-field indexed assignment and cell-member store-back are ratcheted through semantic place chains; remaining forms are likely object/dynamic/dot descriptor work.
- `rvalue` / `operand`: async/future/spawn/temp modeling or unsupported semantic forms; classify by source reproducer before implementing.
- `{count} call outputs`: semantic user-function, `feval`, `size`, min/max family, sort/set/index builtins (`sort`, `unique`, `find`, `union`, `ismember`, `sortrows`), and linalg factorization builtins (`chol`, `lu`, `qr`, `svd`, `eig`) are ratcheted through multi-output bytecode/runtime output context; generic rvalue call outputs and broader builtin output splitting remain call ABI/output-list policy, so avoid ad hoc bytecode variants until call descriptor design is settled.
- `call callee`: semantic resolver/DefPath work; do not fall back to string builtin guesses.
- `aggregate kind`: struct/object aggregate design.
- `function handle target`: builtin and anonymous semantic handle `feval` are ratcheted; remaining method/DefPath targets appear to require resolver/DefPath function-handle ABI work.
- `assignment place`: multi-assign output storage now reuses MIR place assignment for non-local targets; remaining assignment-place gaps should be explicit source reproducers or object/dynamic descriptor plans, not generic slot-only lowering.

## Recommended Semantic Design Slice

The next high-leverage slice is replacing remaining name-shaped callable behavior with semantic descriptors, not another isolated callback patch.

Concrete plan:

1. Replace non-external dynamic callback sites with registry lookup or typed semantic lowering.
2. Extend registry-backed lowering beyond direct calls to remaining callable shapes that still carry only dynamic names, where layout/capture information is available.
3. Keep passing semantic callable identity through end-expression callback paths instead of rediscovering names at runtime.
4. Add callable descriptors for remaining dynamic-name callback shapes that can be resolved before runtime.
5. Add coverage that unresolved external callback names fail cleanly instead of reintroducing legacy recompilation.

Current ratchet status:

- VM basics, matrix-division, bitwise row-vector, and import-error bytecode tests now use the semantic HIR/MIR `compile` path where they only need source-level or semantic bytecode behavior.
- The FFT end-range bytecode assertion in VM basics now uses semantic bytecode instead of `compile_legacy`.
- Simple VM basics execution tests for arithmetic, zero-output builtin calls, and `nextpow2` now run semantic bytecode.
- Additional VM basics execution tests for complex literals, leading-dot numeric forms, elementwise division, `chol`, `uint16`, and `atan2` RHS expressions now run semantic bytecode.
- FFT `end/2` range materialization, complex range assignment, multidimensional `end` ranges, out-of-bounds `end+1`, variable-offset `end`, builtin/user-function-call `end` expressions, pow/leftdiv end-expression combinations, and `fftn`/`ifftn` indexing basics now run semantic bytecode.
- Remaining VM basics ratchets are semantic bytecode/accel-graph shape assertions (including multi-output argument shape and object range-end protocol payloads).
- Matrix-division execution tests now run semantic bytecode, including accel-graph shape assertions.
- Loop execution tests now run semantic bytecode, including the stochastic-evolution instruction-shape assertion.
- VM `for` terminator lowering now supports non-range iterables through column iteration (`for v = A`) by materializing `A(:, k)` columns in bytecode, and control-flow semantic VM tests now cover matrix-column loop accumulation.
- Dead MIR await-cleanup compatibility surface has been removed end-to-end: `MirTerminatorKind::Await` no longer carries an always-`None` `cleanup` edge, MIR CFG analyses now treat await terminators as single-resume edges, and VM lowering no longer carries an unreachable `"await cleanup blocks are not supported"` error branch.
- Several VM functions success-path and arity tests now run semantic bytecode, including nested user-function calls, function-handle/cellfun round-trips, `nargin`/`nargout`, fixed-arity/minimum-varargin input errors, fixed-output arity errors, shared input/output names, inline `fprintf`/`sprintf` cast arguments, root/function-output struct materialization, nested/dynamic member assignment, mixed member/cell/index reads, and numeric bitwise array operations.
- VM functions import execution, import ambiguity, and import-shadowing/metaclass guard tests now run semantic bytecode for package builtin and static-method imports.
- Parser+HIR function-handle lowering now accepts qualified literals (for example `@pkg.remote_inc`) and lowers dotted handle targets to `DefPath`/external-handle bytecode identities instead of generic name-shaped handles.
- Dotted unresolved direct calls with unbound identifier bases (for example `pkg.remote_inc(1)`) now lower through plain qualified call identities with `ExternalBoundary` fallback instead of method-style dynamic-name fallback.
- Nested dotted unresolved direct calls with unbound member-chain bases (for example `pkg.sub.remote(1)`) are ratcheted through HIR/MIR/VM tests and lower through plain qualified call identities with `ExternalBoundary` fallback.
- Runtime `cellfun`/`arrayfun` text callback parsing now classifies qualified callback names (for example `pkg.callback`) as external callable identities with `ExternalBoundary` fallback instead of builtin/dynamic-name callbacks.
- Runtime `timeit` callback normalization now accepts `ExternalFunctionHandle` and `SemanticFunctionHandle` values directly, preserving typed callback identities through `feval` dispatch instead of rejecting non-legacy handle shapes.
- Runtime `pagefun` callback parsing now accepts `ExternalFunctionHandle` and `SemanticFunctionHandle` values and normalizes them through the existing operation parser instead of rejecting typed handle shapes as unsupported callback inputs.
- Runtime optimization callbacks (`fzero`/`fsolve`) are ratcheted to accept `SemanticFunctionHandle` values directly, preserving typed callback identities through solver `feval` calls.
- Runtime ODE callbacks (`ode23`/`ode45`/`ode15s`) are ratcheted for typed-handle execution (`SemanticFunctionHandle` and qualified `ExternalFunctionHandle`), confirming entrypoint callback values flow through shared `feval` descriptor dispatch without legacy handle-shape rejection.
- Qualified static method function handles are now ratcheted across HIR/MIR/VM (`@Point.origin`): semantic lowering preserves a def-path/imported identity path, and VM bytecode preserves the qualified handle identity string (`"Point.origin"`) for execution through `feval`.
- Dead HIR compatibility surface for function-handle method targets has been trimmed: `FunctionHandleTarget::Method` was unused by source lowering and removed, keeping method identity handling only on actual call-callee paths.
- Dead constructor callable identity compatibility surface has been trimmed across HIR/MIR/VM: `HirCallableRef::ClassConstructor` and `CallableIdentity::ClassConstructor` were unused by source lowering and removed, along with MIR summary constructor hooks and VM constructor-name fallback synthesis.
- Dead loop-iteration compatibility surface has been trimmed in HIR: `LoopIterationSemantics::WhileCondition` was unused by lowering/analysis and removed, keeping loop iteration semantics explicit to sourced `for` forms.
- Dead HIR evaluation-context compatibility surface has been trimmed: `EvaluationContext::ForRange` was unused by lowering/analysis and removed.
- Dead HIR `EvaluationContext` compatibility enum has been removed entirely after confirming it had no lowering/analysis/runtime consumers.
- Dead HIR `RequestedOutputs` compatibility struct has been removed after confirming no lowering/analysis/runtime consumers remained.
- Dead command/DefPath compatibility branches have been trimmed in HIR/MIR: `CommandArgument::OptionToken` plus `CommandOptionName`, and unused `DefPathSegment` variants (`Class`, `Method`, `Property`, `Entrypoint`, `Synthetic`) with `SyntheticName` were removed after confirming only function-path identities are currently produced.
- Dead MIR async/effects compatibility exports have been trimmed: unused `AwaitPoint`, `AsyncFact`, and `MirEffects` types (and the now-empty `effects` module) were removed while preserving active async behavior and spawn-boundary analysis paths.
- Dead loop-semantics plumbing has been removed across HIR/MIR: `HirStmtKind::For` and `MirTerminatorKind::For` no longer carry an unused `semantics` field, and `LoopIterationSemantics` has been removed after confirming lowering/analysis consume concrete loop shapes directly.
- VM functions direct cell-expansion, explicit function-return propagation, semantic `varargin` packing, and `feval(@f, varargin{:})` forwarding now run semantic bytecode.
- VM functions comma-list argument expansion now supports end-relative cell selectors (`end`, `end-1`, `end+1` error surfacing) through semantic MIR/bytecode without ad hoc fallback lowering.
- VM functions fixed-output user-function return propagation through outer user-call arguments, such as `h(g())` and `f(1, g())`, now runs semantic bytecode.
- VM functions `varargout` expansion through builtin and user-call arguments, such as `max(h())` and `f(g())`, now runs semantic bytecode.
- VM functions global and persistent declarations now lower to semantic bytecode workspace effects.
- VM functions classdef registration now lowers to semantic bytecode for property access enforcement.
- VM functions tensor indexing/write ratchets for logical mask assignment, gather/scatter roundtrip, shape broadcasting, column-major RHS mapping, and range/`end` read/write cases now run semantic bytecode.
- VM functions struct `isfield`/`fieldnames`, computed integer column-slice read/write ratchets, string aggregate concatenation, and `containers.Map` package calls now run semantic bytecode.
- VM functions type-class static `zeros` calls for `double.zeros` and `logical.zeros` now resolve through primitive class metadata and run through semantic bytecode.
- VM functions nested try/catch `rethrow` exception propagation now runs semantic bytecode; `functions.rs` has no ignored tests remaining.
- Operator-overload diagnostic bytecode in VM functions tests now uses semantic compilation instead of `compile_legacy`.
- VM functions operator-overload execution tests now assert concrete numeric outcomes for mixed-sided `+`, `.*`, `*`, relational (`<`, `==`), and element/matrix division variants (`.\\`, `./`) instead of fallback-only smoke execution.
- VM functions metaclass/static/dependent-property ratchets now assert concrete semantic outcomes: wildcard static-method import resolution materializes `Point` origin objects with expected properties, metaclass postfix `?Point.origin()` and `?Point.staticValue` assert object/property and static-value behavior, and dependent-property `setfield/getfield` coverage asserts both getter paths plus `p_backing` object storage.
- VM functions inline-cast `fprintf` coverage now asserts the returned byte count (`14` for `"Value: %.4f\\n"` with `double(single(3.14))`) in semantic execution, tightening the prior stack-underflow smoke path into an explicit behavioral check.
- VM functions class-property access ratchet now asserts concrete private-set diagnostics (`"Property 'secret' is private"`) instead of relying on a registration smoke pre-step and generic `is_err()` checks.
- VM functions import-ambiguity ratchets now assert diagnostic content (ambiguity/conflict/duplicate import wording) across specific-import, wildcard-import, handle-import, and duplicate class-star import conflict paths instead of generic compile-failure checks.
- Additional VM functions object/class/operator/OOP ratchets now run semantic bytecode, including object member get/set and method dispatch, class registration/static/inheritance paths, static classref calls, object cell expansion, operator overload variants, and negative OOP try/catch paths.
- VM range, meshgrid-range, global/persistent, and logical-operator tests now run semantic bytecode.
- VM indexing, matrix-slicing, and lvalue-assignment tests now run semantic bytecode.
- VM cell-array, exception, datetime, and closure/callback tests now run semantic bytecode, including captured and nested closure capture coverage.
- VM control-flow tests now mostly run semantic bytecode, including unknown-builtin catch behavior.
- VM multidimensional indexing tests now run semantic bytecode for logical row selection, slice assignment, and 3D indexing/slicing coverage; remaining N-D selector work is a semantic lowering/runtime-shape gap, not a legacy-execution gap.
- VM indexing-property tests now run semantic bytecode for scalar/logical/range write broadcasts, end-arithmetic stores, negative-step linear indexing, roundtrip scatter, fastpath broadcasts, simple cell expansion, and advanced N-D empty/vector selector and expansion coverage.
- VM indexing-property coverage now ratchets empty-`varargout` expansion behavior from a smoke path to an explicit semantic assertion: fixed single-output calls to zero-length `varargout` now assert `RunMat:VarargoutMismatch` instead of silently passing on unchecked runtime errors.
- Remaining `functions.rs` work is semantic behavior tightening (for example, stricter metaclass/postfix and dependent-property expectations), not legacy-execution migration.
- Turbine mixed arithmetic/function-call, simple scalar function compilation, scalar callback variable-isolation, compute-intensive scalar callback, nested scalar callback, function-parameter validation, and error-handling success-path coverage now use semantic registry-backed named calls instead of hand-built legacy function metadata.
- Turbine fallback-boundary tests now use unresolved-name bytecode or semantic registry fixtures; no `LegacyUserFunction` test fixtures remain.
- Remaining production legacy recompilation usage is gone.
- Dead MIR initialization compatibility surface has been trimmed: `MirLocalFact` no longer stores test-only `initialized` state, init dataflow no longer exposes per-local `final_state`, and `InitFact` is now internal to analysis implementation.
- Turbine `CallFunctionMulti` JIT lowering now resolves callable targets from typed `CallableIdentity` first (with optional `display_name` fallback), so named multi-output JIT calls no longer require name-only metadata to reach semantic function IDs.
- VM callable descriptor metadata now infers `display_name` from typed `CallableIdentity` when callsites omit explicit names, reducing name-duplication seams between bytecode call records and runtime call dispatch diagnostics.
- VM MIR call lowering no longer populates redundant `display_name` payloads on typed `CallFunction*` and method/member-index call instructions; runtime call diagnostics now rely on `CallableIdentity` + descriptor-side inference instead of duplicated bytecode name metadata.
- VM method/member-index dispatch no longer threads optional display-name arguments through closure call helpers; runtime method-call name resolution now derives strictly from `CallableIdentity` to avoid duplicate callsite name plumbing.
- Method/member-index bytecode instruction variants no longer carry a `display_name` field at all; dispatch patterns and tests now treat `CallableIdentity` as the sole callable-name source for those shapes.
- Function-call bytecode instruction variants (`CallFunctionMulti`, `CallFunctionExpandMultiOutput`) no longer carry `display_name`; VM dispatch and Turbine hashing/JIT resolution now derive names from typed `CallableIdentity` where needed.
- End-expression callback records (`EndExpr::ResolvedCall`) no longer carry `display_name`; compile-time rebinding, runtime end-expression evaluation, and object index end-expression encoding now derive names directly from `CallableIdentity`.
- VM callable descriptor construction no longer accepts explicit `display_name` on `CallableDescriptor::resolved`; direct-call and end-expression callsites now rely on descriptor-side inference from `CallableIdentity`, keeping explicit name overrides only on `feval`-shaped descriptor paths.
- VM call-descriptor metadata no longer carries a post-construction `with_call_kind` mutator shim; end-expression callsites now construct descriptors with the final call kind directly, reducing leftover compatibility-style metadata plumbing.
- VM class access-control ratchet now asserts stable semantic diagnostic content (`secret` + `Point`) for private property get failures instead of generic `is_err()` coverage in `classes_static_and_inheritance`.
- VM function-handle lowering for external/imported/method callables now uses only `mir_runtime_name_callee` runtime-name derivation (no secondary `CallableIdentity::display_name()` fallback), keeping handle emission aligned to the typed callee classification path.
- VM functions tests no longer keep an unused dead-code function-definition smoke helper; coverage remains through active semantic function-call tests.
- VM functions error-path ratchets now replace generic `is_err()` checks with semantic diagnostic assertions for: unqualified static property resolution failure (`staticValue` undefined), class property/method attribute conflict pairs (`Constant`+`Dependent`, `Abstract`+`Sealed`), non-cell brace expansion failure, and mixed range-end assignment shape-mismatch failure.
- VM `functions.rs` no longer contains generic `is_err()` assertions; error-path coverage is now consistently `expect_err` + identifier/message checks.
- Turbine bytecode hashing for `Instr::CallFunctionMulti` now keys on typed callable identity + fallback policy (with arity/output count) instead of reducing to `display_name`, removing a remaining name-shaped JIT cache key seam for unresolved/external call identities.
- `CallableIdentity`/`CallableFallbackPolicy` now derive `Hash` (and `Eq` for `CallableIdentity`) in HIR, enabling typed identity hashing at compiler-product boundaries without ad hoc string reconstruction.
- VM object end-expression encoding no longer serializes a `"<unnamed>"` placeholder for call identities without a display name; it now fails with `RunMat:UndefinedFunction` when a callable name cannot be represented for object protocol selector descriptors.
- VM shared-call tests now cover this boundary explicitly: object paren selector expression encoding rejects `EndExpr::ResolvedCall` values whose typed identity has no encodable callable name.
- VM callable descriptor unresolved-identity diagnostics no longer emit `"<unnamed callable>"`; when runtime fallback naming is unavailable and a typed identity has no display name, `UndefinedFunction` now reports the typed callable identity (`{identity:?}`) directly.
- VM descriptor tests now assert this typed-identity diagnostic path for unresolved anonymous callable identities, guarding against reintroduction of placeholder name diagnostics.
- Runtime direct-call unresolved diagnostics now mirror this typed-identity behavior: when `dispatch_callable_with_policy` cannot derive a fallback name from a typed callable identity, `RunMat:UndefinedFunction` includes the typed identity (`{identity:?}`) instead of a generic unnamed-function message.
- Runtime direct-call unresolved diagnostics now derive display names through explicit typed identity mapping (`strict_callable_display_name`) instead of generic `CallableIdentity::display_name()`, with malformed external identities ratcheted to typed-identity diagnostics.
- VM object end-expression selector encoding now resolves callable names through explicit typed identity rules (`strict_callable_display_name`) instead of generic `CallableIdentity::display_name()`, rejecting malformed external-name segments before encoding `EndExpr::ResolvedCall` descriptors.
- VM method/member-index dispatch now resolves callable names through explicit typed identity rules (`DynamicName`, `Method`, or single-segment `ExternalName`) instead of generic `CallableIdentity::display_name()`, and rejects unsupported callable identity shapes at the method/member boundary.
- VM callable descriptor metadata inference now resolves fallback display names through explicit typed identity rules (`strict_callable_display_name`) instead of generic `CallableIdentity::display_name()`, and descriptor tests now ratchet malformed external identities to preserve `display_name: None`.
- Turbine named multi-output JIT target resolution now gates semantic-registry name lookup by `CallableFallbackPolicy::allows_semantic_name_resolution_for(identity)` instead of unconditional `display_name()` lookup for all non-semantic identities.
- Turbine JIT coverage now asserts imported callable identities do not resolve through display-name reconstruction in named multi-output JIT lowering (`Undefined function: pkg.double_pair`), while dynamic-name identity resolution coverage remains green.
- Turbine named multi-output JIT target resolution now derives semantic lookup names from typed identity shapes only (`DynamicName`, or well-formed multi-segment `ExternalName`) instead of generic `CallableIdentity::display_name()` reconstruction; malformed/single-segment external identities stay off the JIT semantic path while well-formed external identities still lower through semantic registry `FunctionId` resolution.
- Runtime semantic callback descriptor resolution (`try_call_semantic_descriptor`) now derives semantic resolver names from typed identity shapes only (`DynamicName`, or well-formed multi-segment `ExternalName`) instead of generic `CallableIdentity::display_name()` reconstruction; malformed/single-segment external identities now skip semantic resolver lookup, with runtime callback policy tests ratcheting this boundary.
- HIR callable fallback policies now enforce the same well-formed external-name contract: `allows_semantic_name_resolution_for` and `allows_vm_name_fallback_for` accept `ExternalName` only when qualified names are multi-segment and contain no empty segments, preventing malformed external identities from reaching runtime/JIT name fallback paths.
- HIR `CallableFallbackPolicy::vm_fallback_name_for` no longer delegates to generic `CallableIdentity::display_name()`; it now derives fallback names through explicit typed identity mapping (`DynamicName`, `Method`, `Imported` module path, and well-formed multi-segment `ExternalName`) to keep fallback naming aligned with strict policy gating.
- VM compiler `mir_runtime_name_callee` no longer delegates external/runtime-handle naming to generic `QualifiedName::display_name()`; function-handle runtime names are now derived through explicit typed checks (non-empty dynamic/method/imported module names, and external names with no empty segments), preserving strict malformed-name rejection at handle emission time.
- Turbine named multi-output malformed external-identity coverage now asserts unresolved-call behavior (`Undefined function: pkg..remote_pair`) after policy/descriptor strictness, while well-formed external and dynamic identity semantic-resolution coverage remains green.
- HIR imported callable identity display names no longer fall back from `DefPath.module` to `DefPath.item` leaf names; `CallableIdentity::Imported.display_name()` now reflects only module-qualified import paths, with unit coverage that empty-module imported identities produce no display name.
- HIR `QualifiedName::display_name()` now rejects malformed empty segments (returns `None` instead of joining with empty components), so name rendering at typed identity boundaries preserves malformed-path strictness by default.
- HIR `CallableIdentity::display_name()` now rejects empty builtin/method/dynamic symbol names (returns `None`), removing remaining empty-string name synthesis at typed callable identity boundaries.
- HIR wildcard import resolution for call/function-handle targets no longer round-trips through `display_name`/`split('.')` string reconstruction; it now carries typed `QualifiedName` candidates through builtin qualification checks and `DefPath` construction directly.
- VM compiler imported callable runtime-name extraction no longer falls back from `DefPath.module` to `DefPath` leaf display names; imported function-handle emission now requires a module-qualified runtime name path (`path.module.display_name()`), matching typed imported identity semantics.
- VM imported-builtin resolution no longer falls back to manual `path.join` / `format!("{path}.{leaf}")` string synthesis when building qualified builtin call names; it now uses typed `QualifiedName(...).display_name()` only and skips invalid empty-segment import paths.
- VM `call::builtins` unit tests now cover typed qualified-name construction and explicit no-name behavior for empty import paths.
- VM imported-builtin qualified-name construction is now strict about typed path validity: any empty import path segment causes name construction to return `None` (no silent segment dropping), preserving the typed-path contract instead of normalizing invalid import metadata.
- VM feval named-target classification no longer normalizes malformed dotted names by dropping empty segments; malformed handles such as `pkg..remote_inc` now remain `DynamicName` + `RuntimeNameResolution` instead of being silently rewritten to external-boundary `pkg.remote_inc`.
- VM callable descriptor qualified-name conversion now preserves malformed dotted strings as a single external-name segment when needed (no empty-segment collapsing), and descriptor tests ratchet this behavior.
- Runtime qualified-name classification for handle-like callables is now strict: malformed dotted names (for example `Point..origin`) are no longer treated as external-qualified identities in `str2func`/`callable_identity_for_handle_name`/`arrayfun`/`cellfun` callback parsing; they remain dynamic/builtin-name forms unless a semantic resolver provides an identity.
- Runtime `str2func` and handle-identity tests now explicitly cover malformed dotted names (`Point..origin`, `pkg..remote_inc`) to guard against reintroducing empty-segment normalization in callback resolution paths.
- VM class/member external-identity construction in `call/shared` is now strict about base-name shape: malformed class bases are preserved as a single segment instead of being split after empty-segment filtering, while well-formed dotted bases continue to split into qualified segments.
- VM `external_qualified_display_name` no longer uses ad hoc string fallback formatting; it now relies on the typed identity display-name invariant, with unit coverage for malformed and well-formed base cases.
- HIR lowering no longer threads `CompatibilityMode` through lowering context for extension gating: `LoweringContext` now carries an explicit `runmat_extensions_enabled` semantic policy bit, `await`/`spawn` strict-mode checks consume that bit directly, LSP compatibility mapping now translates parser mode to that explicit policy, and semantic-lowering tests ratchet strict-mode rejection for both `spawn` and `await`.
- Core session lowering now propagates strict compatibility into semantic lowering via `runmat_extensions_enabled` for both normal script compilation and `input()` expression-eval hook lowering, with an integration ratchet asserting strict mode rejects `spawn` during execution.
- Parser compatibility policy now owns the RunMat-extension semantic toggle (`CompatMode::allows_runmat_extensions()`), and core/LSP lowering paths consume that shared parser-mode contract instead of duplicating strict-mode extension gating helpers.
- Dead HIR compatibility enum surface has been removed: `runmat_hir::CompatibilityMode` is no longer part of the compiler product, `ExecutionRequest.compatibility` now uses parser-native `runmat_parser::CompatMode`, and session request handling no longer maps through `parser_compat_from_abi`.
- Core request host policy now controls top-level await lowering: `ExecutionRequest.host_policy.top_level_await` is threaded into session lowering/eval-hook context via `LoweringContext::with_top_level_await_enabled`, synthetic entrypoint policy mirrors that flag, and new HIR/core ratchets assert that disabling the policy rejects top-level `await`.
- `runmat-core` semantic-VM unit ratchets that previously matched removed bytecode `display_name` payloads now assert against typed callable identities (`EndExpr::ResolvedCall.identity` and method/member-index instruction identities), keeping core tests aligned with display-name seam removal.
- Dead host-policy materialization ABI surface has been removed: `HostExecutionPolicy` no longer carries an unconsumed `MaterializationPolicy` field/enum, leaving only enforced request policy bits (`top_level_await`) at the core execution boundary.
- VM shared call helper `external_qualified_display_name` now resolves through strict typed callable-name mapping (`strict_callable_display_name`) instead of generic `CallableIdentity::display_name()`, with coverage for malformed and well-formed external base-name rendering.
- Dead core request entrypoint ABI surface has been removed: `ExecutionRequest` no longer carries an unconsumed `EntrypointSelector`, and request-path tests now validate behavior through semantic default-entrypoint compilation only.
- Dead core request input/resolver ABI surface has been removed: `ExecutionRequest` no longer carries unconsumed `inputs` and `resolver` fields (`RuntimeFlow` request-input payload and `ResolverHandle`), with request-path tests updated to the reduced enforced policy set.
- Dead builtin compatibility enum surface has been trimmed: `BuiltinCompatibility::RunMatExtended` was removed from `runmat-builtins` after confirming no production usage, keeping compatibility labels scoped to behavior-backed modes.
- `runmat-gc` stress coverage no longer depends on removed legacy VM/HIR test APIs (`HirProgram`, `runmat_vm::execute`); it now compiles and executes through semantic lowering + MIR + VM bytecode (`lowering.assembly` -> `runmat_mir::lower_assembly` -> `runmat_vm::compile` -> `runmat_vm::interpret`).

## Validation Cadence

For each coherent slice:

- focused regression
- `cargo fmt --all --check`
- `cargo test -p runmat-core --test semicolon_suppression`
- `cargo check --workspace`
- `git diff --check`
- commit as `RM-378: <3-4 word message>`
