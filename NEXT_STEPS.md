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

Start with MIR indexing/assignment classification.

Concrete first target:

- Inspect MIR place/index lowering for assignments.
- Ensure range/vector/logical paren assignments lower to slice assignment intentionally.
- Add regression tests that assert semantic bytecode uses slice store paths for `A(2:3) = ...` and similar cases.
- Remove or narrow the VM `StoreIndex` fallback logic added to reinterpret tensor-valued indices.

This directly addresses the most common band-aid pattern from the recent work.

## Follow-Up Slice

After indexing assignment classification, target remaining hard-coded function-handle and dynamic-call dispatch names.

Concrete next target:

- Keep `CreateFunctionHandle` as the typed function-handle lowering path and remove remaining legacy string-era assumptions around handle targets.
- Replace remaining `feval` string-era fallback paths with explicit dynamic-call lowering and resolver facts.
- Preserve semantic closure behavior and existing function-handle ratchets.

This is narrower than object `subsref` / `subsasgn` and should reduce compiler/runtime string coupling without requiring a full object protocol rewrite.

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
- Function-handle calls to functions defined in previous interactive inputs execute through the session semantic registry before legacy fallback.
- `feval` multi-output calls through previous-input function handles resolve names through the session semantic registry before runtime/legacy fallback.
- Expanded `feval` calls through previous-input function handles resolve names through the session semantic registry before runtime/legacy fallback.
- Expanded multi-output `feval` calls through previous-input function handles resolve names through the session semantic registry before runtime/legacy fallback.
- `feval` closure dispatch resolves closure names through the semantic registry when an embedded semantic function id is unavailable.
- Legacy named user-call bytecode dispatch now checks the semantic registry before builtin fallback or `compile_legacy_user_dispatch_fallback`.
- Multi-output `feval` legacy user fallback is centralized behind one dispatch helper instead of duplicated in direct and expanded `feval` bytecode handlers.
- Named legacy user fallback preparation and compilation is centralized behind `compile_legacy_named_user_dispatch_fallback` for VM dispatch, callback runner, and Turbine fallback sites.
- Unresolved/external dynamic user-function callbacks still centralize through `compile_legacy_user_dispatch_fallback`, which wraps `compile_legacy` over reconstructed `LegacyHirProgram`.
- The centralized unresolved/external fallback is explicitly named `compile_legacy_user_dispatch_fallback` so new semantic call paths do not treat it as normal dispatch infrastructure.

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

- `PreparedUserCall`, `PreparedUserDispatch`, and `UserFunction` should become transitional compatibility structures. Their long-term replacement is a semantic call descriptor keyed by `FunctionId`/`DefPath` plus layout/capture data.
- `compile_legacy_user_dispatch_fallback` should be treated as the final centralized legacy boundary before removal, not a reusable abstraction to extend.

First implementation slice:

- Add a semantic user-function invoker path that can execute a `SemanticFunctionBytecode` by `FunctionId` with captures, args, and requested outputs.
- Route one dynamic callback site through that path when the callee maps to a semantic function already present in the current bytecode product.
- Keep the centralized legacy fallback only for unresolved/external dynamic functions until the registry is complete.

### 2. Collapse Legacy HIR Compatibility Seams

Observed older-HIR artifacts worth collapsing:

- `LegacyHirProgram`, `LegacyHirStmt`, and `LegacyHirExpr` remain in VM compiler modules and many tests.
- `compile_legacy` is no longer re-exported from the `runmat-vm` crate root or `runmat_vm::bytecode`; direct test users go through `runmat_vm::bytecode::compile::compile_legacy` and production unresolved callbacks go through the centralized dynamic callback fallback.
- `RunMatSession` keeps `workspace_bindings` to seed workspace variables across REPL inputs, but those bindings are still plain VM slot indices rather than durable semantic workspace binding IDs.
- `LoweringResult` still carries both `assembly` and legacy `hir`, `variables`, `functions`, `var_types`, and legacy inference placeholders.
- LSP analysis still consults legacy variable maps and legacy type helpers.
- `CompatibilityMode::RunMatExtended` is still mapped through parser compatibility as MATLAB mode in some places, which obscures the intended distinction between compatibility policy and parser syntax mode.

Target cleanup direction:

- Keep source compatibility behavior, but represent it through semantic assembly, analysis facts, and workspace ABI records.
- Replace session `workspace_bindings` VM slot mapping with a semantic workspace binding table keyed by stable names plus semantic binding/session IDs.
- Continue replacing remaining legacy function fallback sites with the semantic registry.
- Move tests that only need compiler behavior from hand-built legacy HIR to semantic source fixtures or semantic MIR fixtures.
- Stop exposing `bytecode::compile::compile_legacy` once runtime/Turbine callbacks and remaining tests no longer depend on it.

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
- Object protocol calls still assemble `subsref`/`subsasgn` style data in VM/runtime helper paths.

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

### 6. Varargout And Multi-Output Semantics

Current state:

- Multi-output direct, dynamic, and expanded calls are typed in bytecode.
- `MirOutputTarget::VarargoutExpansion` still has a bytecode gap, but it may not be constructed by current semantic lowering.

Design decision needed:

- If varargout target expansion is not representable from current parser/HIR, mark it future design and do not keep a misleading runtime gap.
- If it is intended soon, define the target-list ABI:
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

### 9. Compatibility Mode Cleanup

Current state:

- Compatibility policy, parser mode, and RunMat extension flags are still partly entangled.

Target state:

- Parser mode answers syntax acceptance questions.
- Semantic compatibility mode answers behavior policy questions.
- Runtime/session mode answers host policy questions such as top-level await and workspace export.

Collapse opportunity:

- Replace broad variants such as `RunMatExtended` where they mask multiple independent policy bits.

## Remaining Gap Classification

Treat current MIR bytecode gap markers as follows:

- `control-flow terminator`: design gap for async/await or future terminators, not a small VM patch unless a concrete source reproducer exists.
- `varargout expansion`: HIR/MIR output-target design gap; first confirm whether it is constructible.
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

The next high-leverage slice is replacing the remaining dynamic callback fallbacks with the session semantic registry, not another isolated callback patch.

Concrete plan:

1. Inventory remaining `compile_legacy_user_dispatch_fallback` sites and classify which ones are true unresolved/external fallback.
2. Replace non-external dynamic callback sites with registry lookup or typed semantic lowering.
3. Extend registry-backed lowering beyond direct calls to remaining callable shapes that still need dynamic-name fallback, where layout/capture information is available.
4. Keep `compile_legacy_user_dispatch_fallback` as a fallback only for identities not yet in the semantic registry.
5. Add ratchets that callbacks to functions defined in previous REPL inputs do not call `compile_legacy` when semantic bytecode is available.

Current ratchet status:

- VM basics, matrix-division, bitwise row-vector, and import-error bytecode tests now use the semantic HIR/MIR `compile` path where they only need source-level or semantic bytecode behavior.
- The FFT end-range bytecode assertion in VM basics now uses semantic bytecode instead of `compile_legacy`.
- Matrix-division execution tests now run semantic bytecode; only the accel graph assertions keep legacy bytecode for legacy graph shape coverage.
- Loop execution tests now run semantic bytecode; only the stochastic-evolution instruction assertion keeps legacy bytecode shape coverage.
- Operator-overload diagnostic bytecode in VM functions tests now uses semantic compilation instead of `compile_legacy`.
- Remaining test `compile_legacy` references are still tied to legacy execution helpers, native-accel graph construction, legacy multi-output bytecode shape assertions, or Turbine/accelerate legacy suites.
- Remaining production `compile_legacy` usage is centralized behind `compile_legacy_user_dispatch_fallback`; the remaining transitional API is `runmat_vm::bytecode::compile::compile_legacy` for legacy tests and fallback plumbing.

## Validation Cadence

For each coherent slice:

- focused regression
- `cargo fmt --all --check`
- `cargo test -p runmat-core --test semicolon_suppression`
- `cargo check --workspace`
- `git diff --check`
- commit as `RM-378: <3-4 word message>`
