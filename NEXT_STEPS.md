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

- Keep Turbine expanded call instructions on interpreter fallback until Turbine has a non-scalar value ABI for cell/comma-list expansion.
- Keep the centralized legacy fallback only for unresolved/external callback identities that are not present in a semantic registry.
- Start object/index descriptor design before replacing the remaining `subsref` / `subsasgn` protocol assembly paths.

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
- Function-handle calls to functions defined in previous interactive inputs execute through `Value::SemanticFunctionHandle` identities before legacy fallback.
- Local zero-capture named function handles now lower directly to `CreateSemanticFunctionHandle`; anonymous and captured function handles remain closures because they need capture/layout metadata.
- Function-handle literals for functions defined in previous interactive inputs now bind to `CreateSemanticFunctionHandle` after session registry attachment, so `@f` carries semantic identity instead of a name-only `FunctionHandle` when the target is known.
- `feval` multi-output calls through previous-input semantic function handles invoke semantic identities before runtime/legacy fallback.
- Expanded `feval` calls through previous-input semantic function handles invoke semantic identities before runtime/legacy fallback.
- Expanded multi-output `feval` calls through previous-input semantic function handles invoke semantic identities before runtime/legacy fallback.
- Multi-output `feval` fallback dispatch now checks the semantic registry before invoking the centralized legacy named fallback.
- `feval` closure dispatch resolves closure names through the semantic registry when an embedded semantic function id is unavailable.
- Runtime `feval` now invokes embedded `Closure.semantic_function` and `Value::SemanticFunctionHandle` identities directly before name fallback.
- Runtime `feval`, `cellfun`, and `arrayfun` now ask a VM-installed semantic name resolver before name-only user-function or builtin fallback, so runtime callback strings/`FunctionHandle(name)` values can still resolve to session semantic functions when the active VM bytecode registry knows the name.
- Local and previous-input user-function calls inside `end` expressions now carry `EndExpr::SemanticCall` identities instead of relying on name recovery.
- Compiler-produced `feval('name', ...)` callees for local/session semantic functions now bind to `CreateSemanticFunctionHandle` before reaching runtime `feval`, including multi-output and expanded-argument forms.
- `cellfun` and `arrayfun` string callback literals for local/session semantic functions now bind to `CreateSemanticFunctionHandle` bytecode before reaching runtime builtins.
- Legacy named user-call bytecode dispatch now checks the semantic registry before builtin fallback or the centralized named legacy fallback.
- Turbine direct `CallSemanticFunction` bytecode now compiles through a semantic host callback by `FunctionId` instead of falling back to the interpreter.
- Turbine named `CallFunction(name)` bytecode now resolves the active `SemanticFunctionRegistry` at compile time and lowers semantic-known names directly by `FunctionId`, before considering legacy-shaped callback definitions.
- Turbine direct and named semantic multi-output calls now compile through a semantic host callback that writes numeric outputs into JIT result slots; expanded call variants still require a future non-scalar argument ABI.
- Multi-output `feval` legacy user fallback is centralized behind one dispatch helper instead of duplicated in direct and expanded `feval` bytecode handlers.
- Named legacy user fallback preparation and compilation is centralized behind crate-private VM dispatch helpers; Turbine now enters through `runmat_vm::legacy::execute_legacy_user_function_isolated` instead of destructuring compiled fallback internals.
- Unresolved/external dynamic user-function callbacks still centralize through crate-private `compile_legacy_named_user_dispatch_fallback`, which wraps a private `compile_legacy_user_dispatch_fallback` over reconstructed `LegacyHirProgram`.
- The raw unresolved/external fallback is private inside VM dispatch so new semantic call paths cannot treat it as normal dispatch infrastructure.

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

- `PreparedLegacyUserCall`, `PreparedLegacyUserDispatch`, `CompiledLegacyUserDispatch`, and `LegacyUserFunction` are transitional compatibility structures. The prepared/compiled dispatch records are crate-private; their long-term replacement is a semantic call descriptor keyed by `FunctionId`/`DefPath` plus layout/capture data.
- The private raw legacy fallback should be treated as the final centralized legacy boundary before removal, not a reusable abstraction to extend.

Current legacy fallback caller inventory:

- Raw fallback compiler boundary: `compile_legacy_user_dispatch_fallback` is private in VM dispatch and is only reached through crate-private `compile_legacy_named_user_dispatch_fallback`.
- Multi-output `feval` fallback: `handle_feval_user_multi_output` first asks `SemanticFunctionRegistry` / `runmat_runtime::user_functions::try_call_semantic_function`; it falls back only when `FevalDispatch::InvokeUser` carries a name that did not resolve semantically.
- Named/expanded user-call fallback: `handle_prepared_user_function_call` first checks the semantic registry, then builtin dispatch, then legacy named fallback. Primary MIR lowering emits `CallSemanticFunction*` for semantic function callees; remaining generic `CallFunction*` emissions are in legacy HIR compiler paths or unresolved/name-only dynamic forms.
- End-expression callback fallback: local/session user-function calls now carry `EndExpr::SemanticCall`; remaining fallback is for unresolved names or values without semantic identity.
- Turbine external callback fallback: direct semantic calls and semantic-known named calls lower by `FunctionId`; the remaining `runmat_call_user_function` name callback invokes semantic functions when the Turbine runtime context registry resolves the name, otherwise it uses `runmat_vm::legacy::execute_legacy_user_function_isolated` for legacy-shaped callback definitions. This is the concrete external compatibility boundary for unresolved/external callback names.
- Dynamic closure/`feval` unresolved fallback: VM `call::feval::execute_feval` and runtime `feval` resolve embedded closure semantic IDs, semantic function handles, and VM-resolved semantic names before name fallback. Compiler-produced `feval` string callees now bind to semantic handles when the registry knows the function, including multi-output and expanded-argument forms. Remaining unresolved closure/function-handle names still gather or ask name-based user-function maps only when the active semantic resolver cannot resolve them.
- Runtime callback builtins: `cellfun` and `arrayfun` invoke embedded semantic closures or semantic function handles directly, and runtime string/name-only callback values now ask the active VM semantic resolver before name fallback. Compiler-produced local/session string callbacks are still rewritten to semantic function handles when possible.

Classification:

- Semantic-first, replaceable next: runtime producers that can be upgraded to semantic handles before crossing the VM/runtime ABI, plus expanded JIT call shapes after Turbine can pass non-scalar cell/comma-list arguments.
- Semantic-first, blocked by callable identity shape: runtime-created strings and plain `Value::FunctionHandle(name)` values where no active semantic resolver is installed or the resolver cannot map the name to a stable semantic identity; compiler/session-produced handles now carry `Value::SemanticFunctionHandle` identity.
- External compatibility boundary: Turbine callbacks whose host context still owns only `LegacyUserFunction` records for unresolved/external exported functions.
- Dead/duplicate raw fallback call sites: none found; the remaining four VM raw fallback calls all sit behind semantic-first checks or the Turbine external boundary.

Completed implementation slices:

- Added a semantic user-function invoker path that executes `SemanticFunctionBytecode` by `FunctionId` with captures, args, and requested outputs.
- Routed VM/runtime callback paths and Turbine direct/named single-output and multi-output call paths through semantic registry identity when the current bytecode product knows the callee.
- Kept the centralized legacy fallback only for unresolved/external dynamic functions until the registry is complete.

### 2. Collapse Legacy HIR Compatibility Seams

Observed older-HIR artifacts worth collapsing:

- `LegacyHirProgram`, `LegacyHirStmt`, and `LegacyHirExpr` remain in crate-private VM compiler modules and many tests.
- The legacy-shaped user-function record and fallback counters are now exposed only under `runmat_vm::legacy::*`; the older root-level `runmat_vm::LegacyUserFunction`, root fallback-counter exports, `runmat_vm::functions`, Turbine public legacy-function execution surfaces, and Turbine public bytecode compiler entrypoints have been removed, and downstream callers should move to semantic function bytecode/registry APIs.
- VM execution internals such as `CallFrame` and `ExecutionContext` are no longer root-level `runmat_vm` exports or bytecode prelude exports; call-stack diagnostics use runtime call-frame types instead.
- Legacy bytecode compilation is no longer exposed through VM public modules; production unresolved callbacks go through crate-private dynamic callback fallback helpers.
- `RunMatSession` keeps `workspace_bindings` to seed workspace variables across REPL inputs, but those bindings are still plain VM slot indices rather than durable semantic workspace binding IDs.
- `LoweringResult` still carries both `assembly` and legacy `hir`, `variables`, `functions`, `var_types`, and legacy inference placeholders.
- LSP analysis still consults legacy variable maps and legacy type helpers.
- `CompatibilityMode::RunMatExtended` is still mapped through parser compatibility as MATLAB mode in some places, which obscures the intended distinction between compatibility policy and parser syntax mode.

Target cleanup direction:

- Keep source compatibility behavior, but represent it through semantic assembly, analysis facts, and workspace ABI records.
- Replace session `workspace_bindings` VM slot mapping with a semantic workspace binding table keyed by stable names plus semantic binding/session IDs.
- Continue replacing remaining legacy function fallback sites with the semantic registry.
- Move tests that only need compiler behavior from hand-built legacy HIR to semantic source fixtures or semantic MIR fixtures.
- Keep remaining legacy dispatch helpers private to unresolved/external callback fallback plumbing until those identities are registry-backed.

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

1. Replace non-external dynamic callback sites with registry lookup or typed semantic lowering.
2. Extend registry-backed lowering beyond direct calls to remaining callable shapes that still need dynamic-name fallback, where layout/capture information is available.
3. Pass semantic callable identity through end-expression callback paths instead of rediscovering names at runtime.
4. Keep `compile_legacy_user_dispatch_fallback` as a fallback only for identities not yet in the semantic registry.
5. Add ratchets that callbacks to functions defined in previous REPL inputs do not call `compile_legacy` when semantic bytecode is available.

Current ratchet status:

- VM basics, matrix-division, bitwise row-vector, and import-error bytecode tests now use the semantic HIR/MIR `compile` path where they only need source-level or semantic bytecode behavior.
- The FFT end-range bytecode assertion in VM basics now uses semantic bytecode instead of `compile_legacy`.
- Simple VM basics execution tests for arithmetic, zero-output builtin calls, and `nextpow2` now run semantic bytecode.
- Additional VM basics execution tests for complex literals, leading-dot numeric forms, elementwise division, `chol`, `uint16`, and `atan2` RHS expressions now run semantic bytecode.
- FFT `end/2` range materialization, complex range assignment, multidimensional `end` ranges, out-of-bounds `end+1`, variable-offset `end`, builtin/user-function-call `end` expressions, pow/leftdiv end-expression combinations, and `fftn`/`ifftn` indexing basics now run semantic bytecode.
- Remaining VM basics legacy execution ratchets cover legacy accel graph shape assertions, legacy multi-output argument bytecode shape, and object range-end protocol payloads.
- Matrix-division execution tests now run semantic bytecode; only the accel graph assertions keep legacy bytecode for legacy graph shape coverage.
- Loop execution tests now run semantic bytecode; only the stochastic-evolution instruction assertion keeps legacy bytecode shape coverage.
- Several VM functions success-path and arity tests now run semantic bytecode, including nested user-function calls, function-handle/cellfun round-trips, `nargin`/`nargout`, fixed-arity/minimum-varargin input errors, fixed-output arity errors, shared input/output names, inline `fprintf`/`sprintf` cast arguments, root/function-output struct materialization, nested/dynamic member assignment, mixed member/cell/index reads, and numeric bitwise array operations.
- VM functions import execution, import ambiguity, and import-shadowing/metaclass guard tests now run semantic bytecode for package builtin and static-method imports.
- VM functions direct cell-expansion, explicit function-return propagation, semantic `varargin` packing, and `feval(@f, varargin{:})` forwarding now run semantic bytecode.
- VM functions fixed-output user-function return propagation through outer user-call arguments, such as `h(g())` and `f(1, g())`, now runs semantic bytecode.
- VM functions `varargout` expansion through builtin and user-call arguments, such as `max(h())` and `f(g())`, now runs semantic bytecode.
- VM functions global and persistent declarations now lower to semantic bytecode workspace effects.
- VM functions classdef registration now lowers to semantic bytecode for property access enforcement.
- VM functions tensor indexing/write ratchets for logical mask assignment, gather/scatter roundtrip, shape broadcasting, column-major RHS mapping, and range/`end` read/write cases now run semantic bytecode.
- VM functions struct `isfield`/`fieldnames` and computed integer column-slice read/write ratchets now run semantic bytecode; string aggregate concatenation and `containers.Map` package calls remain legacy-executed.
- VM functions type-class static `zeros` calls for `double.zeros` and `logical.zeros` now resolve through primitive class metadata and run through semantic bytecode.
- VM functions nested try/catch `rethrow` exception propagation now runs semantic bytecode; `functions.rs` has no ignored tests remaining.
- Operator-overload diagnostic bytecode in VM functions tests now uses semantic compilation instead of `compile_legacy`.
- Additional VM functions object/class/operator/OOP ratchets now run semantic bytecode, including object member get/set and method dispatch, class registration/static/inheritance paths, static classref calls, object cell expansion, operator overload variants, and negative OOP try/catch paths.
- VM range, meshgrid-range, global/persistent, and logical-operator tests now run semantic bytecode.
- VM indexing, matrix-slicing, and lvalue-assignment tests now run semantic bytecode.
- VM cell-array, exception, datetime, and most closure/callback tests now run semantic bytecode; captured and nested closure capture tests remain legacy-executed pending semantic capture propagation.
- VM control-flow tests now mostly run semantic bytecode; unknown-builtin catch behavior and varargout mismatch diagnostics remain legacy-executed semantic gaps.
- VM multidimensional indexing tests now run semantic bytecode for logical row selection and slice assignment cases; 3D indexing/slicing cases remain legacy-executed pending semantic N-D selector support.
- VM indexing-property tests now run semantic bytecode for scalar/logical/range write broadcasts, end-arithmetic stores, negative-step linear indexing, roundtrip scatter, fastpath broadcasts, and simple cell expansion; advanced N-D empty/vector selector and expansion cases remain legacy-executed.
- Remaining `functions.rs` legacy execution sites cover semantic gaps for varargout mismatch diagnostics, struct-field vector/range indexing through member reads, test-class constructor resolution, metaclass postfix member/method lowering, dependent property backing behavior, `containers.Map` package calls, and string aggregate concatenation.
- Turbine mixed arithmetic/function-call, simple scalar function compilation, scalar callback variable-isolation, compute-intensive scalar callback, nested scalar callback, and error-handling success-path coverage now use semantic registry-backed named calls instead of hand-built legacy function metadata.
- Remaining production legacy compilation usage is centralized behind `compile_legacy_user_dispatch_fallback`; remaining legacy-shaped test fixtures are tied to unresolved/external callback fallback coverage, native-accel graph construction, or semantic gaps called out above.

## Validation Cadence

For each coherent slice:

- focused regression
- `cargo fmt --all --check`
- `cargo test -p runmat-core --test semicolon_suppression`
- `cargo check --workspace`
- `git diff --check`
- commit as `RM-378: <3-4 word message>`
