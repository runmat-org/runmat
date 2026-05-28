# Runtime And Workspace ABI Design

## Purpose

This document defines the target runtime/workspace ABI for the semantic compiler architecture described in `TARGET_MODEL.md`.

The key design constraint is that RunMat must keep MATLAB-compatible source behavior while removing legacy compiler/runtime paths. Compatibility behavior may remain, but it must be represented through semantic HIR, MIR, analysis facts, VM layout, runtime services, and structured diagnostics. It must not depend on legacy `VarId`, legacy HIR statements, string-keyed function reconstruction, or bytecode/source heuristics.

## Core Principle

The ABI is a set of explicit boundaries between identity layers.

- HIR IDs such as `BindingId`, `FunctionId`, `ExprId`, and `StmtId` are local compiler-product identities.
- MIR IDs such as local IDs and block IDs are derived analysis/codegen identities.
- VM slots, frame indexes, storage handles, and capture slots are runtime layout identities.
- `DefPath`, source identity, package identity, and session workspace keys are stable boundary identities.
- Runtime `Value`s, provider handles, object handles, futures, tasks, and workspace values are execution identities.

Compiler-local IDs are allowed only inside one compiler product. Anything crossing into session state, caches, dynamic lookup, snapshots, LSP identity, runtime persistence, or workspace storage must use a stable identity layer.

## Grounding Examples

The rest of this document is grounded in two minimal complete user projects. They share the same MATLAB-shaped program so we can distinguish project/entrypoint/runtime policy from source semantics, and they give us a stable reference whenever compatibility mode changes behavior.

RunMat supports two entrypoint styles:

- Source-path entrypoints: `runmat ./analyze_sales.m`. This is the default CLI behavior and does not require project configuration.
- Named project entrypoints: `runmat run analyze` or equivalent host selection. These are optional project-declared selectors over `HirEntrypoint`, not a separate Rust-like `main` function taxonomy.

Named entrypoints are useful for projects, reproducible workflows, notebooks, services, and packaged applications. They should not replace MATLAB-compatible source-path execution.

### Example A: Default RunMat Project

This project uses default RunMat behavior. The config shows defaults as comments so the file is self-documenting. Omitting the file entirely should still allow `runmat ./analyze_sales.m` source-path execution with equivalent defaults where no package metadata is needed.

Project layout:

```text
demo-runmat/
  runmat.toml
  analyze_sales.m
  normalizeRows.m
  private/
    localScale.m
  +stats/
    summarize.m
  @Report/
    Report.m
```

`runmat.toml`:

```toml
[package]
name = "demo_runmat" # optional; defaults to directory-derived package identity

[sources]
roots = ["."] # optional; default is ["."]

[runtime]
compatibility = "runmat" # optional; this is the default RunMat mode
top_level_await = true             # optional in RunMat/interactive hosts where supported

# Named entrypoints are optional. Source-path execution still works without this section.
[[entrypoints]]
name = "analyze"
path = "analyze_sales.m"
kind = "script"                    # optional if inferable from source
workspace_export = "top-level"      # optional for scripts; default is top-level export
```

Run commands:

```bash
cd demo-runmat
runmat ./analyze_sales.m   # source-path entrypoint
runmat run analyze         # named project entrypoint, optional
```

### Example B: MATLAB-Compatible Project

This project pins MATLAB-compatible behavior. The source files are intentionally the same as the default RunMat example so differences are attributable to mode and host policy rather than different program structure.

Project layout:

```text
demo-matlab-compat/
  runmat.toml
  analyze_sales.m
  normalizeRows.m
  private/
    localScale.m
  +stats/
    summarize.m
  @Report/
    Report.m
```

`runmat.toml`:

```toml
[package]
name = "demo_matlab_compat" # optional; defaults to directory-derived package identity

[sources]
roots = ["."] # optional; default is ["."]

[runtime]
compatibility = "matlab-strict" # explicit; rejects RunMat-only syntax where possible
top_level_await = false          # optional; default for MATLAB-strict source execution

# Optional named entrypoint. MATLAB-compatible source-path execution still works without this.
[[entrypoints]]
name = "analyze"
path = "analyze_sales.m"
kind = "script"                   # optional if inferable from source
workspace_export = "top-level"     # optional for scripts; default is top-level export
```

Run commands:

```bash
cd demo-matlab-compat
runmat ./analyze_sales.m   # source-path entrypoint
runmat run analyze         # named project entrypoint, optional
```

Shared `analyze_sales.m`:

```matlab
load sales.mat revenue expenses

data = [revenue expenses];
scaled = localScale(data);
normalized = normalizeRows(scaled);

[totals, averages] = stats.summarize(normalized);

report = Report("sales", totals);
headline = report.title();

cells = {totals, averages, headline};
[summaryTotals, summaryAverages, summaryTitle] = cells{:};

formatter = @(x) x + 1;
adjusted = feval(formatter, summaryTotals);

clear expenses
cd results
plot(adjusted)
```

Shared `normalizeRows.m`:

```matlab
function out = normalizeRows(x)
    rowTotals = sum(x, 2);
    out = x ./ rowTotals;
end
```

Shared `private/localScale.m`:

```matlab
function y = localScale(x)
    y = x * 100;
end
```

Shared `+stats/summarize.m`:

```matlab
function [totals, averages] = summarize(x)
    totals = sum(x, 1);
    averages = mean(x, 1);
end
```

Shared `@Report/Report.m`:

```matlab
classdef Report
    properties
        Name
        Totals
    end

    methods
        function obj = Report(name, totals)
            obj.Name = name;
            obj.Totals = totals;
        end

        function title = title(obj)
            title = ["Report: " + obj.Name];
        end
    end
end
```

The walkthrough at the bottom uses the MATLAB-compatible project for conservative grounding:

```bash
cd demo-matlab-compat
runmat ./analyze_sales.m
```

When the design needs to explain mode differences, use these two projects as references:

- In `demo-runmat`, omitted compatibility config defaults to RunMat behavior, and RunMat syntax such as future async constructs may be accepted according to feature support.
- In `demo-matlab-compat`, `compatibility = "matlab-strict"` rejects RunMat-only syntax and disables top-level await by default.
- In both projects, `runmat ./analyze_sales.m` is source-path execution, while `runmat run analyze` selects the optional named entrypoint.

Expected high-level behavior:

- `analyze_sales.m` is a script-like source-path entrypoint.
- `load sales.mat revenue expenses` introduces top-level workspace-visible bindings through an explicit workspace effect.
- `localScale` resolves to `private/localScale.m` relative to `analyze_sales.m`.
- `normalizeRows` resolves to the sibling function file.
- `stats.summarize` resolves to the package function `+stats/summarize.m` and is requested for exactly two outputs.
- `Report("sales", totals)` resolves to a class constructor in `@Report/Report.m`.
- `report.title()` resolves to method dispatch on the `Report` class.
- `cells{:}` produces a comma-list consumed by the multi-assignment target list.
- `formatter` is an anonymous function handle with capture/runtime handle semantics.
- `feval(formatter, summaryTotals)` executes through the normal function-handle call ABI.
- `clear expenses` is an explicit workspace effect.
- `cd results` is an explicit environment effect and resolver invalidation point.
- `plot(adjusted)` is a language-synchronous builtin call that may internally suspend through host/provider integration.
- On completion, script top-level visible bindings export according to entrypoint policy and initialization facts; function locals from `normalizeRows`, `localScale`, `stats.summarize`, and `Report.title` do not leak into the ordinary workspace.

## Compiler Product

A compiled unit is not just bytecode. The runtime receives a product that preserves semantic identity, normalized execution form, analysis facts, layout, and source mappings.

```rust
pub struct CompilerProduct {
    pub hir: HirAssembly,
    pub mir: MirAssembly,
    pub analysis: AnalysisStore,
    pub layout: VmAssemblyLayout,
    pub bytecode: Bytecode,
    pub source_maps: SourceMapBundle,
}
```

Rules:

- `HirAssembly` is the semantic/source truth for modules, functions, classes, bindings, and entrypoints.
- `MirAssembly` is normalized control flow, dataflow, effect, capture, async, place, and call structure.
- `AnalysisStore` contains facts keyed by semantic IDs and MIR IDs.
- `VmAssemblyLayout` maps semantic bindings and MIR locals to concrete runtime slots/storage.
- `Bytecode` is derived from MIR plus layout and selected analysis facts.
- `SourceMapBundle` maps runtime locations back to source/HIR/MIR for diagnostics, call stacks, profiling, async resumes, and LSP.

## Execution Boundary

Execution begins from an explicit request and returns a structured outcome. Session, CLI, kernel, desktop, LSP tooling, and WASM hosts should consume this boundary instead of inspecting HIR or bytecode to infer behavior.

```rust
pub struct ExecutionRequest {
    pub source: SourceInput,
    pub entrypoint: EntrypointSelector,
    pub compatibility: CompatibilityMode,
    pub host_policy: HostExecutionPolicy,
    pub inputs: RuntimeFlow,
    pub requested_outputs: RequestedOutputCount,
    pub workspace: WorkspaceHandle,
    pub resolver: ResolverHandle,
}
```

```rust
pub struct ExecutionOutcome {
    pub flow: RuntimeFlow,
    pub workspace_delta: WorkspaceDelta,
    pub display_events: Vec<DisplayEvent>,
    pub diagnostics: Vec<Diagnostic>,
    pub effects: Vec<ObservedEffect>,
    pub suspension: Option<Suspension>,
    pub profiling: ExecutionProfile,
}
```

Rules:

- Compatibility mode is a first-class input to parsing, lowering, diagnostics, entrypoint policy, runtime behavior, and display/workspace policy.
- Execution starts from `HirEntrypoint`, not from raw statement lists.
- Runtime outcome is structured. It is not inferred from console output, final stack state, legacy statements, or last bytecode instructions.
- `ExecutionOutcome` must be preserved by interpreter, JIT fallback, async execution, CLI, kernel, and host integrations.

## Runtime Flow

MATLAB has transient list-valued flows that are not durable values. Runtime ABI must distinguish ordinary values from output lists and comma-separated lists.

```rust
pub enum RuntimeFlow {
    NoValue,
    Single(Value),
    OutputList(Vec<Value>),
    CommaList(Vec<Value>),
    DynamicList(DynamicListHandle),
}
```

Rules:

- Function calls produce output-list flows according to requested output count.
- `varargin{:}`, `varargout{:}`, cell expansion, and struct-array field access may produce comma-list flows.
- Function argument lists and assignment target lists consume comma lists.
- Scalar expression contexts reject or collapse list-valued flows according to MATLAB-compatible rules.
- Workspace values are durable `Value`s. A comma-list never becomes a stored workspace value directly.

## Function And Call ABI

All callable targets use one call ABI: user functions, builtins, constructors, methods, function handles, `feval`, dynamic names, and unresolved runtime fallback.

```rust
pub struct FunctionCallRequest {
    pub callee: CallableTarget,
    pub args: RuntimeFlow,
    pub requested_outputs: RequestedOutputCount,
    pub call_site: Span,
}
```

```rust
pub enum CallableTarget {
    Function(FunctionId),
    Builtin(BuiltinId),
    ClassConstructor(ClassId),
    Method(MethodId),
    FunctionHandle(FunctionHandleId),
    DynamicName(SymbolName),
    Unresolved(QualifiedName),
}
```

```rust
pub struct VmCallFrame {
    pub function: FunctionId,
    pub layout: VmFunctionLayout,
    pub slots: Vec<Value>,
    pub nargin: usize,
    pub nargout: RequestedOutputCount,
    pub captures: CaptureEnvironment,
}
```

Rules:

- Fixed inputs bind through `FunctionAbi.fixed_inputs`.
- Excess inputs pack into `varargin` when present.
- Missing required inputs are diagnosed through ABI validation.
- Fixed outputs are read from `FunctionAbi.fixed_outputs`.
- Excess requested outputs are read from `varargout` when present.
- `nargin` and `nargout` are implicit read-only function-local bindings backed by ABI slots.
- An output that shares a source name with an input reuses the same semantic `BindingId`.
- Builtin dispatch receives requested output count and may be output-count-sensitive.
- Direct user calls use semantic function identity, not string names.
- Dynamic and unresolved calls go through resolver runtime explicitly.

## Turbine Value ABI

Turbine's optimizing tier must not depend on f64-only slots for runtime calls. Numeric f64 slots remain an optimization for straight-line scalar code, but any host ABI that can cross semantic calls, expanded arguments, cells, tensors, strings, objects, or output lists uses an explicit tagged value slot.

```rust
#[repr(u32)]
pub enum TurbineValueTag {
    Empty = 0,
    Num = 1,
    Bool = 2,
    Int = 3,
    GcHandle = 4,
}

#[repr(C)]
pub struct TurbineValue {
    pub tag: TurbineValueTag,
    pub reserved: u32,
    pub payload: u64,
}
```

Rules:

- `Num` stores `f64::to_bits()` in `payload`.
- `Bool` stores `0` or `1` in `payload`.
- `Int` stores signed integer bits in `payload`; wider exact integer tagging can extend the reserved field without changing the slot size.
- `GcHandle` stores a handle/pointer to a GC-managed runtime `Value` for tensors, cells, strings, objects, handles, closures, and output lists.
- Semantic call host callbacks that can receive non-scalar values take `TurbineValue[]` arguments and write `TurbineValue[]` outputs.
- The durable semantic host callbacks are `runmat_call_semantic_function_value` for one output and `runmat_call_semantic_function_values` for requested-output vectors.
- Expanded call lowering must compile to this ABI, not to interpreter fallback or name-only legacy dispatch.
- The current f64 host callbacks remain scalar fast paths and may be replaced or specialized once `TurbineValue` call paths are available.

## Function Handles

Function handles carry callable identity and optional closure environment.

```rust
pub enum FunctionHandleTarget {
    Function(FunctionId),
    Builtin(BuiltinId),
    Method(MethodId),
    Anonymous(FunctionId),
    DefPath(DefPath),
    DynamicName(SymbolName),
}
```

```rust
pub struct FunctionHandleValue {
    pub target: FunctionHandleTarget,
    pub captures: CaptureEnvironment,
    pub spawn_safety: SpawnSafetyFact,
}
```

Rules:

- Anonymous function handles reference real `HirFunction`s.
- Captures use the ordinary capture model and do not duplicate semantic bindings.
- `feval`, `arrayfun`, `cellfun`, `str2func`, and `func2str` operate on this identity model.
- Function handles are spawn-safe only when their target and captured environment are spawn-safe.

## Evaluation Context

MATLAB semantics are context-sensitive. Evaluation context must be represented during lowering and preserved as needed into MIR/runtime operations.

```rust
pub enum EvaluationContext {
    Statement,
    Expression,
    AssignmentRhs { targets: OutputTargetList },
    FunctionArgument,
    ConcatElement,
    CellElement,
    IndexOperand,
    CommandArgument,
    NameValueArgument,
    LineSpecArgument,
    DynamicFieldName,
    ForRange,
    Condition,
    SwitchExpression,
    CaseExpression,
    ReturnValue { requested_outputs: RequestedOutputCount },
}
```

Rules:

- Bracket syntax in statement context may be assignment target syntax; in expression context it may be concatenation.
- Command syntax is valid only in statement context and lowers to ordinary call semantics with command-argument rules.
- `ans` assignment policy is determined by source statement context and entrypoint policy.
- `foo(x)` resolution depends on whether `foo` resolves to a value binding, function, builtin, class, package item, dynamic target, or unresolved name.
- `[]` may be an ordinary empty value, a concatenation identity, or a deletion marker depending on context.
- Conditions, switch/case expressions, and for-ranges use MATLAB-compatible semantics and are not generic expression contexts.
- Name-value and line-spec behavior is driven by builtin or method metadata.

## Places And Mutation

Mutation is semantic. VM stores are derived from places plus layout metadata.

```rust
pub enum RuntimePlace {
    Slot(VmSlotId),
    Global(StorageHandle),
    Persistent(StorageHandle),
    Capture(CaptureSlotId),
    Indexed {
        base: Box<RuntimePlace>,
        indexing: IndexingSemantics,
    },
    Field {
        base: Box<RuntimePlace>,
        field: MemberName,
    },
}
```

```rust
pub struct PlaceMutationRuntimeOp {
    pub place: RuntimePlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
    pub value: RuntimeFlow,
}
```

Rules:

- Plain assignment may create a binding only where source/entrypoint policy permits it.
- Indexed assignment may grow arrays where MATLAB-compatible.
- Field assignment may create struct fields and intermediate structs where MATLAB-compatible.
- Assignment of `[]` through an indexed place is deletion, not ordinary value assignment.
- Slice assignment enforces MATLAB shape rules and scalar/implicit expansion.
- Object mutation may dispatch through class metadata such as `subsasgn`.
- Handle-object and value-object mutation behavior is class metadata.

## Workspace Identity And Export

Workspace export is a two-stage process. During one execution, exports are derived from local compiler IDs and frame layout. Persistent workspace state uses stable workspace keys.

```rust
pub struct FrameExport {
    pub binding: BindingId,
    pub key: WorkspaceBindingKey,
    pub value: Value,
    pub visibility: WorkspaceVisibility,
    pub initialized: InitFact,
}
```

```rust
pub enum WorkspaceBindingKey {
    Interactive {
        session: SessionId,
        name: BindingName,
    },
    SourceBinding {
        source: SourceIdentity,
        def_path: DefPath,
        binding: BindingName,
    },
    Global {
        scope: GlobalScopeKey,
        name: BindingName,
    },
    Persistent {
        function: DefPath,
        name: BindingName,
    },
}
```

```rust
pub struct WorkspaceDelta {
    pub upserts: Vec<WorkspaceBindingValue>,
    pub removals: Vec<WorkspaceBindingKey>,
    pub full_snapshot_required: bool,
}
```

Rules:

- `BindingId` is local to one `HirAssembly` and cannot be the durable session workspace key.
- `WorkspaceVisibility` determines export eligibility.
- `WorkspaceExportPolicy` on the selected entrypoint determines what is actually exported.
- `TopLevel` bindings may export from script, REPL, or notebook entrypoints.
- Ordinary function params, outputs, locals, temporaries, and nested locals do not enter ordinary workspace inspection.
- `ImplicitAns` exports only when implicit-result behavior creates it.
- `Hidden` bindings never export.
- Globals and persistents use storage handles and stable global/persistent workspace keys.
- Workspace export must honor initialization facts and runtime assigned bits.
- Failed execution applies only committed effects according to transaction policy.

## Initialization Facts

Workspace export and read diagnostics require explicit initialization state.

```rust
pub enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
```

```rust
pub struct WorkspaceExportCandidate {
    pub binding: BindingId,
    pub key: WorkspaceBindingKey,
    pub slot: VmSlotId,
    pub visibility: WorkspaceVisibility,
    pub init: InitFact,
}
```

Rules:

- `DefinitelyAssigned` exports or updates normally.
- `MaybeAssigned` exports only if runtime frame assigned-bit tracking confirms assignment on this path.
- `Unassigned` does not export.
- Reads of unassigned values produce structured diagnostics.
- Runtime frames must not initialize phantom workspace values solely because a slot exists.

## Display, Public Result, And `ans`

Display/result behavior is source policy lowered into HIR/MIR/runtime operations. It is not inferred from console output, final stack state, legacy statements, or bytecode suffixes.

```rust
pub enum StatementResultPolicy {
    Suppressed,
    DisplayValue { label: DisplayLabel },
    AssignDisplay { binding: BindingId, label: DisplayLabel },
    ImplicitAns { binding: BindingId },
    None,
}
```

```rust
pub enum DisplayEvent {
    Value {
        label: DisplayLabel,
        value: Value,
        span: Span,
    },
    Warning(Diagnostic),
}
```

Rules:

- Semicolon suppression is a source-level statement policy.
- Assignment display and expression display are explicit events.
- Public execution result is separate from display.
- `ans` is a semantic implicit binding backed by layout/runtime state.
- Console output is never used to infer result values.

## Workspace And Environment Effects

Workspace and environment mutation are explicit operations and analysis barriers.

```rust
pub enum WorkspaceEffectOp {
    Load {
        source: LoadSource,
        target: WorkspaceTarget,
    },
    Clear {
        selector: ClearSelector,
    },
    AssignIn {
        scope: WorkspaceScope,
        name: BindingName,
        value: RuntimeFlow,
    },
    EvalIn {
        scope: WorkspaceScope,
        source: Value,
    },
    DeclareGlobal {
        binding: BindingId,
    },
    DeclarePersistent {
        binding: BindingId,
    },
}
```

```rust
pub enum EnvironmentEffectOp {
    ChangeDirectory(PathValue),
    MutatePath(PathMutation),
    ClearFunctionCache,
    ClearClassCache,
    InvalidateDynamicLookup,
}
```

Rules:

- `load`, `clear`, `assignin`, `evalin`, `eval`, `global`, and `persistent` cannot be hidden ordinary builtin side effects.
- `cd`, `addpath`, `rmpath`, `path`, and `rehash` are environment effects and may invalidate resolver products.
- Effects are barriers for fusion, JIT, caching, parallel execution, and reordering unless proven safe.
- Runtime applies effects through session/runtime services and records observed effects in `ExecutionOutcome`.

## Resolver Runtime

Static resolver products and runtime resolver services are related but distinct. Static resolution classifies what is known at compile time. Runtime resolver services handle dynamic lookup, path-dependent behavior, invalidation, and MATLAB-compatible introspection.

```rust
pub trait ResolverRuntime {
    fn lookup_dynamic(&self, query: LookupQuery) -> LookupResult;
    fn which(&self, query: SymbolName) -> WhichResult;
    fn exist(&self, query: SymbolName, kind: ExistKind) -> ExistResult;
    fn invalidate(&self, reason: LookupInvalidation);
}
```

Rules:

- Local functions, nested functions, package functions, class constructors, imports, private functions, builtins, runtime classes, and path functions have explicit resolution precedence.
- Dynamic fallback is explicit and distinct from static resolution.
- `which`, `exist`, `feval`, `str2func`, dynamic calls, and unresolved names use resolver runtime where static resolution cannot prove a target.
- `addpath`, `rmpath`, `cd`, `clear functions`, class cache invalidation, and project changes invalidate relevant resolver products.
- Source/project-aware results should be used for runtime behavior, diagnostics, LSP, and CLI reporting.

## Class Dispatch Runtime

Class and object behavior uses nominal metadata and dispatch services rather than parser/runtime special cases.

```rust
pub trait ClassDispatchRuntime {
    fn construct(
        &self,
        class: ClassId,
        args: RuntimeFlow,
        nargout: RequestedOutputCount,
    ) -> CallResult;

    fn call_method(
        &self,
        receiver: Value,
        method: MethodId,
        args: RuntimeFlow,
        nargout: RequestedOutputCount,
    ) -> CallResult;

    fn get_property(&self, receiver: Value, property: MemberName) -> RuntimeFlow;
    fn set_property(&self, receiver: Value, property: MemberName, value: Value) -> Value;
    fn subsref(&self, receiver: Value, indexing: IndexingSemantics) -> RuntimeFlow;
    fn subsasgn(&self, receiver: Value, mutation: PlaceMutationRuntimeOp) -> Value;
    fn resolve_end(&self, receiver: Value, dim: usize, rank: usize) -> Value;
}
```

Rules:

- Constructors, methods, static methods, static properties, properties, dependent properties, operator overloads, indexing overloads, `subsref`, `subsasgn`, and object `end` dispatch use nominal metadata.
- Object values carry class identity.
- Structs remain structural; classes are nominal.
- Value-object and handle-object mutation semantics are class metadata.
- `class`, `isa`, `metaclass`, `properties`, and `methods` query nominal metadata.
- Runtime-defined domains such as graphics handles, GPU arrays, data objects, tables, and future domain objects should use the same nominal dispatch surface when they need object behavior.

## Analysis Store

Analysis facts are first-class compiler products. VM/runtime lowering should consume analysis products rather than reconstructing source semantics from bytecode.

```rust
pub struct AnalysisStore {
    pub binding_facts: HashMap<BindingId, BindingFact>,
    pub expr_facts: HashMap<ExprId, ExprFact>,
    pub function_summaries: HashMap<FunctionId, FunctionSummary>,
    pub mir_facts: MirAnalysisFacts,
}
```

Facts include:

- type facts
- shape facts
- value-flow facts
- initialization facts
- workspace and environment effects
- async behavior
- async value facts
- spawn safety
- capture reads/writes
- function summaries
- class/member summaries
- fusibility
- parallel safety
- acceleration eligibility

Rules:

- Analysis facts are keyed by semantic IDs where possible.
- MIR-local facts are keyed by MIR IDs where facts are purely MIR-local.
- `runmat_builtins::Type` may remain a compatibility projection during migration, but it is not the long-term canonical analysis fact model.
- Runtime/provider residency is not a static type or shape fact.

## Source Maps And Diagnostics

Runtime and tooling must share source mappings and diagnostic structures.

```rust
pub struct SourceMapBundle {
    pub hir_spans: HashMap<StmtId, Span>,
    pub expr_spans: HashMap<ExprId, Span>,
    pub mir_to_hir: HashMap<MirLocation, HirLocation>,
    pub bytecode_to_mir: HashMap<usize, MirLocation>,
}
```

```rust
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub primary_span: Span,
    pub secondary_spans: Vec<LabeledSpan>,
    pub message: String,
    pub notes: Vec<String>,
    pub help: Option<String>,
    pub category: DiagnosticCategory,
    pub source_context: SourceContext,
}
```

Rules:

- Parser, lowering, semantic analysis, MIR lowering, VM compilation, runtime, LSP, CLI, kernel, and desktop diagnostics use the same diagnostic model.
- Diagnostics present source/project concepts, not internal slot/layout details.
- Runtime errors are diagnostic-backed and include source/module/package context where possible.
- Call stacks, async resumes, profiling, and debugger views use the source map bundle.

## Async And Suspension

Async is explicit execution semantics. It is not an incidental callback path.

```rust
pub struct Suspension {
    pub task: TaskId,
    pub frame: FrameId,
    pub resume_point: ResumePoint,
    pub rooted_values: RootSet,
    pub pending: PendingOperation,
    pub cancellation: CancellationState,
}
```

Rules:

- Async function/block creation is lazy and does not run user code until awaited or spawned.
- `await` polls in the current execution context and may use ordinary lexical captures.
- `spawn` schedules concurrent execution and requires spawn-safe captures and values.
- Spawned tasks must not alias mutable parent frame storage.
- Live locals, temporaries, stack values, captures, provider handles, and object handles across await are rooted explicitly.
- Cancellation/drop releases frame roots and provider resources cooperatively.
- Language-synchronous builtins may internally suspend while still returning ordinary values to sequential code.
- JIT either supports suspension correctly or refuses/falls back before async regions.

## Runtime Residency And Materialization

Language type/shape facts are separate from concrete runtime residency.

```rust
pub enum MaterializationPolicy {
    MetadataOnly,
    Preview { limit: usize },
    HostValue,
    PreserveProvider,
}
```

```rust
pub struct WorkspaceValue {
    pub value: Value,
    pub metadata: ValueMetadata,
    pub residency: RuntimeResidency,
}
```

Rules:

- `Value::Tensor` and provider-backed values such as GPU tensors are runtime data representations.
- Type facts and shape facts do not encode concrete provider placement.
- Workspace inspection should not force provider materialization unless host policy requests it.
- Metadata previews may avoid downloading large provider-backed values.
- `gather`, plotting, serialization, and explicit host policies choose materialization.
- Runtime/provider planners own concrete placement, promotion, download, buffer reuse, and dispatch decisions.

## Caching And Incrementality

Cache keys use stable identities and content/config dependencies, not local arena IDs alone.

Cache inputs include:

- package identity
- module path
- source content hash
- manifest/project config hash
- dependency graph hash
- compatibility mode
- compiler/runtime version
- feature/config flags
- resolver/path/import state where relevant

Cacheable products include:

- parsed source
- source unit summaries
- HIR modules/assemblies
- MIR function bodies
- analysis facts
- function summaries
- class metadata
- VM layout
- bytecode
- snapshots

Rules:

- Local IDs may exist inside cached compiler products, but cache lookup and invalidation do not rely on local IDs alone.
- Function and module summaries support dependency-edge invalidation.
- LSP should recompute affected modules/functions without rebuilding unrelated project state.
- Runtime/provider-specific residency is not a static analysis cache fact.

## Multi-File CLI Walkthrough

This walkthrough constrains the ABI using a concrete scenario.

Command:

```bash
cd demo-matlab-compat
runmat ./analyze_sales.m
```

The project is `demo-matlab-compat` from the grounding examples above. `analyze_sales.m` is a script-like source-path entrypoint that composes a private helper, sibling function, package function, class constructor and method, function handle, workspace effects, environment effects, and builtins. The same flow applies to `demo-runmat`; compatibility mode and host policy differ, but source-path and named-entrypoint mechanics are the same.

### 1. CLI Creates An Execution Request

The CLI does not directly parse and run `analyze_sales.m` as an isolated blob. It constructs an execution request.

```rust
ExecutionRequest {
    source: SourceInput::Path("./analyze_sales.m"),
    entrypoint: EntrypointSelector::SourcePath("./analyze_sales.m"),
    compatibility: config.compatibility_mode,
    host_policy: cli_host_policy,
    inputs: RuntimeFlow::NoValue,
    requested_outputs: RequestedOutputCount::Zero,
    workspace: cli_workspace,
    resolver: project_resolver,
}
```

Current working directory, source path, project configuration, compatibility mode, and host policy are part of the execution environment.

### 2. Project Discovery

RunMat locates project context.

Conceptual order:

1. Start at current directory.
2. Look for RunMat project configuration, such as a future `runmat.toml`.
3. Determine source roots.
4. Determine package roots.
5. Determine private directories relevant to `analyze_sales.m`, including `private/`.
6. Determine class folders such as `@Report`.
7. Determine package folders such as `+stats`.
8. Register runtime builtins and runtime classes.
9. Build resolver environment.

If no project config exists, RunMat uses MATLAB-like path behavior rooted at cwd plus source file location.

This produces a project context.

```rust
ProjectContext {
    package: PackageIdentity,
    source_roots: Vec<SourceRoot>,
    path_entries: Vec<PathEntry>,
    compatibility: CompatibilityMode,
    resolver_cache_key: ResolverCacheKey,
}
```

### 3. Source Indexing

Before compiling `analyze_sales.m`, the resolver needs a source index. It scans or lazily indexes discoverable source units:

- `analyze_sales.m`
- sibling function files such as `normalizeRows.m`
- local/private functions such as `private/localScale.m`
- package functions under `+stats`, including `+stats/summarize.m`
- class files and class folders under `@Report`, including `@Report/Report.m`
- files referenced by imports or direct calls where discoverable

This does not require full eager compilation of every file. Lightweight summaries are enough for initial resolution.

```rust
SourceUnitSummary {
    path,
    source_unit_kind,
    module_name,
    declared_functions,
    declared_classes,
    package_path,
    private_scope,
    class_folder_scope,
    content_hash,
}
```

This matters because calls in `analyze_sales.m` resolve through different mechanisms: `localScale(data)` uses private-function precedence, `normalizeRows(scaled)` uses sibling source discovery, `stats.summarize(normalized)` uses package resolution, `Report("sales", totals)` uses class constructor resolution, and `report.title()` uses class method dispatch.

### 4. Parse Root Source

RunMat parses `analyze_sales.m` into AST.

For this project, `analyze_sales.m` is a script:

- top-level executable statements become a synthetic entry function;
- local functions become ordinary `HirFunction`s;
- source-path execution uses script/source-path entrypoint policy.

If the source-path target were a function file instead:

- the primary function becomes an ordinary `HirFunction`;
- source-path execution targets that primary function according to CLI policy;
- local functions are visible within the source unit.

If the source-path target were a class file instead:

- ordinary execution as a script is invalid unless the CLI requested an inspect/build mode;
- diagnostics should report source-context-aware invalid execution.

### 5. Semantic Lowering Of Root Module

The compiler lowers `analyze_sales.m` into semantic HIR.

```rust
HirAssembly {
    modules,
    functions,
    classes,
    bindings,
    entrypoints,
}
```

For root script `analyze_sales.m`, the entrypoint resembles:

```rust
HirEntrypoint {
    target: synthetic_analyze_sales_entry_function,
    origin: EntrypointOrigin::SourcePath,
    policy: EntrypointPolicy {
        workspace_export: WorkspaceExportPolicy::ExportTopLevelBindings,
        top_level_await: compatibility_or_host_policy,
    },
}
```

Top-level script variables become `HirBinding`s with `WorkspaceVisibility::TopLevel`. Function locals remain hidden.

### 6. Resolver Pulls In Dependencies

As HIR lowering resolves references, it consults the project resolver.

The root script contains:

```matlab
load sales.mat revenue expenses

data = [revenue expenses];
scaled = localScale(data);
normalized = normalizeRows(scaled);

[totals, averages] = stats.summarize(normalized);

report = Report("sales", totals);
headline = report.title();

cells = {totals, averages, headline};
[summaryTotals, summaryAverages, summaryTitle] = cells{:};

formatter = @(x) x + 1;
adjusted = feval(formatter, summaryTotals);

clear expenses
cd results
plot(adjusted)
```

Resolution may classify:

- `load` as a workspace-effecting builtin/command form that introduces `revenue` and `expenses`;
- `localScale` as `private/localScale.m` visible from `analyze_sales.m`;
- `normalizeRows` as sibling function file `normalizeRows.m`;
- `stats.summarize` as package function `+stats/summarize.m`;
- `[totals, averages] = ...` as a call requesting exactly two outputs;
- `Report` as class constructor from `@Report/Report.m`;
- `report.title()` as method dispatch on class `Report`;
- `cells{:}` as a comma-list producer consumed by multi-assignment;
- `formatter` as an anonymous function handle;
- `feval(formatter, summaryTotals)` as function-handle call ABI;
- `clear` as a workspace effect;
- `cd` as an environment effect and resolver invalidation point;
- `plot` as builtin identity that may internally use host interaction.

For dependencies in other files:

- statically needed modules may be parsed/lowered into the same or linked `HirAssembly`;
- unresolved or dynamic calls are represented explicitly and may defer lookup to runtime resolver services;
- resolver products record dependency edges and invalidation keys.

The composed compiler product should know its module graph.

```rust
ModuleGraph {
    root: analyze_sales_module,
    modules: Vec<ModuleId>,
    edges: Vec<ModuleDependency>,
}
```

### 7. Stable Identity Assignment

Every source/module/function/class receives local IDs inside the compilation product and stable identity at boundaries.

Example stable identity:

```rust
DefPath {
    package,
    module: QualifiedName(vec![SymbolName("stats"), SymbolName("summarize")]),
    item: vec![DefPathSegment::Function(SymbolName("summarize"))],
}
```

Rules:

- `FunctionId` is for this `HirAssembly`.
- `DefPath` is for diagnostics, cache keys, LSP, function handles, dynamic lookup, snapshots, persistent storage keys, and cross-module identity.

### 8. HIR To MIR For The Composed Program

The compiler lowers all needed executable functions to MIR.

MIR normalizes:

- control flow;
- calls;
- requested outputs;
- places;
- indexing;
- workspace effects;
- environment effects;
- captures;
- async boundaries;
- comma-list expansion and consumption;
- dynamic calls.

Each function has a MIR body.

```rust
MirBody {
    function: FunctionId,
    locals,
    blocks,
    source_map,
}
```

Cross-module calls use semantic call operands, not string names.

### 9. Analysis Runs

Analysis runs over HIR plus MIR.

It computes:

- type facts;
- shape facts;
- initialization facts;
- value-flow and comma-list facts;
- function summaries;
- effect summaries;
- capture read/write summaries;
- async behavior;
- spawn safety;
- fusibility and parallel safety;
- diagnostics.

For multi-file projects, summaries matter. The compiler should not inline-analyze the world every time.

```rust
FunctionSummary {
    function: DefPath,
    inputs,
    requested_output_sensitive_outputs,
    effects,
    async_behavior,
    may_call_unknown,
}
```

If `normalizeRows.m`, `private/localScale.m`, `+stats/summarize.m`, or `@Report/Report.m` changes, summaries depending on the changed file are invalidated. Unrelated modules are not rebuilt unnecessarily.

### 10. VM Layout Derivation

VM layout derives concrete frame slots from semantic bindings and MIR locals.

```rust
VmFunctionLayout {
    function: FunctionId,
    frame_abi,
    binding_slots,
    mir_local_slots,
    captures,
    local_count,
}
```

For the root entrypoint:

```rust
VmEntrypointLayout {
    entrypoint,
    target,
    workspace_export,
    exports: Vec<VmWorkspaceExport>,
}
```

Workspace exports in layout are local to this compiled product. Long-lived workspace state uses `WorkspaceBindingKey`.

### 11. Bytecode Generation

Bytecode is generated from MIR plus layout and selected analysis facts.

Rules:

- Direct function calls use `FunctionId` or layout-resolved function table entries.
- Builtins use stable `BuiltinId`.
- Class constructors and methods use class dispatch metadata.
- Unresolved calls use resolver runtime explicitly.
- Bytecode preserves source mapping to MIR/HIR.
- Unsupported JIT instructions fall back without changing semantics.

### 12. Runtime Begins Execution

Runtime creates the root call frame.

For a script entrypoint:

- there are no ordinary user input args;
- requested output count is zero or host-defined;
- top-level bindings have layout-backed slots;
- hidden temporaries and locals have slots;
- workspace-visible exports are known from layout and filtered by init/assigned state.

For a function entrypoint:

- CLI may pass args if supported;
- outputs return as `RuntimeFlow::OutputList`;
- locals do not enter ordinary workspace.

### 13. Calls Across Modules

When bytecode calls `normalizeRows`, runtime behavior depends on resolution.

If statically resolved:

```rust
CallableTarget::Function(normalize_rows_function_id)
```

Runtime creates a `VmCallFrame` using `normalizeRows`'s `VmFunctionLayout`.

If dynamically resolved:

```rust
CallableTarget::Unresolved(QualifiedName(vec![SymbolName("normalizeRows")]))
```

Runtime asks resolver runtime:

```rust
resolver.lookup_dynamic(query)
```

The result may be:

- compiled project function;
- builtin;
- class constructor;
- method;
- dynamic dispatch handle;
- structured diagnostic/error.

If a source file is lazily discovered, runtime may trigger compile/load through a compiler service and cache the result.

### 14. Workspace Effects During Execution

The root script contains:

```matlab
load sales.mat revenue expenses
clear expenses
cd results
```

These are not ordinary calls with hidden side effects. They lower to explicit effects or effectful builtins routed through runtime services.

```rust
WorkspaceEffectOp::Load { .. }
WorkspaceEffectOp::Clear { .. }
WorkspaceEffectOp::AssignIn { .. }
EnvironmentEffectOp::ChangeDirectory(..)
```

Runtime applies them through `WorkspaceRuntime` and `ResolverRuntime`.

Effects produce:

- workspace deltas;
- resolver invalidation;
- diagnostics if unsupported;
- fusion/JIT/reordering barriers.

### 15. Async And Host Interaction

If root code uses `await`, plotting, input, provider-backed work, filesystem async operations, or host services:

- MIR marks suspension-capable points;
- runtime may suspend with a `Suspension` record;
- CLI may block/poll to completion;
- notebook/kernel may surface pending interaction;
- live slots, temporaries, captures, object handles, and provider handles are rooted.

A synchronous CLI can use an async runtime internally, but the model does not hide async capability.

### 16. Workspace Export At Completion

When the root entrypoint completes, runtime exports workspace candidates.

For each candidate:

1. Read `VmEntrypointLayout.exports`.
2. Check `WorkspaceVisibility`.
3. Check `InitFact` and runtime assigned bits.
4. Read final frame slot or storage value.
5. Convert local `BindingId` to stable `WorkspaceBindingKey`.

For a script binding in `analyze_sales.m`:

```rust
WorkspaceBindingKey::SourceBinding {
    source: SourceIdentity::PathAndContentHash("./analyze_sales.m", hash),
    def_path: DefPath { .. synthetic entry .. },
    binding: BindingName("adjusted"),
}
```

For CLI one-shot execution, workspace may be displayed and dropped. For REPL/session execution, workspace persists.

For globals:

```rust
WorkspaceBindingKey::Global { scope, name }
```

For persistents:

```rust
WorkspaceBindingKey::Persistent { function: normalize_rows_def_path, name }
```

### 17. CLI Presents Outcome

CLI receives structured outcome.

```rust
ExecutionOutcome {
    flow,
    workspace_delta,
    display_events,
    diagnostics,
    effects,
    profiling,
    suspension: None,
}
```

The CLI prints:

- display events from unsuppressed statements;
- explicit builtin output;
- warnings and diagnostics;
- final public result if host policy says to show it.

The CLI does not inspect bytecode, HIR, legacy statements, or console buffers to guess result/display behavior.

### 18. Caching

Future runs reuse cached products when dependencies are unchanged.

Cache keys include:

- source content hash;
- project config hash;
- dependency graph hash;
- compiler/runtime version;
- compatibility mode;
- feature flags;
- resolver/path/import state where relevant.

Cacheable products include:

- source summaries;
- parsed AST;
- HIR modules/assemblies;
- MIR bodies;
- analysis summaries;
- class metadata;
- VM layout;
- bytecode;
- snapshots.

### Multi-File Sanity Checks

- Multi-file calls do not become string lookups when statically resolvable.
- Dynamic lookup remains possible but explicit.
- Workspace state is not tied to ephemeral `BindingId`.
- Script locals export only by entrypoint policy and binding visibility.
- Function locals never leak into ordinary workspace.
- Class/package/private/path resolution is project-aware.
- `cd`, `addpath`, `clear functions`, `eval`, and `load` invalidate or mutate explicit runtime services.
- JIT and fusion cannot cross unknown effects.
- Async boundaries survive through MIR, bytecode, and runtime execution.
- CLI, LSP, kernel, WASM, and desktop hosts consume shared diagnostics and source maps.

In summary, `runmat ./analyze_sales.m` means: discover project context, build resolver/source graph for `analyze_sales.m`, `normalizeRows.m`, `private/localScale.m`, `+stats/summarize.m`, and `@Report/Report.m`, lower the root and needed modules into semantic HIR, lower to MIR, analyze, derive layout, compile bytecode, execute the selected entrypoint through runtime services, export workspace by semantic policy into stable workspace keys, and present structured outcome.
