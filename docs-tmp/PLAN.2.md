# Plan 2: MIR, Analysis, Diagnostics, And Incrementality

## Objective

Introduce `runmat-mir` and make normalized control-flow/dataflow analysis a first-class compiler layer between semantic HIR and VM/runtime lowering.

Plans 0-1 create semantic HIR and resolver products. Plan 2 lowers HIR into MIR, defines the canonical analysis store, and establishes the diagnostic and incremental compilation products that later consumers use.

## Desired Resting State

The compiler has a coherent middle layer:

- `runmat-mir` crate exists
- every `HirFunction` can lower to a `MirBody`
- MIR bodies are CFG-based
- analysis facts are keyed by semantic IDs and MIR-local IDs as appropriate
- diagnostics are structured compiler products
- function summaries are keyed by `FunctionId`
- binding facts are keyed by `BindingId`
- local dataflow operates over MIR, not recursive source-shaped statement trees
- async/suspension behavior is represented in MIR and summaries

Downstream runtime crates may still be broken until Plan 3.

## Core Invariants

- HIR is semantic/source truth.
- MIR is derived normalized compiler IR.
- MIR locals are not semantic bindings.
- MIR locals can map back to `BindingId`s or represent compiler temporaries.
- Control flow is explicit through basic blocks and terminators.
- Async boundaries are explicit enough to prevent invalid reordering.
- Analysis does not use VM slots as semantic identity.
- Static analysis may express acceleration eligibility, not concrete runtime residency.

## Primary Crates

- `runmat-mir`
- `runmat-hir`
- `runmat-static-analysis`

## Secondary Crates

- `runmat-builtins`
- `runmat-lsp`
- `runmat-vm`

## MIR Shape

Suggested initial model:

```rust
pub struct MirAssembly {
    pub bodies: HashMap<FunctionId, MirBody>,
}

pub struct MirBody {
    pub function: FunctionId,
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<BasicBlock>,
    pub source_map: MirSourceMap,
}

pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<MirStmt>,
    pub terminator: MirTerminator,
}
```

MIR should include:

- normalized assignments
- normalized places
- explicit place mutation operations with creation, deletion, scalar-expansion, and shape policies
- normalized indexing descriptors with colon, logical, and symbolic `end` components
- compatibility mode and source unit facts where they affect diagnostics/lowering
- rvalues
- explicit operator kinds and function handle values
- calls with requested-output count
- comma-list expansion and consumption points
- workspace-effect operations
- environment-effect operations
- empty-array role, expansion, aggregate, numeric class, string/char, and control-flow semantic markers where needed
- branches
- returns
- loop control through CFG edges
- environment/capture accesses where needed
- async-capable call markers
- future creation for async functions and async blocks
- await suspension points with explicit resume edges
- spawn boundaries that schedule futures and produce task handles
- spawn-safety markers for values and captures crossing task boundaries
- source mapping back to HIR spans and IDs

## Implementation Plan

1. Create `runmat-mir`.

Add crate, workspace membership, core IDs, and MIR data structures.

2. Add HIR-to-MIR lowering.

Lower each `HirFunction` to `MirBody`:

- create entry block
- lower structured control flow to CFG
- lower expressions to rvalues and temporaries where useful
- lower `HirPlace` to MIR places
- lower `PlaceMutation`, `IndexingSemantics`, requested-output calls, and comma-list flows explicitly
- preserve source spans and HIR ID source mapping
- map semantic bindings to MIR locals where appropriate

3. Add MIR source maps.

Diagnostics and LSP need mappings from MIR facts back to:

- `ExprId`
- `StmtId`
- `BindingId`
- source spans
- function/module/class context

4. Add async boundary representation.

MIR calls or terminators should represent whether an operation:

- never suspends
- may suspend
- requires async runtime

This should be conservative and metadata-driven where possible.

MIR should model user-facing async with Rust-like lazy future semantics:

- async function bodies and async blocks lower to future state machines or future-producing MIR bodies
- creating a future has no user-code side effects
- `await` is a terminator or explicit statement with suspend, poll, resume, and error edges
- `spawn` is the only eager scheduling/concurrency primitive in the language model
- task handles are distinct from futures in analysis facts
- ordinary language-synchronous builtin calls may still be marked `MaySuspend` internally without producing future values
- futures passed to `spawn` must be classified as spawn-safe before VM lowering
- mutable lexical captures from a parent frame are not spawn-safe unless represented by explicit synchronized/runtime-managed handles

Liveness across await is a correctness requirement. MIR must expose the locals, stack values, captures, and temporaries live across each suspension point so the VM/runtime can root them and so optimizers cannot move side effects across await unsafely.

5. Add MATLAB compatibility semantic representation.

MIR should preserve enough semantic structure for correct runtime lowering and diagnostics:

- requested-output count on calls
- output target lists and discarded outputs
- compatibility mode and source unit kind where relevant to diagnostics and lowering
- list-valued flow for comma-separated lists
- function handle targets and dynamic callable effects
- `varargin{:}` / cell expansion consumption points
- explicit `nargin` and `nargout` reads
- assignment creation policies
- indexed assignment growth and deletion
- symbolic `end` resolution context
- workspace effects from `load`, `clear`, `eval`, `evalin`, `assignin`, globals, and persistents
- environment effects from path/cwd/cache mutation APIs
- empty-array roles, concat semantics, scalar/implicit expansion, operator kinds, numeric classes, string/char facts, aggregate facts, and control-flow semantics
- nominal dispatch hooks for constructors, methods, operators, and overloaded indexing

6. Add canonical analysis store.

Suggested initial shape:

- `AnalysisStore`
- `SemanticIndex`
- `ResolutionIndex`
- `MirIndex`
- `BindingFact`
- `ExprFact`
- `MirLocalFact`
- `FunctionSummary`
- `ModuleSummary`
- `TypeFact`
- `ShapeFact`
- `InitFact`
- `EffectSummary`
- `FusibilityFact`
- `ParallelSafetyFact`
- `AccelEligibilityFact`
- `DataMovementPolicyHint`
- `AsyncBehaviorFact`
- `AsyncValueFact`
- `SpawnSafetyFact`
- `ValueFlowFact`
- `WorkspaceEffect`
- `EnvironmentEffect`
- `FunctionHandleTarget`
- `EmptyArrayRole`
- `ExpansionSemantics`
- `OperatorKind`
- `NumericClass`
- `AggregateKind`
- `LoopIterationSemantics`
- `TensorElementDomainFact`

7. Implement local flow analysis over MIR.

Track:

- binding facts
- MIR local facts
- initialization
- type facts
- shape facts
- effects
- workspace effects
- environment effects
- value-flow facts for single values, no-value flows, and comma-separated lists
- function-handle facts
- requested-output-sensitive call facts
- indexing and assignment mutation facts
- empty-array role facts
- expansion, operator, numeric-class, string/char, aggregate, and control-flow facts
- tensor element-domain facts
- async behavior
- async value facts for futures, tasks, and non-awaitable values
- live values across await/suspension points
- spawn-safety facts for futures, captures, and provider/runtime handles

Use explicit joins at CFG merge points and widening for loops.

8. Implement function summaries.

Summaries should include:

- parameter facts
- output facts
- output facts by requested-output count where behavior differs
- effects
- workspace effects
- capture read/write sets
- async behavior
- spawn safety
- call behavior
- fusibility and acceleration eligibility

9. Replace old function reconstruction inside analysis.

Remove reliance on helpers like `collect_function_defs`. Analysis should iterate `HirAssembly.functions` and `MirAssembly.bodies` keyed by `FunctionId`.

10. Add structured diagnostics.

Diagnostics should support:

- stable code
- severity
- primary span
- secondary spans
- notes/help
- optional suggestions
- source/module/package context
- MATLAB semantic categories such as arity mismatch, invalid comma-list use, invalid assignment growth, invalid `end`, command-syntax misuse, workspace-effect misuse, and class dispatch failures
- compatibility-mode, source-unit, function-handle, environment-effect, empty-array, expansion, numeric-class, string/char, aggregate, and control-flow diagnostics

11. Add incremental compilation scaffolding.

Define cacheable product boundaries and keys for:

- HIR modules
- MIR bodies
- function summaries
- class metadata
- analysis facts

No full incremental engine is required in this plan, but product boundaries and identity rules must not block it.

12. Port initial static-analysis lints.

Start with shape and data API lints because they currently expose the pain from string-keyed calls and slot-like variable IDs.

13. Add acceleration eligibility facts.

Static analysis may mark expressions/functions as fusible, acceleration-eligible, acceleration-preferred, or blocked by known semantic reasons. It must not encode concrete runtime residency, device IDs, provider availability, buffer IDs, or provider-specific allocation state.

## Tests

Add tests for:

- simple function lowers to MIR CFG
- `if` creates merge points
- loops create backedges and widening sites
- `try/catch` preserves control-flow shape
- assignments map semantic bindings to MIR locals
- temporaries do not become semantic bindings
- source mapping from MIR facts to HIR/source spans
- definite assignment facts
- requested-output call facts
- function handle and dynamic callable facts
- multi-output destructuring and discarded outputs
- comma-list expansion and consumption facts
- indexed assignment growth/deletion facts
- symbolic `end` indexing facts
- workspace effects block unsafe reordering
- environment effects invalidate or barrier cached/dynamic lookup where needed
- empty-array role, expansion, operator, numeric-class, string/char, aggregate, and control-flow facts are represented
- tensor element-domain facts track real vs complex results
- shape joins across branches
- invalid capture/isolated diagnostics map correctly
- async-capable calls are marked conservatively
- async functions/blocks are lazy until awaited or spawned
- await points expose live-across-suspension values
- spawned futures with mutable lexical captures are rejected
- spawn-safety failures produce source-mapped diagnostics
- function summaries are keyed by `FunctionId`
- facts are keyed by `BindingId` / `ExprId` / MIR-local IDs as appropriate

## Acceptance Criteria

- `runmat-mir` exists and compiles.
- `runmat-hir` can lower semantic HIR functions to MIR bodies.
- Local dataflow analysis runs over MIR.
- Structured diagnostics are available to analysis and lowering.
- Function summaries and binding facts use semantic IDs.
- Async behavior is represented conservatively in MIR and function summaries.
- Futures, tasks, `await`, and `spawn` have distinct MIR and analysis representation.
- Spawn-safety is represented and enforced before VM/runtime lowering.
- Requested-output calls, comma-separated lists, function ABI facts, function handles, place mutations, indexing semantics, workspace/environment effects, core MATLAB semantic facts, and tensor element domains are represented in MIR/analysis.
- Acceleration/fusion facts express semantic eligibility and policy hints only, not concrete runtime placement.

## Explicit Non-Goals

- Do not restore all VM/runtime consumers in this plan; that is Plan 3.
- Do not complete full MATLAB core semantics; that is Plan 4.
- Do not implement the full project manifest model; that is Plan 5.
- Do not implement runtime class/builtin metadata generation; that is Plan 6.
- Do not complete accelerate/fusion/GC hardening; that is Plan 7.
- Do not implement a complete incremental compiler cache yet; define boundaries and keys first.
