# Remaining Items Queue (Objective Item 3)

Date: 2026-05-21
Scope: Non-builtin semantic-product closeout for Objective Item 3.

## Scope Locks (Confirmed)

1. Objective Item 3 closeout requires completion of queue items through P2 unless explicitly reclassified with evidence.
2. Varargout target-list expansion ABI is in scope.
3. Async/future/spawn runtime ABI completion is in scope.
4. Method function-handle support is in scope via typed method-handle identity (not permanent unsupported rejection).
5. Classes/structs are in scope:
   - in-scope now: class/object/member semantics and struct/object runtime behavior on active IR paths.
   - also in scope now: struct/object aggregate-literal source-form enablement (parser/HIR/MIR + typed bytecode construction), previously tracked as forward design.

## Execution Rules

1. Execute items strictly in order unless an item is explicitly marked parallel-safe.
2. Do not start item `N+1` until item `N` meets its acceptance criteria.
3. For each item, add ratchet tests before or with implementation.
4. Preserve typed identifiers; do not add message-fragment-only diagnostics.
5. After each item, run required gates:
   - `cargo fmt --all --check`
   - `cargo test -p runmat-core --test -- --nocapture`
   - `cargo check --workspace`
   - `git diff --check`
6. Commit changes with a descriptive commit message.

## P0: Compiler/Runtime Semantic Ownership Closure

### 1) MIR Assignment-Place Classification Completion
- Goal: MIR explicitly classifies all assignment-place forms so VM does not infer from stack/value shape.
- Includes: scalar paren, slice/logical/range paren, deletion, cell paren replace/delete, brace content assign, member/indexed store-back.
- Acceptance:
  - No runtime-only category inference required for covered forms.
  - New/updated MIR + VM ratchets assert exact shape + identifier contracts.

### 2) Slice Lowering Normalization
- Goal: slice cases lower intentionally as slice ops, not scalar-store reinterpretation.
- Includes: explicit deletion intent encoding in MIR/bytecode.
- Acceptance:
  - Scalar index => scalar store only.
  - Colon/range/logical/vector => slice store only.
  - Delete intent represented explicitly, not inferred from RHS emptiness at runtime.

### 3) Centralized Cell Selector/Assignment Plan API
- Goal: one cell selector-plan API across read/write/expand/delete.
- Includes: linear/subscript/colon/range/logical selection, expansion order, RHS expansion/shape checks, write barriers.
- Acceptance:
  - Direct scattered `ca.data[...]` access removed from dispatch paths in scope.
  - Cell operations route through shared plan executor with identifier-preserving failures.

### 4) Dispatch Simplification Pass
- Goal: dispatch executes explicit operations only.
- Includes:
  - `StoreIndex` scalar-only.
  - `StoreSlice` planned-slice-only.
  - Cell/object lanes execute structured descriptors/plans.
- Acceptance:
  - No syntax/classification heuristics in dispatch for covered lanes.
  - Existing behavior ratchets remain green.

## P1: Callable ABI and Call-Shape Closure

### 5) Remaining Internal String-Hook Removal (In-Scope)
- Goal: eliminate remaining compiler-internal string hooks where typed ABI exists.
- Acceptance:
  - Internal service paths use typed instructions/helpers.
  - Public builtin dispatch boundaries remain valid.
  - Ratchets prove no regression to internal string-call lowering.

### 6) Callable Descriptor Unification for Residual Name-Shaped Paths
- Goal: migrate remaining dynamic-name callback producers to semantic descriptors where resolvable.
- Includes: unresolved/external identities remain unresolved unless semantic registry resolves them.
- Acceptance:
  - No legacy-style dynamic recompilation/fallback introduced.
  - Resolver/fallback policy remains typed and explicit.

### 7) MIR Call-Shape Normalization
- Goal: converge call shapes toward one descriptor family in MIR.
- Includes: callee kind, syntax kind, expansion spec, requested outputs, dispatch/effect facts.
- Acceptance:
  - Bytecode specialization is lowering detail, not semantic rediscovery.
  - Ratchets assert descriptor fact completeness on active call forms.

### 8) Turbine Expanded-Call Value-Lane Completion
- Goal: complete typed host-bridge coverage for remaining expanded-call surfaces.
- Includes: remaining unresolved method/member expanded object/member dispatch lanes in scope.
- Acceptance:
  - No compile-time fallback gate regressions.
  - JIT path coverage asserts typed bridge behavior and unresolved-call contracts.

## P2: Index/Object/Output Policy Finalization

### 9) Object/Member Protocol Descriptor Completion
- Goal: all object/member index callsites build/consume structured descriptors directly.
- Acceptance:
  - No remaining ad hoc selector-cell protocol assembly in scope paths.
  - Subsref/subsasgn routes are descriptor-first.

### 10) Residual Selector-Plan Edge Normalization
- Goal: close remaining narrow non-tensor selector-plan edge gaps.
- Acceptance:
  - Remaining `slice index` gap markers in audit are either closed or reclassified with explicit reproducer.
  - Identifier contracts are stable and ratcheted.

### 11) Assignment-Place Residual Gap Sweep
- Goal: resolve remaining assignment-place gaps with explicit source reproducers.
- Acceptance:
  - No generic slot-only fallback logic for in-scope assignment-place paths.
  - Reproducers converted to compile/runtime contracts.

### 12) Multi-Output / Varargout Target-List ABI (If Enabled)
- Goal: define and implement explicit target-list ABI for varargout expansion support.
- Includes: fixed outputs, varargout capture target, discard targets, output-count propagation.
- Acceptance:
  - MIR owns output target list semantics before bytecode.
  - No runtime stack-heuristic output shaping in covered paths.

## P3: Remaining Design-Track Closures (Non-Blocking for Current Item 3 Closeout Unless Elevated)

### 13) Async/Future/Spawn Runtime ABI Completion (In Scope)
- Goal: implement agreed suspension/resume/cancellation/diagnostic model.
- Acceptance:
  - Await/spawn/future bytecode/runtime paths complete under defined host policy.

### 14) Compatibility Policy Final Disentanglement
- Goal: fully separate parser syntax policy, semantic behavior policy, and host execution policy.
- Acceptance:
  - No remaining mixed policy seams in active compiler/runtime boundaries.

### 15) Struct/Object Aggregate-Literal Product Work (In Scope)
- Goal: implement parser/HIR/MIR source form plus typed aggregate construction for struct/object aggregate literals.
- Acceptance:
  - Strict evaluation order and duplicate-field policy documented + ratcheted.

## Exit Criteria For Objective Item 3

Objective Item 3 can move from `partial` to `met` only when:

1. All P0, P1, and P2 items are complete and verified.
2. P3 items that are marked in-scope are complete and verified.
3. `docs-tmp/DELIVERABLE_AUDIT.md` section `### 3` is updated from `partial` to `met` with concrete artifact evidence.
4. Full required gates are green after the final closure slice.
