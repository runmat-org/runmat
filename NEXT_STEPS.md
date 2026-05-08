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
  - Currently used by function-handle lowering.
  - Prefer a typed `CreateFunctionHandle` / `CreateDynamicFunctionHandle` instruction or runtime ABI call.

- `feval`
  - Currently special-cased by string name in VM lowering.
  - Prefer MIR/VM representation for dynamic invocation instead of `name == "feval"` checks.

- `call_method`
  - Currently used for object `subsref` / `subsasgn` plumbing.
  - Prefer typed object dispatch operations.

- `__make_cell`
  - Currently used to package indices for object protocol calls.
  - Prefer structured index descriptors.

- object protocol strings: `"subsref"`, `"subsasgn"`, `"()"`, `"{}"`
  - These are MATLAB protocol concepts, but the VM should not manually assemble ad hoc argument lists for them.
  - Prefer `ObjectSubsref` / `ObjectSubsasgn` style operations carrying structured index data.

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

After indexing assignment classification, target hard-coded function-handle dispatch names.

Concrete next target:

- Replace `make_handle` string-call lowering with a typed function-handle instruction.
- Replace `feval` string special-casing with explicit dynamic-call lowering.
- Preserve semantic closure behavior and existing function-handle ratchets.

This is narrower than object `subsref` / `subsasgn` and should reduce compiler/runtime string coupling without requiring a full object protocol rewrite.

## Validation Cadence

For each coherent slice:

- focused regression
- `cargo fmt --all --check`
- `cargo test -p runmat-core --test semicolon_suppression`
- `cargo check --workspace`
- `git diff --check`
- commit as `RM-378: <3-4 word message>`
