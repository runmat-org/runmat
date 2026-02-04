# Builtin Resolver Context Migration Plan

This document is the authoritative execution plan for migrating all builtins to the unified resolver API with `ResolveContext` support. It is intentionally detailed so subagents can follow it mechanically without improvisation.

## Scope

We are migrating builtin type inference to use a single resolver API that can accept either:
- Legacy resolver signature: `fn(&[Type]) -> Type`
- Context-aware resolver signature: `fn(&[Type], &ResolveContext) -> Type`

The public entrypoint for both is the `type_resolver(...)` attribute, which now accepts either signature. `type_resolver_ctx(...)` is deprecated and should not be used.

This migration is repository-wide and will not be partially shipped. All changes are staged until the entire conversion is complete and tests pass.

## Guiding Rules

1. Do not add `#[allow(...)]` or lint suppressions.
2. Do not add new behavior unless explicitly called out here.
3. Do not introduce new per-builtin special cases in HIR; all logic goes through builtin resolvers and shared helpers.
4. Use ASCII only in new or edited files.
5. Prefer shared helper updates over per-builtin logic to keep changes systemic.
6. Do not run formatting tools that rewrite large files unless necessary.
7. Keep behavior conservative: if literal context is unavailable or ambiguous, return `Unknown` or unknown shapes rather than guessing.
8. Use a single canonical resolver implementation: context-aware helpers use the original function names, and legacy adapters (only while migrating) are explicitly suffixed `_legacy`.

Temporary resolver annotation rule:
- While both resolver signatures exist, use `type_resolver(...)` plus `type_resolver_context = true` to indicate a context-aware resolver.
- This flag is temporary and must be removed in Phase 5 once all resolvers accept `ResolveContext`.

## Naming Rules

- Canonical ctx-aware helpers keep the original function name (no suffixes).
- Temporary legacy adapters must be suffixed `_legacy`.
- No `_ctx` or `_with_ctx` suffixes are allowed in new or edited code.

## Current State (Baseline)

- `ResolveContext` exists with `literal_args: Vec<LiteralValue>` and helpers:
  - `numeric_dims()`
  - `numeric_dims_from(start)`
- `TypeResolverKind` exists with `Legacy` and `WithContext` variants.
- `type_resolver(...)` is the standard attribute; `type_resolver_ctx(...)` is no longer used in builtins.
- HIR builds `ResolveContext` for builtin calls and routes to `infer_return_type_with_context`.

## ResolveContext Extensions (Required)

These must be added and used during the migration. They are not optional.

1) Literal numeric vectors
   - Capture literal numeric vectors and arrays (e.g., `[2 3]`).
   - Enables `reshape`, `permute`, `ipermute`, and index vector inference.

2) Literal strings and chars
   - Capture literal strings and char arrays.
   - Enables keyword-driven shape inference such as `sum(A, "all")`.

3) Literal booleans
   - Capture literal logical scalars.
   - Enables flag-driven output shape decisions when relevant.

4) Literal complex scalars (optional, but recommended)
   - Enables correct scalarization and dtype inference in some cases.

## ResolveContext Data Model (Proposed)

This is the expected structure for `ResolveContext`. The exact type names can vary, but the data must be representable.

- `literal_args: Vec<LiteralValue>` (parallel to builtin argument list)

Where `LiteralValue` includes:
- `Number(f64)`
- `Bool(bool)`
- `String(String)` (lowercased when appropriate for keyword matching)
- `Vector(Vec<LiteralValue>)` (for numeric and logical vectors; allow nested vectors for 2-D literals)
- `Unknown`

Required helper methods on `ResolveContext`:
- `numeric_dims()` and `numeric_dims_from(start)` (existing)
- `literal_string_at(index) -> Option<String>`
- `literal_bool_at(index) -> Option<bool>`
- `literal_vector_at(index) -> Option<Vec<LiteralValue>>`
- `numeric_vector_at(index) -> Option<Vec<Option<usize>>>` (lossy conversion; unknown where non-numeric)
- `empty() -> ResolveContext` (new, for legacy adapters only)

HIR is responsible for populating these fields from literal expressions only. Non-literal expressions must map to `Unknown`.

## ResolveContext API Changes (Concrete)

These are the concrete code-level changes required in `runmat/crates/runmat-builtins/src/lib.rs`.

### 1) Extend `LiteralValue`

Add new variants (or equivalent) to support strings, booleans, and vectors:

```
pub enum LiteralValue {
    Number(f64),
    Bool(bool),
    String(String),
    Vector(Vec<LiteralValue>),
    Unknown,
}
```

### 2) Add `ResolveContext::empty()`

```
impl ResolveContext {
    pub fn empty() -> Self {
        Self { literal_args: Vec::new() }
    }
}
```

### 3) Add helper accessors

```
impl ResolveContext {
    pub fn literal_string_at(&self, index: usize) -> Option<String> { ... }
    pub fn literal_bool_at(&self, index: usize) -> Option<bool> { ... }
    pub fn literal_vector_at(&self, index: usize) -> Option<Vec<LiteralValue>> { ... }
    pub fn numeric_vector_at(&self, index: usize) -> Option<Vec<Option<usize>>> { ... }
}
```

Rules:
- `literal_string_at` returns lowercase for keyword matching.
- `literal_vector_at` returns a flat vector for 1-D literals; 2-D literals may return a nested `Vector` structure that helper code must interpret.
- `numeric_vector_at` returns `None` when the literal is not a numeric vector; `Some(vec![None, ...])` when the vector exists but includes non-numeric values.
- Do not coerce non-literal expressions.

### 4) Update HIR literal extraction

HIR should map literal expressions into the new `LiteralValue` variants:
- Numeric scalars -> `Number`
- Logical scalars -> `Bool`
- String or char literals -> `String`
- Vector literals -> `Vector` of element `LiteralValue`
- Any non-literal expression -> `Unknown`

## How Context Maps to Builtin Families

This section specifies which context fields should be used by which builtin families. It is a binding requirement.

### Array creation
- `colon`, `range`, `linspace`, `logspace`:
  - Use numeric literals for count/step when present to infer output length.
  - Use numeric literal vectors for size arguments where applicable.
- `randi`, `randperm`, `fill`, `meshgrid`:
  - Use numeric literal vectors for output size.
  - Use `"like"` keyword literal detection when provided (string/char).

### Array shape
- `reshape`:
  - Use numeric literal vectors to compute exact output shape when possible.
  - Support `-1` in literal dims by inferring the missing dimension if `numel` is known.
- `permute` / `ipermute`:
  - Use numeric literal vectors to compute exact axis permutation and output rank.
- `cat`, `horzcat`, `vertcat`:
  - Use numeric literal dimension when provided (for `cat`).
  - Preserve existing concat shape logic for shape inference of inputs.
- `squeeze`:
  - Use literal shape info from input types (no new literals required).

### Indexing
- `find`:
  - If a literal dimension is provided, adjust the output shape accordingly.
- `sub2ind`, `ind2sub`:
  - Use numeric literal vectors for size input to infer output shape.

### Reduction
- `sum`, `mean`, `prod`, `min`, `max`, `cumsum`, `cumprod`, `cummin`, `cummax`, `diff`, `nnz`, `all`, `any`, `median`, `std`, `var`:
  - Use numeric literal dimension where present.
  - Use string keywords (`"all"`, `"omitnan"`, `"includenan"`) where relevant.

### Linalg
- `dot`:
  - Use numeric literal dimension if present to infer scalar vs vector output.

### Strings / IO / Diagnostics
- These families generally do not need ctx extensions for shape, but can use literal strings to refine union outputs when explicit keywords are present.

## ResolveContext Extension Implementation Order

1) Numeric vector literals (including `[2 3]` and row/column vectors)
2) String and char literals (lowercased for keyword matching)
3) Boolean literals
4) Complex literals (optional)

## Shared Argument Parsing (Required)

We must avoid duplicating argument parsing logic between runtime execution and type inference.

### Approach

Introduce a shared, type-level argument token model in runtime common code. Both the runtime and resolver will parse arguments through this layer.

**Location:** `runmat/crates/runmat-runtime/src/builtins/common/arg_tokens.rs`

**Canonical token model:**

```
pub enum ArgToken {
    Number(f64),
    Bool(bool),
    String(String),
    Vector(Vec<ArgToken>),
    Unknown,
}
```

**Required constructors:**

- `tokens_from_values(args: &[Value]) -> Vec<ArgToken>`
  - Used by runtime code for option/flag parsing.
  - Does not replace runtime numeric conversions for data, only option parsing.

- `tokens_from_context(ctx: &ResolveContext) -> Vec<ArgToken>`
  - Used by type resolvers.
  - Must reflect literal-only data from `ResolveContext`.

**Parsing rule:**

For any builtin with non-trivial argument parsing, define a `ParsedArgs` struct and a single parser function that consumes `&[ArgToken]`. Both runtime and resolver logic must call this parser.

### Example (Reduction)

```
struct ParsedReductionArgs {
    dim: DimSelection,
    nan_mode: ReductionNaN,
    all: bool,
}

fn parse_reduction_args(tokens: &[ArgToken]) -> Result<ParsedReductionArgs, String> { ... }
```

Runtime:
- Convert values to tokens with `tokens_from_values`.
- Call `parse_reduction_args` for options.
- Use parsed output to drive runtime computation.

Resolver:
- Convert `ResolveContext` to tokens with `tokens_from_context`.
- Call `parse_reduction_args`.
- Use parsed output to determine inferred output shape.

### Notes

- This is not a replacement for runtime `Value` parsing of data; it is solely for shared option and dimension semantics.
- Use lowercase strings for keyword matching.
- Do not auto-coerce non-literal expressions in `tokens_from_context`.

## High-Leverage Helper Hubs

These files are the primary targets. Changes here propagate to many builtins.

1) `runmat/crates/runmat-runtime/src/builtins/array/type_resolvers.rs`
2) `runmat/crates/runmat-runtime/src/builtins/math/reduction/type_resolvers.rs`
3) `runmat/crates/runmat-runtime/src/builtins/math/linalg/type_resolvers.rs`
4) `runmat/crates/runmat-runtime/src/builtins/math/type_resolvers.rs`
5) `runmat/crates/runmat-runtime/src/builtins/logical/type_resolvers.rs`

Also relevant:
- `runmat/crates/runmat-builtins/src/shape_rules.rs`

## Migration Order (Strict)

This sequence must be followed to keep the system consistent and minimize churn.

### Phase 1: Resolver API conformance (already done)
- Ensure `type_resolver(...)` accepts both resolver signatures.
- Ensure `BuiltinFunction` uses `TypeResolverKind` and `infer_return_type_with_context`.
- Ensure `type_resolver_ctx(...)` is not used in builtins.
- Add temporary attribute flag `type_resolver_context = true` for context-aware resolver paths.

### Phase 2: Add ctx-aware helper variants (no builtin edits yet)

**Goal:** Add context-aware helper functions as the canonical implementation. Legacy helpers remain only as thin adapters until the end of the migration.

#### 2.0 ResolveContext + ArgToken foundations
File: `runmat/crates/runmat-builtins/src/lib.rs`

- Implement the `ResolveContext` extensions described above.
- Add `ResolveContext::empty()` for legacy adapter use only.

File: `runmat/crates/runmat-runtime/src/builtins/common/arg_tokens.rs`

- Add `ArgToken` model and `tokens_from_values` / `tokens_from_context` constructors.
- Do not wire into builtins yet; that happens in Phase 3.

#### 2.1 Array helpers
File: `runmat/crates/runmat-runtime/src/builtins/array/type_resolvers.rs`

Convert helpers to context-aware functions using their original names. Exact names:
- `rank_from_dims_args(args: &[Type], ctx: &ResolveContext) -> Option<usize>`
  - Use literal dims from `ctx.numeric_dims()` when present.
  - If literal dims exist, return `Some(dims.len())`.
  - Otherwise, fall back to existing `rank_from_dims_args` logic.

- `tensor_type_from_rank(args: &[Type], ctx: &ResolveContext) -> Type`
  - If `tensor_type_from_literal_dims(args, ctx)` returns `Some`, return that.
  - Otherwise, use `rank_from_dims_args` and then `tensor_type_from_rank_legacy`.

- `row_vector_type(ctx: &ResolveContext) -> Type`
  - If `ctx.numeric_dims().get(0)` is `Some(Some(n))`, return `Type::Tensor { shape: Some(vec![Some(1), Some(n)]) }`.
  - If literal length not available, return `row_vector_type_legacy()`.
  - Do not attempt to parse literals beyond `numeric_dims()`.

Legacy adapters (temporary, to be deleted in Phase 5):
- `rank_from_dims_args_legacy(args)` calls `rank_from_dims_args(args, &ResolveContext::empty())`
- `tensor_type_from_rank_legacy(args)` calls `tensor_type_from_rank(args, &ResolveContext::empty())`
- `row_vector_type_legacy()` calls `row_vector_type(&ResolveContext::empty())`

#### 2.2 Reduction helpers
File: `runmat/crates/runmat-runtime/src/builtins/math/reduction/type_resolvers.rs`

Convert helpers to context-aware functions using their original names. Exact names:
- `reduction_shape_from_args(args: &[Type], ctx: &ResolveContext) -> Option<Vec<Option<usize>>>`
  - If `args.len() == 1`, match prior `reduction_shape_from_args` behavior.
  - If `args.len() >= 2`, attempt to read a literal reduction dimension from `ctx.numeric_dims_from(1)`. If a literal dimension is present, compute output shape by setting that dimension to 1 (1-based indexing).
  - If literal dimension is missing or invalid, fall back to `unknown_shape(shape.len())`.

- `reduce_numeric_type(args: &[Type], ctx: &ResolveContext) -> Type`
- `reduce_logical_type(args: &[Type], ctx: &ResolveContext) -> Type`
  - These mirror existing functions but call the new `reduction_shape_from_args`.

Legacy adapters (temporary, to be deleted in Phase 5):
- `reduce_numeric_type_legacy(args)` calls `reduce_numeric_type(args, &ResolveContext::empty())`
- `reduce_logical_type_legacy(args)` calls `reduce_logical_type(args, &ResolveContext::empty())`

#### 2.3 Linalg helpers
File: `runmat/crates/runmat-runtime/src/builtins/math/linalg/type_resolvers.rs`

Add context-aware versions only where dimension arguments exist or shape behavior depends on literal dims. For now:
- `dot_type(args: &[Type], ctx: &ResolveContext) -> Type`
  - If args include a literal dimension index in `ctx.numeric_dims_from(2)`, reduce along that dimension for vector inputs when possible. If not, fall back to existing `dot_type`.

Legacy adapter (temporary, to be deleted in Phase 5):
- `dot_type_legacy(args)` calls `dot_type(args, &ResolveContext::empty())`

Note: Matmul does not take literal dims; do not add a ctx version unless a dimension argument exists.

#### 2.4 Math + Logical helpers
Files:
- `runmat/crates/runmat-runtime/src/builtins/math/type_resolvers.rs`
- `runmat/crates/runmat-runtime/src/builtins/logical/type_resolvers.rs`

Only add ctx-aware variants if they have literal arguments that affect shape. Otherwise, keep legacy only.
If ctx-aware variants are added, make the ctx version canonical (same name), and keep a thin `_legacy` adapter until Phase 5.

### Phase 3: Update builtins to call ctx-aware helpers

**Goal:** Switch builtins to the new ctx-aware helper functions while preserving existing behavior when literal context is missing.

Order within this phase:

1) **Array creation + shape** (highest impact on HIR tests)
   - `array/creation`: `colon`, `linspace`, `logspace`, `range`, `meshgrid`, `randi`, `randperm`, `fill`
   - `array/shape`: `reshape`, `cat`, `permute`, `ipermute`, `horzcat`, `vertcat`, `squeeze`

2) **Indexing helpers**
   - `array/indexing`: `find`, `sub2ind`, `ind2sub`

3) **Math reduction**
   - All reduction builtins (sum/mean/prod/min/max/cumsum/cumprod/etc.) use ctx-aware reduction helpers (same names).

4) **Linalg ops**
   - Only those where literal dimensions are meaningful (e.g., `dot` if dimension arg is present).

For each builtin:
- Keep `#[runtime_builtin(type_resolver(...))]`.
- Swap to the ctx-aware resolver function (same name, now with `ResolveContext` parameter).
- If a helper has a ctx-aware canonical version, the builtin must use it (no legacy fallback at the call site).

### Phase 4: Align direct shape_rules users

Some builtins implement shape logic directly. Replace with shared ctx-aware helpers where practical.

Target files:
- `runmat/crates/runmat-runtime/src/builtins/array/shape/cat.rs`
- `runmat/crates/runmat-runtime/src/builtins/array/shape/horzcat.rs`
- `runmat/crates/runmat-runtime/src/builtins/array/shape/reshape.rs`
- `runmat/crates/runmat-runtime/src/builtins/array/shape/permute.rs`
- `runmat/crates/runmat-runtime/src/builtins/array/shape/ipermute.rs`

Rules:
- Prefer `shape_rules::concat_shape` or the array helper if a shared version is added.
- Do not duplicate new shape logic in these files.

### Phase 5: Test pass and cleanup

Run:
- `cargo test -p runmat-hir --test type_inference`

Iterate only on the missing shapes or lint expectations revealed by the tests.

Finally, run:
- `rg "type_resolver_ctx" runmat` and ensure no remaining usage.

Legacy cleanup checklist (must be completed before finish):
- Remove all `_legacy` adapters from helper modules.
- Replace any remaining `_legacy` call sites with canonical ctx-aware helpers.
- Ensure `ResolveContext::empty()` is only used in deleted adapters (i.e., no production code references).
- Remove `type_resolver_context = true` flags and rely on `type_resolver(...)` only.

## Subagent Usage Rules

Subagents are allowed only for mechanical, localized edits once the plan is stable. Provide them this document and a precise, bounded task list. No subagent should:
- invent new behavior
- refactor unrelated code
- change HIR rules
- modify tests unless explicitly instructed

## Planned Builtin Targets by Family

Use this list as the authoritative target set for Phase 3 and 4 (do not expand without explicit approval):

Array creation:
- `colon`, `linspace`, `logspace`, `range`, `meshgrid`, `randi`, `randperm`, `fill`

Array shape:
- `reshape`, `cat`, `permute`, `ipermute`, `horzcat`, `vertcat`, `squeeze`

Indexing:
- `find`, `sub2ind`, `ind2sub`

Math reduction:
- `sum`, `mean`, `prod`, `min`, `max`, `cumsum`, `cumprod`, `cummin`, `cummax`, `diff`, `nnz`, `all`, `any`, `median`, `std`, `var`

Linalg:
- `dot` (only if literal dimension is used)

## Known Failure Targets (HIR tests)

These failing tests should be resolved by this migration if the helpers are implemented correctly:
- `infer_matmul_shape_with_known_dims`
- `lint_shape_mismatches`
- `lint_dot_and_reshape`
- `lint_concat_mismatches`
- `lint_logical_index_mismatch`
- `lint_reduction_dim_out_of_range`

## Completion Criteria

1) All builtins use `type_resolver(...)` and no `type_resolver_ctx(...)` remains.
2) All targeted builtins use ctx-aware resolvers where specified.
3) No new HIR special casing is added.
4) HIR type inference tests pass.
5) No extra lint suppressions were introduced.

## Example Set (Full Walkthrough Sketch)

This example demonstrates how a full set should look on paper before implementation. It is intended to be copied and adapted by subagents.

Example target set: array creation and reduction.

Step 1: Extend ResolveContext (parser layer)
- Add literal capture for numeric vectors and strings so we can read `[2 3]`, "all", and boolean flags.
- HIR should populate these fields when literal expressions are present.

Step 2: Update helpers (canonical ctx versions)
- In `array/type_resolvers.rs`, convert:
  - `rank_from_dims_args` to accept `ResolveContext`
  - `tensor_type_from_rank` to accept `ResolveContext`
  - `row_vector_type` to accept `ResolveContext`
- In `math/reduction/type_resolvers.rs`, convert:
  - `reduction_shape_from_args` to accept `ResolveContext`
  - `reduce_numeric_type` to accept `ResolveContext`
  - `reduce_logical_type` to accept `ResolveContext`
- Add thin `_legacy` adapters that call the ctx versions with `ResolveContext::empty()`.

Step 3: Builtin updates (mechanical)
- `array/creation/linspace.rs`
  - Update `linspace_type` to accept `ResolveContext` and use `row_vector_type(ctx)`.
  - Keep `type_resolver(linspace_type)`.
- `array/creation/colon.rs`
  - Update `colon_type` to accept `ResolveContext` and use range length when literals exist.
  - Keep `type_resolver(colon_type)`.
- `math/reduction/sum.rs`
  - Replace resolver with `reduce_numeric_type(args, ctx)`.
  - Respect literal `"all"` or `dim` from context.

Step 4: Test focus
- Run `cargo test -p runmat-hir --test type_inference`.
- Verify that shapes in linting for `sum`, `reshape`, and `cat` are now concrete when literals are available.

Step 5: Remove legacy adapters
- Once all builtins in the target set use ctx resolvers, delete legacy helper functions and adapters.
