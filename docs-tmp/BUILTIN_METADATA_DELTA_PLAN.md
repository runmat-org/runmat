# Builtin Metadata Delta Plan (Execution Cookbook)

## Objective

Define one repeatable migration loop that upgrades every builtin to first-class typed metadata for LSP hover, completion, and signature help, without heuristic parsing and without per-builtin custom hacks.

This is the plan-of-record for the builtin metadata completion pass.

## Current Snapshot (Why This Pass Exists)

Repository state today:

1. `runtime_builtin` declarations: `568`.
2. Entries with `summary`: `442`.
3. Entries with `keywords`: `441`.
4. Entries with `category`: `478`.
5. Entries with `type_resolver`: `481`.
6. Entries using rich doc fields (`errors`/`related`/`examples`/`introduced`/`status`): `29`.

Implication:

1. Registration exists for nearly all builtins, but metadata depth is inconsistent.
2. LSP still relies on markdown/json text parsing for syntax-like output in several paths.
3. Signature help currently favors user functions and does not use a first-class builtin signature model.

## Scope

In scope:

1. `runmat-builtins` typed builtin metadata model.
2. Macro-side registration and declaration ergonomics for builtin metadata.
3. LSP consumption path for hover/completion/signature help from typed builtin metadata.
4. Validation/lints/tests that enforce completeness and parity.

Out of scope:

1. Runtime execution behavior changes.
2. Parser/lowering semantic behavior changes.
3. Non-builtin LSP architecture items already completed.

## Target End State (Data Model + Ownership)

### A) Code Is Source-Of-Truth For Typed Signatures

Each builtin has typed metadata in code (inventory-registered, wasm-compatible), analogous to how GPU/Fusion specs are registered today.

### B) `builtins-json` Stays Narrative

`builtins-json` continues to provide long-form docs, examples, FAQs, links, GPU notes, and marketing/reference text. It is no longer parsed for primary signature shape or argument structure.

### C) LSP Uses Typed Metadata First

1. Hover starts with typed signature block from code metadata.
2. Completion detail/label derives from typed signatures.
3. Signature help for builtin calls is sourced from typed signatures with active parameter index.
4. Narrative sections are appended from JSON/doc fields.

## Macro-Side Contract (Plan-Of-Record)

### 1) Add a Builtin Descriptor Model (Attached To BuiltinFunction)

Add a descriptor type in `runmat-builtins` and attach it to each builtin registration:

1. `BuiltinDescriptor` (typed signatures + output mode + completion policy + stable errors).
2. `BuiltinFunction` gains `descriptor: Option<&'static BuiltinDescriptor>`.
3. `runtime_builtin` macro accepts `descriptor(path::TO_DESCRIPTOR)` and attaches it at registration time.

Rationale:

1. Single builtin object remains the source-of-truth for runtime + LSP contract.
2. No dual-registry drift between function registration and metadata registration.
3. Enables wave-by-wave migration while preserving fallback behavior for unmigrated builtins.

### 2) Keep `runtime_builtin` Focused On Execution Binding

`runtime_builtin` remains the function registration macro (name, wrapper, runtime binding, resolver path), now with optional descriptor attachment.

Rationale:

1. Execution binding and typed signature metadata evolve together at declaration sites.
2. Keeps attribute noise low while avoiding ad-hoc LSP heuristics.
3. Allows incremental migration with zero behavior change for builtins without descriptors.

### 3) Add Shared Metadata Constructor Helpers

Create shared constructors/macros for common patterns:

1. `meta_sig!` for overload declaration.
2. `meta_arg!` and `meta_out!` for named typed inputs/outputs.
3. Reusable type aliases/pattern helpers for common domains:
   - numeric scalar / numeric tensor / logical tensor / string scalar / char row / struct / cell
   - row-vector, column-vector, matrix, nd-tensor, dims-vector.
4. Reusable behavior fragments for frequent options:
   - `"like"` prototype option
   - reduction nan-mode (`"omitnan"` / `"includenan"`)
   - dim selectors (`dim`, `vecdim`, `"all"`)
   - sink/output suppression descriptors for plotting and side-effect builtins.

Rationale:

1. Prevents duplication across hundreds of builtins.
2. Aligns signatures with existing shared type-resolver helper modules.
3. Makes migration loop mechanical and auditable.

### 4) Exact Metadata Shape (Canonical Fields)

Plan-of-record shape (code-level):

```rust
pub struct BuiltinDescriptor {
    pub signatures: &'static [BuiltinSignatureDescriptor],
    pub output_mode: BuiltinOutputMode,
    pub completion_policy: BuiltinCompletionPolicy,
    pub errors: &'static [BuiltinErrorDescriptor],
}

pub struct BuiltinSignatureDescriptor {
    pub label: &'static str,
    pub inputs: &'static [BuiltinParamDescriptor],
    pub outputs: &'static [BuiltinParamDescriptor],
}

pub struct BuiltinParamDescriptor {
    pub name: &'static str,
    pub ty: BuiltinParamType,
    pub arity: BuiltinParamArity,
    pub default: Option<&'static str>,
    pub description: &'static str,
}

pub enum BuiltinParamArity {
    Required,
    Optional,
    Variadic,
}

pub struct BuiltinErrorDescriptor {
    pub code: &'static str,
    pub identifier: Option<&'static str>,
    pub when: &'static str,
    pub message: &'static str,
}

pub enum BuiltinOutputMode {
    Fixed,
    ByRequestedOutputCount,
}

pub enum BuiltinCompletionPolicy {
    Public,
    HiddenInternal,
    MethodOnly,
}
```

Notes:

1. We keep this shape intentionally minimal and stable.
2. It is enough for LSP quality targets without adding noisy/unused fields.
3. Category/summary/docs remain on `BuiltinFunction` + `builtins-json`; descriptor focuses on typed call contract.
4. `output_mode` is required for MATLAB-style builtins where `nargout` affects behavior.
5. `completion_policy` controls discoverability for method/internal builtins without hiding metadata.

### 5) Macro/Helper Usage Template (Per Builtin File)

Canonical layout in builtin files (next to `GPU_SPEC`/`FUSION_SPEC`):

```rust
pub const SIN_DESCRIPTOR: BuiltinDescriptor = meta_profiles::numeric_unary_with_like(
    "sin",
    meta_profiles::errors::NUMERIC_INPUT_REQUIRED,
);

#[runtime_builtin(
    name = "sin",
    ...,
    descriptor(crate::builtins::math::trigonometry::sin::SIN_DESCRIPTOR),
    ...
)]
async fn sin_builtin(...) { ... }
```

For non-profiled/unique builtins:

```rust
pub const FOO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[/* explicit overloads */],
    output_mode: BuiltinOutputMode::Fixed,
    errors: &[/* explicit error rows */],
    completion_policy: BuiltinCompletionPolicy::Public,
};

#[runtime_builtin(
    name = "foo",
    ...,
    descriptor(crate::...::FOO_DESCRIPTOR),
    ...
)]
async fn foo_builtin(...) { ... }
```

Rule:

1. Prefer profile constructor.
2. Fall back to explicit struct only when behavior is uniquely shaped.
3. `BuiltinErrorDescriptor` rows are the only stable error source-of-truth inside migrated builtin files:
   - define `code`/`identifier`/`message` literals directly in the descriptor row,
   - throw stable branches through `foo_error(&FOO_ERROR_...)`,
   - never mirror those stable literals in separate `const IDENT_*`, `const *_CODE`, or `const *_MESSAGE` constants.
4. For runtime errors, keep one per-builtin source-of-truth row per stable error in `BuiltinErrorDescriptor` constants, and reuse those constants when throwing runtime errors.
5. Source-of-truth is the descriptor row itself: identifier/message/code must be authored once in `BuiltinErrorDescriptor` and referenced from throw helpers (`foo_error(&FOO_ERROR_...)`), never duplicated as separate stable constants or repeated literal throw strings.
6. Do not add separate `IDENT_*` / `*_MESSAGE` constants for migrated builtins when they duplicate descriptor rows; identifier/message live in the descriptor row and runtime helpers consume that row.
7. Runtime error builders must not use fallback literal identifiers (for example `unwrap_or("RunMat:...")`) when descriptor rows exist; only attach `error.identifier` directly from the row.
8. Do not add standalone `*_ERROR` message constants when the text is the stable branch message; place the text only in the descriptor row and throw via that row.
9. In migrated builtins, `BuiltinErrorDescriptor` constants are the in-file source of truth for stable identifier/message pairs. Throw sites must reference those constants, never duplicate the same identifier/message text.
10. If another module needs to branch on a migrated builtin error, branch on descriptor identifier (`err.identifier() == FOO_ERROR_BAR.identifier`), never on `err.message()` and never via a duplicated forwarded message constant.
11. Migration audit guardrail: migrated builtins must not define standalone stable error constants for identifier/message/code outside descriptor rows (for example `const ...: &str = "RunMat:..."`, `const ...: &str = "foo: ..."`, or `const ...: &str = "RM.FOO.BAR"` when they mirror a descriptor row). Keep stable identifier/message/code authored only in descriptor rows.
12. Stable-branch throw-sites must call `foo_error(&FOO_ERROR_...)` (or equivalent) so the emitted message/code/identifier come from the descriptor row; do not restate the same literal message string or parallel code/identifier constants in the branch.
13. Descriptor rows must own literal stable fields directly:
   - required: `code: "RM.FOO.BAR"`, `identifier: Some("RunMat:foo:Bar")`, `message: "foo: ..."`
   - disallowed: `identifier: Some(IDENT_FOO_BAR)`, `code: FOO_BAR_CODE`, `message: FOO_BAR_MESSAGE`
14. Keep any extra constants detail-only:
   - allowed: `const ..._DETAIL: &str = "expects positive integer"`
   - disallowed: any const that mirrors stable branch `code`/`identifier`/`message`
15. Keep non-stable detail strings separate from stable messages:
   - allowed: `const ..._DETAIL: &str = "file identifier must be finite"` used with `foo_error_with_detail(&FOO_ERROR_INVALID_INPUT, ..._DETAIL)`.
   - disallowed: `const ..._MESSAGE: &str = "foo: invalid input"` mirroring `FOO_ERROR_INVALID_INPUT.message`.
   - disallowed: `const INVALID_IDENTIFIER_TEXT = "...";` when it duplicates `FOO_ERROR_INVALID_IDENTIFIER.message`.
16. Keep the helper split explicit:
   - `foo_error(&FOO_ERROR_...)` for stable descriptor-backed branches (including the branch's canonical message text).
   - `foo_internal_error(...)` (or `foo_error_with(&FOO_ERROR_INTERNAL, ...)`) for contextual/internal detail text.
17. Canonical in-file source-of-truth:
   - Declare each stable branch as one `const FOO_ERROR_BAR: BuiltinErrorDescriptor = ...`.
   - Build `FOO_ERRORS` from those constants.
   - Throw only through helpers that accept `&'static BuiltinErrorDescriptor`.
   - Do not create parallel `const IDENT_*`, `const *_CODE`, or `const *_MESSAGE` mirrors.

18. Exact anti-pattern to reject (this is not allowed in migrated builtins):

```rust
const IDENT_INVALID_INPUT: &str = "RunMat:foo:InvalidInput";
const ERROR_INVALID_INPUT_MESSAGE: &str = "foo: invalid input";

const FOO_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOO.INVALID_INPUT",
    identifier: Some(IDENT_INVALID_INPUT),
    when: "...",
    message: ERROR_INVALID_INPUT_MESSAGE,
};
```

19. Canonical replacement (required shape):

```rust
const FOO_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOO.INVALID_INPUT",
    identifier: Some("RunMat:foo:InvalidInput"),
    when: "...",
    message: "foo: invalid input",
};

return Err(foo_error(&FOO_ERROR_INVALID_INPUT));
```

Audit command (must stay clean):

1. `rg -n "const IDENT_|const [A-Z0-9_]+_(MESSAGE|CODE): &str" crates/runmat-runtime/src/builtins`
2. `rg -n 'with_identifier\\("RunMat:' crates/runmat-runtime/src/builtins`
3. `cargo test -p runmat-runtime descriptor_error_source_of_truth`

## Shared Helper Reuse Strategy

Use existing resolver/helper modules as migration anchors, not bespoke one-offs.

Primary reuse anchors:

1. `crates/runmat-runtime/src/builtins/array/type_resolvers.rs` for constructor/dims/vector shape patterns.
2. `crates/runmat-runtime/src/builtins/math/reduction/type_resolvers.rs` for reduction argument semantics.
3. `crates/runmat-runtime/src/builtins/io/type_resolvers.rs` for polymorphic IO return domains.
4. `crates/runmat-runtime/src/builtins/plotting/type_resolvers.rs` for plotting-handle return conventions.

Rule:

1. If multiple builtins already share a type resolver pattern, define one shared metadata profile and reference it from each builtin.
2. Only define builtin-local metadata when behavior is genuinely unique.

## Field Population Rules (No Ambiguity)

For every builtin, fill metadata fields from these sources in this order:

1. `name`: exact registered builtin name from `runtime_builtin(name = "...")`.
2. `category`: existing runtime category if present; otherwise canonical family category.
3. `summary`: existing runtime summary if accurate; otherwise tighten from builtin JSON summary.
4. `signatures`: derived from real runtime argument parser branches, not from docs prose.
5. `errors`: derived from actual runtime error builders/messages and identifiers in code.
6. Sink/auto-output behavior: from `runtime_builtin` flags on `BuiltinFunction`.
7. Semantics/effects/purity/async: from `runmat_builtins::semantics` (single source-of-truth).
8. GPU residency behavior: from `GPU_SPEC.residency` when GPU spec exists.
9. `examples`: short code snippets from JSON/examples or minimal code-path tests.
10. `related`: from JSON links for builtin-to-builtin links only (external docs remain narrative).
11. `output_mode`: from runtime branching on requested output count.
12. `completion_policy`: from builtin role (`Public`, `MethodOnly`, `HiddenInternal`).

Disallowed source:

1. Do not reverse-engineer signatures from markdown behavior paragraphs.
2. Do not invent error codes ad hoc; use stable code constants with one-to-one mapping to runtime error branches.
3. Do not duplicate stable identifier/message/code in separate ad-hoc constants; descriptor rows are the canonical source for migrated builtins.
4. Do not build runtime errors with hard-coded identifiers/messages when a descriptor row already exists for that branch.
5. When remapping parser/broadcast/helper failures into builtin error branches, throw via the descriptor row helper (`*_error(&FOO_ERROR_...)`) rather than `format!(...)` message copies.
6. If a branch needs extra context in the text, wrap the descriptor message (`format!("{base}: {detail}")`) while still anchoring the branch to that descriptor row.
7. Canonical throw helper pattern:
   - `fn foo_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError { foo_error_with_message(error.message, error) }`
   - `fn foo_error_with_detail(error: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError { foo_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error) }`
   - Use `foo_error(&FOO_ERROR_...)` for stable branches.
   - Use `foo_error_with_detail(&FOO_ERROR_..., "...")` (or `format!(...)`) only for contextual/internal details.
   - Do not call `foo_error_with_detail(&FOO_ERROR_..., "foo: ...")` with a full stable message literal; pass only suffix detail text (for example `"unsupported datatype 'x'"`).
   - Do not repeat the stable branch message prefix in callsites (avoid literal `"foo: ..."` throw strings once descriptor rows exist).
9. Post-migration hygiene check for each touched builtin file:
   - no `const IDENT_*` or `const *_ERROR: &str` for stable identifier/message rows
   - no standalone `const ERROR_*: &str = "foo: ..."` branch-message constants that duplicate `BuiltinErrorDescriptor.message`
   - no direct `build_runtime_error("...stable branch message...")` in migrated runtime paths
   - all stable branches throw through descriptor-row helpers
   - if an error row exists, any contextual variant must use `foo_error_with_detail(&FOO_ERROR_..., detail)` so the prefix text still comes from the descriptor row
10. Repo-level audit command for migrated builtins (must be clean before commit):
   - `for f in $(rg -l "BuiltinErrorDescriptor" crates/runmat-runtime/src/builtins); do`
   - `  rg -n 'const\\s+((IDENT|MESSAGE_ID|ERROR|CODE)_[A-Z0-9_]+|[A-Z0-9_]+_(IDENT|CODE|MESSAGE|ERROR)):\\s*&str\\s*=\\s*"' "$f" && echo "^^ remove duplicated stable strings in $f";`
   - `  rg -n 'identifier:\\s*Some\\([A-Z0-9_]+\\)' "$f" && echo "^^ use inline identifier text in descriptor row, not a forwarded const in $f";`
   - `  rg -n '^(\\s*)(code|message):\\s*[A-Z0-9_]+\\b' "$f" && echo "^^ use inline literal code/message in descriptor row, not forwarded constants in $f";`
   - `done`

## Per-Builtin Execution Loop (Exact Procedure)

For each builtin `B`, follow this exact loop:

1. Classify `B` into a metadata family profile (unary numeric, reduction, shape transform, plotting sink, IO polymorphic, workspace/meta, object method).
2. Reuse or add shared profile fragments first (never duplicate ad-hoc blocks inline).
3. Add a `BuiltinDescriptor` constant near `GPU_SPEC`/`FUSION_SPEC` in the builtin file and attach it via `runtime_builtin(... descriptor(path::...))`.
4. Populate signatures with named params/outputs and explicit arity.
5. Encode options/default/variadic semantics in typed fields.
6. Encode error identifiers, stable error codes, and trigger text from actual runtime errors used by `B`.
   - Define named per-error descriptor constants (for example `FOO_ERROR_INVALID_INPUT`) and build `FOO_ERRORS` from those constants (no inline anonymous descriptor rows).
   - Runtime throw helpers must take a descriptor row (or descriptor-derived identifier) rather than re-declaring identifier/message literals.
   - If a shared helper in another module needs error context, pass `&BuiltinErrorDescriptor` into that helper instead of exporting free-floating identifier/message constants.
   - Do not retain legacy fallback identifier literals in helper builders once descriptor constants exist.
7. Keep execution traits sourced from `BuiltinFunction` + semantics + GPU spec; descriptor does not duplicate these fields.
8. Keep JSON narrative unchanged unless it contradicts typed metadata.
9. Set `output_mode` and `completion_policy` explicitly for `B`.
10. Add/adjust LSP tests for hover/completion/signature for `B`.
11. Run parity tests (native + wasm) for `B` and group.
12. Pass metadata validator checks for `B` and profile family.

Completion for `B` requires all twelve steps.

## Family Profiles (What We Reuse)

### Profile 1: Numeric Unary Elementwise

Use for: `sin`, `cos`, `tan`, `abs`, etc.

Inputs/outputs baseline:

1. Input: scalar or numeric/logical/complex tensor-like.
2. Options: optional `"like"` prototype where supported.
3. Output: same shape domain; class coercion rule explicit.

Shared hooks:

1. Math unary resolver (for shape/class passthrough).
2. Unary GPU/Fusion traits from existing specs.

### Profile 2: Numeric Reduction

Use for: `sum`, `mean`, `max`, `min`, `prod`, etc.

Inputs/outputs baseline:

1. Primary input array.
2. Optional dim/vecdim/`"all"`.
3. Optional nan mode.
4. Optional output template (`"native"`, `"double"`, `"like"` where supported).

Shared hooks:

1. `math/reduction/type_resolvers.rs`.
2. Shared reduction option fragment.

### Profile 3: Shape Transform / Concatenation

Use for: `reshape`, `permute`, `cat`, `horzcat`, `vertcat`, `repmat`.

Inputs/outputs baseline:

1. Shape-driving args are typed as dims vector or variadic dims.
2. Output shape relation encoded declaratively.
3. `"like"` behavior encoded once for all supporting shape builtins.

Shared hooks:

1. `array/type_resolvers.rs`.
2. Shared dimension argument helper.

### Profile 4: Plotting Sink

Use for: `plot`, `scatter`, `scatter3`, `line3`, etc.

Inputs/outputs baseline:

1. Positional data args.
2. Style/property variadic args.
3. Optional leading axes handle.
4. Output: handle scalar; suppress-auto-output semantics explicit.

Shared hooks:

1. Plot handle return profile.
2. Sink/presentation effect traits.

### Profile 5: IO/Filesystem/Network Polymorphic

Use for: `load`, `fopen`, `fread`, `webread`, etc.

Inputs/outputs baseline:

1. Explicit overloads for output-count-sensitive behavior.
2. Host-only vs GPU-promotion-later traits explicit.
3. Error surface rich and stable.

Shared hooks:

1. IO resolver helpers (`io/type_resolvers.rs`).
2. Multi-output signature model for MATLAB-style `[a,b,...] = ...`.

## Worked Migrations (Studied Builtins)

### `sin` (math unary)

Current:

1. `runtime_builtin` has summary/category/keywords/type_resolver.
2. Actual runtime supports optional `"like"` form.

Migration:

1. Attach unary profile metadata.
2. Define overloads:
   - `y = sin(x)`
   - `y = sin(x, "like", proto)`
3. Record `"like"` constraints and related runtime errors.
4. Reuse `numeric_unary_type` mapping instead of duplicating shape text.

Signature matrix:

1. `y = sin(x)`
2. `y = sin(x, "like", prototype)`

### `sum` (reduction)

Current:

1. Runtime supports dim selectors, nan mode, template controls.
2. Resolver already parses contextual tokens for shape behavior.

Migration:

1. Attach reduction profile metadata.
2. Explicit overload set for common forms:
   - `y = sum(A)`
   - `y = sum(A, dim)`
   - `y = sum(A, vecdim)`
   - `y = sum(A, "all")`
   - variants with `"omitnan"`/`"includenan"` and output template.
3. Map option compatibility constraints into metadata validation.

Signature matrix:

1. `y = sum(A)`
2. `y = sum(A, dim)`
3. `y = sum(A, vecdim)`
4. `y = sum(A, "all")`
5. `y = sum(A, __, "omitnan"|"includenan")`
6. `y = sum(A, __, "native"|"double"|"like", prototype)` where supported

### `reshape` and `cat` (shape)

Current:

1. Both already declare resolver paths and include `"like"`/dim semantics in runtime logic.

Migration:

1. Reuse shape profile + dims helper fragment.
2. `reshape` overloads cover vector dims + variadic dims + `[]` inference note.
3. `cat` overloads include `dim` + variadic arrays + optional `"like"` prototype.
4. Encode GPU mixing constraint (`cat` host/gpu mix errors) in error surface.

Signature matrix:

1. `B = reshape(A, [d1 d2 ...])`
2. `B = reshape(A, d1, d2, ...)`
3. `B = cat(dim, A1, A2, ..., An)`
4. `B = cat(dim, A1, ..., An, "like", prototype)` where runtime supports it

### `plot` and `scatter3` (plot sink)

Current:

1. Return handle scalar, sink behavior, style parsing in runtime.
2. Zero-copy GPU happy path exists when shared context exists.

Migration:

1. Reuse plotting sink profile.
2. Explicit overloads for shorthand vs explicit x/y(/z) forms.
3. Variadic style/property args modeled structurally, not prose-only.
4. Execution traits capture sink + suppress-auto-output + GPU residency mode.

Signature matrix:

1. `h = plot(y)`
2. `h = plot(x, y)`
3. `h = plot(ax, x, y, style?, Name,Value...)`
4. `h = scatter3(x, y, z)`
5. `h = scatter3(x, y, z, s?, c?, marker?, Name,Value...)`
6. `h = scatter3(ax, x, y, z, ...)`

### `load` and `fopen` (IO polymorphic)

Current:

1. Output behavior depends on call shape and requested outputs.
2. Rich runtime error surface exists.

Migration:

1. Use IO polymorphic profile with multiple output signatures.
2. `load` signatures separate workspace-populating call vs struct-return call.
3. `fopen` signatures include single output and multi-output query/list forms.
4. Carry runtime identifier/message mapping into typed error metadata.
5. Set `output_mode = ByRequestedOutputCount`.

Signature matrix:

1. `load(file, vars...)` (workspace-populating when output count is zero)
2. `S = load(file, vars...)`
3. `S = load(file, "-regexp", patterns...)`
4. `fid = fopen(filename, permission?, machinefmt?, encoding?)`
5. `[fid,msg] = fopen(...)`
6. `[fid,msg,machinefmt,encoding] = fopen(...)`
7. `filename = fopen(fid)`
8. `[filename,permission,machinefmt,encoding] = fopen(fid)`
9. `[fids, names, machinefmts, encodings] = fopen("all")`

### `pause` and `duration` (control/object-style edge cases)

Current:

1. `pause` mixes wait/query/state-toggle forms.
2. `duration` has constructor plus method-like dispatch (`duration.subsref`, `duration.subsasgn`, comparisons).

Migration:

1. `pause` gets explicit overloads and stateful execution traits.
2. `duration` constructor and method builtins each receive separate metadata specs keyed by exact builtin names.
3. Method builtins (`duration.subsref`, etc.) are treated as regular builtins with object-style first arg typing.
4. Method builtins default to `completion_policy = MethodOnly` unless explicitly promoted to `Public`.

Signature matrix:

1. `pause()`
2. `pause(seconds)`
3. `state = pause("on"|"off")`
4. `state = pause("query")`
5. `t = duration(h, m, s, Format?)`
6. `out = duration.subsref(obj, kind, payload)`
7. `obj = duration.subsasgn(obj, kind, payload, rhs)`

## LSP Integration Contract After Migration

Required wiring:

1. Completion detail uses typed builtin signature labels, not `param_types` debug fallback.
2. Hover header renders top typed overloads from metadata, then narrative docs.
3. Signature help resolves builtin callsites and parameter index from typed overloads.
4. JSON parsing remains only for long-form sections.
5. Builtin completion list respects `completion_policy`:
   - `Public`: always shown.
   - `MethodOnly`: shown only in method/member call contexts.
   - `HiddenInternal`: not shown in completion, still available for hover/definition when referenced.
6. Signature help respects `output_mode` for display text on polymorphic-output builtins.

No placeholder path remains for builtin signatures.

## Batch Execution Plan (Wave Loop)

For each wave:

1. Pick one family profile (for example: unary math).
2. Add/update shared profile fragments first.
3. Migrate every builtin in that family with the 12-step loop.
4. Add family-focused LSP tests (hover/completion/signature).
5. Run native + wasm parity tests.
6. Run metadata validator.
7. Land commit.

Suggested wave order:

1. Numeric unary + binary math.
2. Reductions/statistics.
3. Array/shape/indexing.
4. Plotting sinks + style-heavy builtins.
5. IO/filesystem/network.
6. Workspace/meta/object-method edge builtins.

Ownership/review contract per wave:

1. Assign one primary owner module for the family (for example, `math/reduction`, `io`, `plotting`).
2. Require one cross-family reviewer from LSP side for each wave.
3. Require one runtime reviewer to confirm signatures/errors map to real execution branches.

## Validation Gates

Gate A: Metadata completeness.

1. Every builtin has category, summary, at least one typed signature.
2. Every signature has named inputs and typed outputs.

Gate B: Signature quality.

1. No placeholder `name(...)` signatures.
2. No malformed arity or duplicate conflicting overloads.

Gate B2: Output-mode correctness.

1. Any builtin with output-count-dependent behavior must be `ByRequestedOutputCount`.
2. Any builtin marked `ByRequestedOutputCount` must have LSP signature-help tests for at least two output-count forms.

Gate C: LSP behavior.

1. Hover includes typed signature block.
2. Completion detail shows typed signature label.
3. Signature help works for builtin callsites and tracks active parameter.
4. Completion filtering obeys `completion_policy`.

Gate D: Native/wasm parity.

1. Same signatures and parameter labels for same corpus.
2. Same hover signature sections for same builtin.

Gate E: Error-code stability.

1. Every `BuiltinErrorDescriptor` has a stable `code`.
2. Codes are unique per builtin and deterministic across native/wasm.

## Validator Tooling

Add a machine-checked validator test/command that fails CI when:

1. A registered builtin expected in the migrated set is missing a corresponding attached `BuiltinDescriptor`.
2. A spec is missing required core fields.
3. `output_mode` is inconsistent with runtime output-count branching.
4. `completion_policy = MethodOnly` but no method-name shape (`Class.method` or equivalent) is present.
5. Error codes are missing/duplicated.

Plan command target:

1. `cargo test -p runmat-builtins metadata_validator`
2. `cargo test -p runmat-lsp builtin_metadata_contract`

## Anti-Hack Rules

1. Do not parse markdown behavior strings to recover signatures.
2. Do not encode signature structure as untyped ad-hoc strings.
3. Do not duplicate shared family metadata inline per builtin.
4. Do not special-case individual builtins in LSP renderers by name.
5. Do not duplicate semantics fields across metadata systems; use semantics as source-of-truth.
6. Do not expose method/internal builtins in completion unless policy says `Public`.

## Done Criteria

This initiative is complete when:

1. All builtins are migrated through this loop with family profiles.
2. Builtin signature help is fully typed in native and wasm.
3. Hover/completion/signature no longer rely on heuristic markdown parsing for primary signature content.
4. CI enforces metadata completeness and parity.
