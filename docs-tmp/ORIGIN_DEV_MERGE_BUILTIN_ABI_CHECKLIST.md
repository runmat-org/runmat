# Origin/Dev Merge: Builtin ABI + Metadata Post-Merge Checklist

This checklist is the plan-of-record for reconciling incoming `origin/dev` builtin changes with the current ABI/descriptor architecture.

## Scope

Incoming builtin-related changes that need ABI-era normalization:

- `crates/runmat-runtime/src/builtins/comms/qammod.rs`
- `crates/runmat-runtime/src/builtins/control/nyquist.rs`
- `crates/runmat-runtime/src/builtins/plotting/ops/contour3.rs`
- `crates/runmat-runtime/src/builtins/plotting/ops/contour.rs`
- `crates/runmat-runtime/src/builtins/io/filetext/filewrite.rs`
- `crates/runmat-runtime/src/builtins/io/tabular/csvwrite.rs`
- `crates/runmat-runtime/src/builtins/io/tabular/dlmwrite.rs`
- `crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs`
- `crates/runmat-runtime/src/builtins/array/shape/repelem.rs`

## Global Rules (Must Hold)

1. Descriptor-first metadata model:
   - Every migrated builtin declares `BuiltinDescriptor` + `BuiltinSignatureDescriptor[]` + `BuiltinErrorDescriptor[]`.
   - `#[runtime_builtin(...)]` includes `descriptor(...)`.
2. Error source of truth:
   - Stable identifier + default message come from the builtin's `BuiltinErrorDescriptor` constants.
   - Throw sites reuse descriptor-backed helpers; no duplicated identifier/message literals.
3. ABI output contract:
   - Runtime output behavior matches descriptor `output_mode`.
   - `ByRequestedOutputCount` builtins must explicitly handle `0`, `1`, `N` paths.
4. No viewport fallbacks in plotting data builders:
   - No implicit `(600, 400)` or similar synthetic viewport defaults.
5. Keep async I/O behavior from incoming wasm starvation fix:
   - Preserve `open_async`/`flush_async` behavior where introduced.

## Per-File Delta Checklist

## 1) `comms/qammod.rs`

Current gap:
- No descriptor/signature/error metadata attachment.

Required migration:
- Add descriptor constants:
  - signatures for `qammod(x, M, ...)` forms supported today.
  - `output_mode = BuiltinOutputMode::Fixed` (single output).
  - completion policy public.
- Add stable error descriptors for:
  - invalid modulation order
  - invalid symbol input/range
  - invalid options/name-value pairs
  - unsupported options/features (`InputType='bit'`, plot constellation)
  - unsupported input types
- Route all `qammod_error(...)` callsites through descriptor-backed helpers.
- Attach `descriptor(crate::builtins::comms::qammod::QAMMOD_DESCRIPTOR)` in macro.

Validation:
- Signature help + completion detail show descriptor labels.
- Error identifiers resolve to stable names from descriptor constants.

## 2) `control/nyquist.rs`

Current gap:
- Multi-output runtime logic exists, but no descriptor/signature/error metadata attachment.

Required migration:
- Add descriptor constants:
  - signatures for statement form and output forms:
    - `nyquist(sys)`
    - `nyquist(sys, w)`
    - `[re] = nyquist(sys, ...)`
    - `[re, im] = nyquist(sys, ...)`
  - `output_mode = BuiltinOutputMode::ByRequestedOutputCount`.
- Add stable error descriptors for:
  - invalid model/class
  - invalid coefficients and time/frequency inputs
  - unsupported delay/model capabilities
  - plotting/rendering failures
  - internal assembly issues
- Replace ad-hoc `nyquist_error(...)` string-only builder with descriptor-backed helpers.
- Keep existing runtime output branching (`current_output_count` / `requested_output_count`) aligned with descriptor contract.
- Attach descriptor in macro.

Validation:
- Runtime outputs and LSP signatures agree for 0/1/2/N request counts.
- Statement form returns empty `OutputList` when appropriate.

## 3) `plotting/ops/contour3.rs`

Current gap:
- No descriptor/signature/error metadata attachment.

Required migration:
- Add descriptor constants:
  - signature variants currently supported by parser.
  - `output_mode = BuiltinOutputMode::Fixed` (handle scalar output).
- Add stable error descriptors for:
  - invalid argument forms
  - invalid level/style options
  - render registration failures
  - GPU path fallback failures that surface as user-visible errors
- Convert error paths to descriptor-backed helper usage.
- Attach descriptor in macro.

Validation:
- LSP signature/hover uses descriptor labels.
- Handle-return behavior unchanged.

## 4) `plotting/ops/contour.rs`

Current merge risk:
- Incoming side contains contour parsing/geometry fixes, but may regress local descriptor conventions and viewport contract.

Required migration:
- Keep our descriptor/error source-of-truth style.
- Integrate incoming contour behavior fixes without reintroducing:
  - implicit viewport defaults
  - ad-hoc error literal duplication
- Ensure contour + contour3 share coherent option/error taxonomy where practical.

Validation:
- Existing contour tests pass.
- New contour3 behavior remains consistent with contour option parsing.

## 5) Async write builtins

Files:
- `io/filetext/filewrite.rs`
- `io/tabular/csvwrite.rs`
- `io/tabular/dlmwrite.rs`
- `io/tabular/writematrix.rs`

Current merge risk:
- Incoming async I/O behavior may regress descriptor-based error reporting.

Required migration:
- Keep `open_async`/`flush_async` codepaths.
- Preserve descriptor-backed error helper APIs and stable identifiers/messages.
- Normalize helper signatures so callsites pass descriptor constants rather than constructing message-only runtime errors.
- Ensure no duplicate error identifiers/messages are reintroduced in throw sites.

Validation:
- wasm async write starvation symptom remains fixed.
- Existing descriptor error source-of-truth tests remain green.

## 6) `array/shape/repelem.rs`

Current merge risk:
- Incoming branch has alternate error helper style and may diverge from current GPU factor handling decisions.

Required migration:
- Preserve current descriptor/error helper conventions.
- Preserve approved GPU factor parsing behavior and residency semantics.
- If incoming logic is adopted, re-home all errors onto descriptor-backed constants.

Validation:
- Existing `repelem` host/gpu factor tests remain green.

## LSP + Metadata Wiring Acceptance

After merge + migrations:

1. `runmat-lsp` signature help includes new descriptors for:
   - `qammod`
   - `nyquist`
   - `contour3`
2. Completion details for the same builtins show descriptor-backed call labels.
3. No fallback to legacy/generated metadata paths for these builtins.

## Test Plan

Run in this order:

1. `cargo fmt --all`
2. `cargo check --workspace`
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
4. `cargo test --workspace`
5. Targeted checks:
   - `cargo test -p runmat-runtime qammod`
   - `cargo test -p runmat-runtime nyquist`
   - `cargo test -p runmat-runtime contour`
   - `cargo test -p runmat-runtime contour3`
   - `cargo test -p runmat-lsp signature_help_uses_runtime_lib_descriptors`
6. Descriptor source-of-truth guard:
   - `cargo test -p runmat-runtime descriptor_error_source_of_truth`

## Done Criteria

1. All scoped files conform to descriptor-first + error source-of-truth rules.
2. Async write behavior from incoming wasm fix is preserved.
3. No viewport fallback defaults are introduced by merge resolution.
4. Full workspace fmt/check/clippy/tests are green.
5. LSP signature/completion behavior for new builtins is descriptor-backed and passing tests.
