# Error Model (mex normalization)

Ignition normalizes all runtime errors to MATLAB-style identifiers via a tiny helper `mex(id, message)`. The VM constructs a canonical string `"<IDENTIFIER>: <message>"` and uses it consistently across all error sites. Tests assert on identifiers to prevent drift.

## Namespacing

`ERROR_NAMESPACE` is currently `"MATLAB"` for all errors emitted by the VM. It is an arbitrary string defined in `vm.rs` and used to prefix all error identifiers, and left as `"MATLAB"` to ensure compatibility with existing post-script code that may exist in existing MATLAB codebases.

## Standard identifiers and when they occur

- MATLAB:UndefinedFunction — unknown function or builtin at call
- MATLAB:UndefinedVariable — unknown variable (primarily a frontend HIR concern)
- MATLAB:NotEnoughInputs / MATLAB:TooManyInputs — function arity mismatch (user functions and some builtins)
- MATLAB:TooManyOutputs — requesting more outputs than a function defines
- MATLAB:VarargoutMismatch — requesting more varargout elements than available
- MATLAB:SliceNonTensor — attempted slicing on a non-tensor/non-string-array value
- MATLAB:IndexOutOfBounds — numeric index out of bounds
- MATLAB:SubscriptOutOfBounds — 2-D scalar subscript out of bounds
- MATLAB:IndexShape — logical mask shape mismatch
- MATLAB:IndexStepZero — range step was 0 in `end` arithmetic forms
- MATLAB:CellIndexType — unsupported cell index type
- MATLAB:CellSubscriptOutOfBounds — 2-D cell subscripting out of bounds
- MATLAB:ExpandError — illegal/unsupported argument expansion
- MATLAB:MissingSubsref / MATLAB:MissingSubsasgn — object lacks the overload
- MATLAB:ShapeMismatch — element-wise comparison or assignment shape mismatch
- MATLAB:CharError — failed to materialize CharArray

The VM also produces plain strings for some internal plumbing; they are wrapped by `vm_bail!` into the try/catch machinery so callers see normalized identifiers at the escape boundary.

## Try/Catch, rethrow, and exception objects

- `EnterTry(catch_pc, catch_var?)` pushes a try frame; on failure, control jumps to `catch_pc`
- If `catch_var` is present, the VM binds the caught exception into the given variable as `Value::MException`
- `PopTry` pops the top try frame on successful completion of the try block
- `rethrow` with no argument rethrows the last caught exception (tracked thread-locally)

Internally, `parse_exception` splits `"IDENT: message"` using the last `": "`, preserving nested identifiers.

## Guidelines for contributors

- Prefer one canonical site per error condition and share it via helpers
- Include actionable details (function name, expected vs. got, etc.)
- Do not emit raw strings from VM instruction handlers; always normalize via `mex`
- Keep test coverage in sync with error identifiers to prevent regressions

See `vm.rs` for concrete mappings and tests under ignition and runtime crates (e.g., `tests/control_flow.rs`, `tests/indexing_properties.rs`).
