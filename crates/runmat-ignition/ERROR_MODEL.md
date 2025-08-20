# Error Model (mex normalization)

All interpreter errors are normalized through `mex(id, message)`, which returns a single string in the form `"<ID>: <message>"`. Tests assert on identifiers to prevent drift.

## Standard identifiers
- MATLAB:UndefinedFunction - unknown function/builtin
- MATLAB:UndefinedVariable - unknown variable
- MATLAB:NotEnoughInputs / MATLAB:TooManyInputs - callsite arity
- MATLAB:TooManyOutputs / MATLAB:VarargoutMismatch - output arity
- MATLAB:SliceNonTensor - non-tensor slicing attempt
- MATLAB:IndexOutOfBounds - numeric index out of bounds
- MATLAB:CellIndexType / MATLAB:CellSubscriptOutOfBounds - cell indexing
- MATLAB:ExpandError - illegal cell/return expansion
- MATLAB:MissingSubsref / MATLAB:MissingSubsasgn - missing overloads on object

## Principles
- The same programmer error should produce the same identifier across code paths.
- Messages should be precise and actionable (mention function name, expected/got counts, etc.).
- VM sites must not emit raw strings; always wrap with `mex`.

See `vm.rs` for concrete mappings and tests in `tests/control_flow.rs` and `tests/indexing_properties.rs`.
