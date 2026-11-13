//! Identity-owned mustBeNumericOrLogical admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{
    MUST_BE_NUMERIC_OR_LOGICAL_CATALOG_ENTRY, MUST_BE_NUMERIC_OR_LOGICAL_ERROR_OUTPUTS,
};
#[runmat_macros::runtime_builtin(
    name = "mustBeNumericOrLogical",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_numeric_or_logical"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_NUMERIC_OR_LOGICAL_CATALOG_ENTRY,
        &MUST_BE_NUMERIC_OR_LOGICAL_ERROR_OUTPUTS,
        args.view(),
    )?;
    super::validate_resident_metadata(&args[0])?;
    super::check_validator(
        MUST_BE_NUMERIC_OR_LOGICAL_CATALOG_ENTRY.identity.name,
        super::value_is_numeric_or_logical(&args[0]),
    )
}
