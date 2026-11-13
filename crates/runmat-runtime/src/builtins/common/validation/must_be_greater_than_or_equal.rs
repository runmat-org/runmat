//! Identity-owned mustBeGreaterThanOrEqual admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{
    MUST_BE_GREATER_THAN_OR_EQUAL_CATALOG_ENTRY, MUST_BE_GREATER_THAN_OR_EQUAL_ERROR_OUTPUTS,
};
#[runmat_macros::runtime_builtin(
    name = "mustBeGreaterThanOrEqual",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_greater_than_or_equal"
)]
async fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_GREATER_THAN_OR_EQUAL_CATALOG_ENTRY,
        &MUST_BE_GREATER_THAN_OR_EQUAL_ERROR_OUTPUTS,
        args.view(),
    )?;
    let passed = {
        super::validate_resident_metadata(&args[0])?;
        super::value_is_greater_than_or_equal_values_async(&args[0], &args[1]).await?
    };
    super::check_validator("mustBeGreaterThanOrEqual", passed)
}
