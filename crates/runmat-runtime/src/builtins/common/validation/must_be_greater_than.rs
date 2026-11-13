//! Identity-owned mustBeGreaterThan admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_GREATER_THAN_CATALOG_ENTRY, MUST_BE_GREATER_THAN_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeGreaterThan",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_greater_than"
)]
async fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_GREATER_THAN_CATALOG_ENTRY,
        &MUST_BE_GREATER_THAN_ERROR_OUTPUTS,
        args.view(),
    )?;
    let passed = {
        super::validate_resident_metadata(&args[0])?;
        super::value_is_greater_than_values_async(&args[0], &args[1]).await?
    };
    super::check_validator("mustBeGreaterThan", passed)
}
