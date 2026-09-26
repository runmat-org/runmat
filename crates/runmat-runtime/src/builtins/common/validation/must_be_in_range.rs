//! Identity-owned mustBeInRange admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_IN_RANGE_CATALOG_ENTRY, MUST_BE_IN_RANGE_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeInRange",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_in_range"
)]
async fn builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_IN_RANGE_CATALOG_ENTRY,
        &MUST_BE_IN_RANGE_ERROR_OUTPUTS,
        &args,
    )?;
    let passed = {
        let inclusivity = super::range_inclusivity_arg("mustBeInRange", &args[3..])?;
        super::validate_resident_metadata(&args[0])?;
        super::value_is_in_range_documented_async(&args[0], &args[1], &args[2], inclusivity).await?
    };
    super::check_validator("mustBeInRange", passed)
}
