//! Identity-owned mustBeReal admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_REAL_CATALOG_ENTRY, MUST_BE_REAL_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeReal",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_real"
)]
async fn builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_REAL_CATALOG_ENTRY,
        &MUST_BE_REAL_ERROR_OUTPUTS,
        &args,
    )?;
    let passed = {
        super::validate_resident_metadata(&args[0])?;
        super::value_is_real_async(&args[0]).await?
    };
    super::check_validator("mustBeReal", passed)
}
