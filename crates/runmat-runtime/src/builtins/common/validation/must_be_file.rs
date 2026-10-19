//! Identity-owned mustBeFile admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_FILE_CATALOG_ENTRY, MUST_BE_FILE_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeFile",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_file"
)]
async fn builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_FILE_CATALOG_ENTRY,
        &MUST_BE_FILE_ERROR_OUTPUTS,
        &args,
    )?;
    let passed = {
        super::admission::reject_resident_text(&MUST_BE_FILE_CATALOG_ENTRY, &args[0])?;
        super::value_is_file_async(&args[0]).await?
    };
    super::check_validator("mustBeFile", passed)
}
