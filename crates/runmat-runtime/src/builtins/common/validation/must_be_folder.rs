//! Identity-owned mustBeFolder admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_FOLDER_CATALOG_ENTRY, MUST_BE_FOLDER_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeFolder",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_folder"
)]
async fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_FOLDER_CATALOG_ENTRY,
        &MUST_BE_FOLDER_ERROR_OUTPUTS,
        args.view(),
    )?;
    let passed = {
        super::admission::reject_resident_text(&MUST_BE_FOLDER_CATALOG_ENTRY, &args[0])?;
        super::value_is_folder_async(&args[0]).await?
    };
    super::check_validator("mustBeFolder", passed)
}
