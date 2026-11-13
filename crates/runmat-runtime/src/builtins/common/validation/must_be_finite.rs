//! Identity-owned mustBeFinite admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_FINITE_CATALOG_ENTRY, MUST_BE_FINITE_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeFinite",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_finite"
)]
async fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_FINITE_CATALOG_ENTRY,
        &MUST_BE_FINITE_ERROR_OUTPUTS,
        args.view(),
    )?;
    let passed = {
        super::validate_resident_metadata(&args[0])?;
        super::ensure_resident_extension(
            &args[0],
            &runmat_builtins::MUST_BE_FINITE_RESIDENT_EXTENSION,
            "mustBeFinite",
        )?;
        super::value_is_finite_async(&args[0]).await?
    };
    super::check_validator("mustBeFinite", passed)
}
