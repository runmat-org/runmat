//! Identity-owned mustBeMember admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_MEMBER_CATALOG_ENTRY, MUST_BE_MEMBER_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeMember",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_member"
)]
async fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args).await
}
pub(super) async fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_MEMBER_CATALOG_ENTRY,
        &MUST_BE_MEMBER_ERROR_OUTPUTS,
        args.view(),
    )?;
    let passed = super::value_is_member_async(&args[0], &args[1]).await?;
    super::check_validator("mustBeMember", passed)
}
