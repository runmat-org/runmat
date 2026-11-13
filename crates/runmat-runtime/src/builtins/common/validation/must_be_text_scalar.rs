//! Identity-owned mustBeTextScalar admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_TEXT_SCALAR_CATALOG_ENTRY, MUST_BE_TEXT_SCALAR_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeTextScalar",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_text_scalar"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_TEXT_SCALAR_CATALOG_ENTRY,
        &MUST_BE_TEXT_SCALAR_ERROR_OUTPUTS,
        args.view(),
    )?;
    super::admission::reject_resident_text(&MUST_BE_TEXT_SCALAR_CATALOG_ENTRY, &args[0])?;
    super::check_validator(
        MUST_BE_TEXT_SCALAR_CATALOG_ENTRY.identity.name,
        super::value_is_text_scalar(&args[0]),
    )
}
