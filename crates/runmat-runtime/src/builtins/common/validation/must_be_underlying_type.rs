//! Identity-owned mustBeUnderlyingType admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{
    MUST_BE_UNDERLYING_TYPE_CATALOG_ENTRY, MUST_BE_UNDERLYING_TYPE_ERROR_OUTPUTS,
};
#[runmat_macros::runtime_builtin(
    name = "mustBeUnderlyingType",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_underlying_type"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_UNDERLYING_TYPE_CATALOG_ENTRY,
        &MUST_BE_UNDERLYING_TYPE_ERROR_OUTPUTS,
        args.view(),
    )?;
    let names = super::type_names_arg(args.view(), 1)?;
    super::validate_resident_metadata(&args[0])?;
    super::check_validator(
        MUST_BE_UNDERLYING_TYPE_CATALOG_ENTRY.identity.name,
        super::value_underlying_type_matches(&args[0], names)?,
    )
}
