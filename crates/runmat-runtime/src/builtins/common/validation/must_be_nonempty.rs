//! Identity-owned mustBeNonempty admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_NONEMPTY_CATALOG_ENTRY, MUST_BE_NONEMPTY_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeNonempty",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_nonempty"
)]
fn builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_NONEMPTY_CATALOG_ENTRY,
        &MUST_BE_NONEMPTY_ERROR_OUTPUTS,
        &args,
    )?;
    super::validate_resident_metadata(&args[0])?;
    super::check_validator(
        MUST_BE_NONEMPTY_CATALOG_ENTRY.identity.name,
        !super::value_is_empty(&args[0]),
    )
}
