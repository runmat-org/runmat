//! Identity-owned mustBeA admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_A_CATALOG_ENTRY, MUST_BE_A_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeA",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_a"
)]
fn builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    super::admission::inputs(&MUST_BE_A_CATALOG_ENTRY, &MUST_BE_A_ERROR_OUTPUTS, &args)?;
    let names = super::type_names_arg(&args, 1)?;
    super::validate_resident_metadata(&args[0])?;
    super::check_validator(
        MUST_BE_A_CATALOG_ENTRY.identity.name,
        super::must_be_a(&args[0], names)?,
    )
}
