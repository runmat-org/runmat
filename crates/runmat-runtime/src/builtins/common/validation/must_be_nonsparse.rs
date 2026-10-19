//! Identity-owned mustBeNonsparse admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{MUST_BE_NONSPARSE_CATALOG_ENTRY, MUST_BE_NONSPARSE_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "mustBeNonsparse",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::must_be_nonsparse"
)]
fn builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: Vec<Value>) -> BuiltinResult<Value> {
    super::admission::inputs(
        &MUST_BE_NONSPARSE_CATALOG_ENTRY,
        &MUST_BE_NONSPARSE_ERROR_OUTPUTS,
        &args,
    )?;
    super::check_validator(
        MUST_BE_NONSPARSE_CATALOG_ENTRY.identity.name,
        super::value_satisfies_nonsparse_validator(&args[0])?,
    )
}
