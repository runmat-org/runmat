//! Identity-owned isvarname admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{ISVARNAME_CATALOG_ENTRY, ISVARNAME_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "isvarname",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::isvarname"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &ISVARNAME_CATALOG_ENTRY,
        &ISVARNAME_ERROR_OUTPUTS,
        args.view(),
    )?;
    Ok(Value::Bool(super::isvarname_value(&args[0])))
}
