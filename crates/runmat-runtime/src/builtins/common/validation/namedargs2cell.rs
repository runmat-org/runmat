//! Identity-owned namedargs2cell admission and execution.
use super::{BuiltinResult, Value};
use runmat_builtins::{NAMEDARGS2CELL_CATALOG_ENTRY, NAMEDARGS2CELL_ERROR_OUTPUTS};
#[runmat_macros::runtime_builtin(
    name = "namedargs2cell",
    binding_variant = "default",
    builtin_path = "crate::builtins::common::validation::namedargs2cell"
)]
fn builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    evaluate(args)
}

pub(super) fn evaluate(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    super::admission::inputs(
        &NAMEDARGS2CELL_CATALOG_ENTRY,
        &NAMEDARGS2CELL_ERROR_OUTPUTS,
        args.view(),
    )?;
    super::namedargs2cell_value(args.into_iter().next().expect("admitted input"))
}
