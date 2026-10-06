use runmat_builtins::Type;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

fn bool_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "saveas",
    category = "plotting",
    summary = "Save a figure to FIG or PNG format.",
    keywords = "saveas,figure save,plotting,png,fig",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(runmat_builtins::SAVEAS_DESCRIPTOR),
    integer_audit(runmat_builtins::SAVEAS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn saveas_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    super::execute_saveas(args).await
}
