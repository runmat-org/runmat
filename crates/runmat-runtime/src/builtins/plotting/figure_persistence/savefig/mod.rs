use runmat_builtins::Type;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

fn bool_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "savefig",
    category = "plotting",
    summary = "Save figures as RunMat FIG files.",
    keywords = "savefig,figure save,plotting,fig",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(runmat_builtins::SAVEFIG_DESCRIPTOR),
    integer_audit(runmat_builtins::SAVEFIG_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn savefig_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    super::execute_savefig(args).await
}
