use runmat_builtins::Type;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

fn handle_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Num
}

#[runtime_builtin(
    name = "hgload",
    category = "plotting",
    summary = "Load graphics object hierarchies from RunMat FIG files.",
    keywords = "hgload,figure load,graphics,fig",
    type_resolver(handle_type),
    descriptor(runmat_builtins::HGLOAD_DESCRIPTOR),
    integer_audit(runmat_builtins::HGLOAD_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgload_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::execute_hgload(args).await
}
