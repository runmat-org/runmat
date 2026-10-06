use runmat_builtins::Type;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

fn bool_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Bool
}

#[runtime_builtin(
    name = "hgsave",
    category = "plotting",
    summary = "Save figures or parent figures of graphics handles to RunMat FIG files.",
    keywords = "hgsave,figure save,graphics,fig",
    sink = true,
    suppress_auto_output = true,
    accel = "metadata",
    type_resolver(bool_type),
    descriptor(runmat_builtins::HGSAVE_DESCRIPTOR),
    integer_audit(runmat_builtins::HGSAVE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgsave_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    super::execute_hgsave(args).await
}
