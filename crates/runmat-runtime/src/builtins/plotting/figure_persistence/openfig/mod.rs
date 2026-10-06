use runmat_builtins::Type;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

fn handle_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Num
}

#[runtime_builtin(
    name = "openfig",
    category = "plotting",
    summary = "Open figures saved as RunMat FIG files.",
    keywords = "openfig,figure open,plotting,fig",
    suppress_auto_output = true,
    type_resolver(handle_type),
    descriptor(runmat_builtins::OPENFIG_DESCRIPTOR),
    integer_audit(runmat_builtins::OPENFIG_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn openfig_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::execute_openfig(args).await
}
