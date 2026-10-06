use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "hgload",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgload_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::execute_hgload(runtime, args).await
}

pub async fn hgload_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::execute_hgload(crate::context::legacy::active(), args).await
}
