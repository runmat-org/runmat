use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "hgsave",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgsave_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    super::execute_hgsave(runtime, args).await
}

pub async fn hgsave_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    super::execute_hgsave(crate::context::legacy::active(), args).await
}
