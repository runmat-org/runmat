use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "savefig",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn savefig_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    super::execute_savefig(runtime, args).await
}

#[cfg(test)]
pub async fn savefig_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    super::execute_savefig(crate::context::legacy::active(), args).await
}
