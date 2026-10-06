use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "openfig",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn openfig_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::execute_openfig(runtime, requested_outputs, args).await
}

#[cfg(test)]
pub(super) async fn openfig_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::execute_openfig(
        crate::context::legacy::active(),
        crate::output_count::current_output_count(),
        args,
    )
    .await
}
