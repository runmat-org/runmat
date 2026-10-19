use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "savefig",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn savefig_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let status = super::execute_savefig(runtime, requested_outputs, args).await?;
    Ok(if requested_outputs == Some(0) {
        Value::output_list(Vec::new())
    } else {
        Value::Bool(status)
    })
}

#[cfg(test)]
pub(super) async fn savefig_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    super::execute_savefig(
        crate::context::legacy::active(),
        crate::output_count::current_output_count(),
        args,
    )
    .await
}
