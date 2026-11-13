use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "saveas",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn saveas_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let status = super::execute_saveas(runtime, requested_outputs, args).await?;
    Ok(if requested_outputs == Some(0) {
        Value::output_list(Vec::new())
    } else {
        Value::Bool(status)
    })
}

#[cfg(test)]
pub(super) async fn saveas_builtin(args: Vec<Value>) -> BuiltinResult<bool> {
    let runtime = crate::context::legacy::active();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, runtime.as_ref())?;
    super::execute_saveas(runtime, crate::output_count::current_output_count(), args).await
}
