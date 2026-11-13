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
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::execute_openfig(runtime, requested_outputs, args).await
}

#[cfg(test)]
pub(super) async fn openfig_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let runtime = crate::context::legacy::active();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, runtime.as_ref())?;
    super::execute_openfig(runtime, crate::output_count::current_output_count(), args).await
}
