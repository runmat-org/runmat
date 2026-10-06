use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "hgload",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "crate::builtins::plotting::figure_persistence"
)]
pub async fn hgload_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: Vec<Value>,
) -> BuiltinResult<runmat_value::ValueSequence> {
    super::execute_hgload(runtime, requested_outputs, args)
        .await
}

#[cfg(test)]
pub(super) async fn hgload_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::execute_hgload(
        crate::context::legacy::active(),
        crate::output_count::current_output_count(),
        args,
    )
    .await
    .map(crate::call::arguments::project_legacy_builtin_value_abi)
}
