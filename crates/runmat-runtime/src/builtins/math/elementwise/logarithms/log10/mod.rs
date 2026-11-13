mod specs;
#[cfg(test)]
mod tests;
use super::operation::LogarithmOperation;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const OPERATION: LogarithmOperation = LogarithmOperation::Common;
#[runtime_builtin(
    name = "log10",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::logarithms::log10"
)]
async fn log10_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::LOG10_DESCRIPTOR,
        &runmat_builtins::LOG10_ERROR_TOO_MANY_OUTPUTS,
        "log10",
    )?;
    super::engine::execute(runtime.as_ref(), OPERATION, value).await
}
