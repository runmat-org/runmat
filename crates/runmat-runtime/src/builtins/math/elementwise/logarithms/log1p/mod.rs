mod specs;
#[cfg(test)]
mod tests;

use super::operation::LogarithmOperation;
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const OPERATION: LogarithmOperation = LogarithmOperation::OnePlus;

#[runtime_builtin(
    name = "log1p",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::logarithms::log1p"
)]
async fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::LOG1P_DESCRIPTOR,
        &runmat_builtins::LOG1P_ERROR_TOO_MANY_OUTPUTS,
        "log1p",
    )?;
    super::engine::execute(OPERATION, value).await
}
