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

pub(super) const OPERATION: LogarithmOperation = LogarithmOperation::Natural;
#[runtime_builtin(
    name = "log",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::logarithms::log"
)]
async fn log_builtin(value: Value) -> BuiltinResult<Value> {
    super::engine::execute(OPERATION, value).await
}
