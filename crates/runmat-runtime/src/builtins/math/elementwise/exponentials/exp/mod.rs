mod specs;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

use super::operation::ExponentialOperation;

pub(super) const OPERATION: ExponentialOperation = ExponentialOperation::Exp;

#[runtime_builtin(
    name = "exp",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::exponentials::exp"
)]
async fn exp_builtin(value: Value) -> BuiltinResult<Value> {
    super::engine::execute(OPERATION, value).await
}
