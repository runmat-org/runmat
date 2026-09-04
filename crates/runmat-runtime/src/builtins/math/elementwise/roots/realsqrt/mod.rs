mod errors;
mod host;
mod provider;
mod specs;
#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

use runmat_builtins::REALSQRT_ERROR_INVALID_INPUT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "realsqrt";

#[runtime_builtin(
    name = "realsqrt",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::roots::realsqrt"
)]
async fn realsqrt_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "complex input is not supported",
        )),
        Value::SparseTensor(sparse) => host::evaluate_sparse(sparse),
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "expected real single or double input",
        )),
        other => host::evaluate(other),
    }
}
