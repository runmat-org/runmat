mod errors;
mod operation;
mod provider;
mod specs;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

const BUILTIN_NAME: &str = "nextpow2";

#[runtime_builtin(
    name = "nextpow2",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::powers_of_two::nextpow2"
)]
async fn nextpow2_builtin(value: Value) -> BuiltinResult<Value> {
    errors::reject_excess_outputs()?;
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        other => operation::host(other),
    }
}
