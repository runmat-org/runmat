mod admission;
mod assembly;
mod error;
mod extract;
mod implementation;
mod input;
mod partition;
mod spec;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(
    name = "mat2cell",
    builtin_path = "crate::builtins::cells::core::mat2cell"
)]
async fn mat2cell_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    implementation::execute(value, rest).await
}
