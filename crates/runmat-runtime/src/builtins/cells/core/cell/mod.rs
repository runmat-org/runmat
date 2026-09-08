mod allocation;
mod arguments;
mod error;
mod prototype;
mod spec;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(name = "cell", builtin_path = "crate::builtins::cells::core::cell")]
async fn cell_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    allocation::build(arguments::parse(args).await?)
}
