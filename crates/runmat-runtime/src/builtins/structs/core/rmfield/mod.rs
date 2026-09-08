//! Top-level field removal over structure metadata.

mod error;
mod execution;
mod names;
mod spec;
mod target;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const BUILTIN_NAME: &str = "rmfield";

#[runtime_builtin(
    name = "rmfield",
    builtin_path = "crate::builtins::structs::core::rmfield"
)]
async fn rmfield_builtin(target: Value, fields: Vec<Value>) -> crate::BuiltinResult<Value> {
    execution::execute(target, fields)
}

#[cfg(test)]
mod tests;
