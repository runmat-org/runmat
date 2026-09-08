//! Exact field-name queries over structure metadata.

mod error;
mod execution;
mod names;
mod output;
mod spec;
mod target;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const BUILTIN_NAME: &str = "isfield";

#[runtime_builtin(
    name = "isfield",
    builtin_path = "crate::builtins::structs::core::isfield"
)]
async fn isfield_builtin(target: Value, names: Value) -> crate::BuiltinResult<Value> {
    execution::execute(&target, names)
}

#[cfg(test)]
mod tests;
