//! Top-level structure field ordering.

mod error;
mod execution;
mod order;
mod output;
mod spec;
mod target;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const BUILTIN_NAME: &str = "orderfields";

#[runtime_builtin(
    name = "orderfields",
    builtin_path = "crate::builtins::structs::core::orderfields"
)]
async fn orderfields_builtin(target: Value, order: Vec<Value>) -> crate::BuiltinResult<Value> {
    execution::execute(target, order.as_slice())
}

#[cfg(test)]
mod tests;
