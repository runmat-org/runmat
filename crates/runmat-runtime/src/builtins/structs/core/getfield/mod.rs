//! MATLAB-compatible `getfield` builtin with struct-array and object support.

mod arguments;
mod errors;
mod execution;
mod object;
mod spec;
mod special;

#[cfg(test)]
pub(crate) mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

pub(crate) use execution::get_member_value;

#[cfg(test)]
pub(crate) use errors::is_undefined_function;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(
    name = "getfield",
    builtin_path = "crate::builtins::structs::core::getfield"
)]
async fn getfield_builtin(base: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    execution::getfield(base, rest, true).await
}

pub(crate) async fn getfield_internal(
    base: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    execution::getfield(base, rest, false).await
}
