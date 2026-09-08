//! Host-controlled callback mapping over scalar-structure fields.

mod callback;
mod error;
mod error_context;
mod execution;
mod fields;
mod options;
mod output;
mod spec;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const BUILTIN_NAME: &str = "structfun";

#[runtime_builtin(
    name = "structfun",
    execution_stack = "process",
    builtin_path = "crate::builtins::structs::core::structfun"
)]
async fn structfun_builtin(
    function: Value,
    structure: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    execution::execute(function, structure, rest).await
}

#[cfg(test)]
mod tests;
