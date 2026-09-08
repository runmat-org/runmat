//! Element-wise callback mapping across ordinary and provider-resident arrays.
//!
//! This implementation applies a scalar MATLAB function to every element of one or
//! more array inputs. Supported builtin callbacks can dispatch directly for
//! `gpuArray` inputs; other callbacks gather authoritative typed storage and
//! re-upload uniform numeric or logical output.

mod callback;
mod error;
mod error_context;
mod execution;
mod gpu;
mod input;
mod options;
mod output;
mod plan;
mod spec;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
pub(super) const BUILTIN_NAME: &str = "arrayfun";

#[runtime_builtin(
    name = "arrayfun",
    binding_variant = "default",
    builtin_path = "crate::builtins::acceleration::gpu::arrayfun"
)]
async fn arrayfun_builtin(func: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    execution::execute(func, rest).await
}

#[cfg(test)]
pub(crate) mod tests;
