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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::structs::core::orderfields"
)]
async fn orderfields_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    target: Value,
    order: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    execution::execute(runtime.as_ref(), target, order.view())
}

#[cfg(test)]
mod tests;
