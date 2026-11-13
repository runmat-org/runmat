//! Host-controlled callback mapping over cell-array contents.

mod callback;
mod error;
mod error_context;
mod execution;
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

pub(super) const BUILTIN_NAME: &str = "cellfun";

#[runtime_builtin(
    name = "cellfun",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::cells::core::cellfun"
)]
async fn cellfun_builtin(
    context: Option<crate::context::RuntimeContext>,
    func: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    execution::execute(context.as_ref(), func, rest).await
}

#[cfg(test)]
mod tests;
