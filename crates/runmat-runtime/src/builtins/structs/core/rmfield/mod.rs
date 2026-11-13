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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::structs::core::rmfield"
)]
async fn rmfield_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    target: Value,
    fields: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::RMFIELD_DESCRIPTOR,
        &runmat_builtins::RMFIELD_ERROR_TOO_MANY_OUTPUTS,
        "rmfield",
    )?;

    execution::execute(runtime.as_ref(), target, fields)
}

#[cfg(test)]
mod tests;
