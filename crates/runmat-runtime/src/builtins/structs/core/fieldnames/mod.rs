//! Field-name introspection for structures and supported RunMat objects.

mod error;
mod execution;
mod names;
mod output;
mod spec;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

pub(super) const BUILTIN_NAME: &str = "fieldnames";

#[runtime_builtin(
    name = "fieldnames",
    binding_variant = "default",
    builtin_path = "crate::builtins::structs::core::fieldnames"
)]
async fn fieldnames_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::FIELDNAMES_DESCRIPTOR,
        &runmat_builtins::FIELDNAMES_ERROR_TOO_MANY_OUTPUTS,
        "fieldnames",
    )?;

    execution::execute(value)
}

#[cfg(test)]
mod tests;
