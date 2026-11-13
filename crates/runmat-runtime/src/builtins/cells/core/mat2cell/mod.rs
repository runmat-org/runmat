mod admission;
mod assembly;
mod error;
mod extract;
mod implementation;
mod input;
mod partition;
mod spec;

#[cfg(test)]
mod tests;

use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(target_arch = "wasm32")]
pub(crate) use spec::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

#[runtime_builtin(
    name = "mat2cell",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::cells::core::mat2cell"
)]
async fn mat2cell_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::MAT2CELL_DESCRIPTOR,
        &runmat_builtins::MAT2CELL_ERROR_TOO_MANY_OUTPUTS,
        "mat2cell",
    )?;

    implementation::execute(runtime.as_ref(), value, rest).await
}
