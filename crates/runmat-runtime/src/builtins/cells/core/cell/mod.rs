//! Runtime-owned sink: size controls gather in their parser; prototypes retain
//! metadata-only handling. Successful construction clears original top-level
//! fusion marks without blanket gathering, freeing inputs, or inventing data roles.
mod allocation;
mod arguments;
mod error;
mod prototype;
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
    name = "cell",
    binding_variant = "default",
    builtin_path = "crate::builtins::cells::core::cell"
)]
async fn cell_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::CELL_DESCRIPTOR,
        &runmat_builtins::CELL_ERROR_TOO_MANY_OUTPUTS,
        "cell",
    )?;

    let resident_controls: Vec<_> = args
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let output = allocation::build(arguments::parse(args).await?)?;
    crate::builtins::common::gpu_helpers::clear_input_residency(&resident_controls, "cell")?;
    Ok(output)
}
