mod complex;
mod errors;
mod host;
mod provider;
mod specs;
#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

use runmat_builtins::{SQRT_ERROR_INVALID_INPUT, SQRT_INTEGER_INPUT_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_value::{SymbolicFunction, Value};

use crate::builtins::math::symbolic::symbolic_function;
use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "sqrt";

#[runtime_builtin(
    name = "sqrt",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::roots::sqrt"
)]
async fn sqrt_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &value,
        &SQRT_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "input",
    )
    .await?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Sqrt) {
        return Ok(symbolic);
    }
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Complex(real, imag) => Ok(complex::evaluate_scalar(real, imag)),
        Value::ComplexTensor(tensor) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(
                &tensor,
                BUILTIN_NAME,
            )?;
            complex::evaluate_tensor(tensor)
        }
        Value::CharArray(chars) => host::evaluate_characters(chars),
        Value::String(_) | Value::StringArray(_) => Err(errors::with_detail(
            &SQRT_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => host::evaluate(other),
    }
}

#[cfg(test)]
use complex::parts_f32 as sqrt_complex_parts_f32;
