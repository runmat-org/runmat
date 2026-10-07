mod complex;
mod evaluation;
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

use runmat_builtins::SQRT_INTEGER_INPUT_EXTENSION;
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
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SQRT_DESCRIPTOR,
        &runmat_builtins::SQRT_ERROR_TOO_MANY_OUTPUTS,
        "sqrt",
    )?;
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQRT_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
        crate::builtins::common::validation::reject_typed_complex_integer_for(
            &value,
            runmat_builtins::SQRT_CATALOG_ENTRY.identity,
            &runmat_builtins::SQRT_ERROR_INVALID_INPUT,
        )?;
    }
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
    crate::builtins::common::real_unary_preparation::execute(
        &runmat_builtins::SQRT_CATALOG_ENTRY,
        runmat_types::RealUnaryPredicate::Nonnegative,
        value,
        evaluation::evaluate_admitted,
    )
    .await
}

#[cfg(test)]
use complex::parts_f32 as sqrt_complex_parts_f32;
