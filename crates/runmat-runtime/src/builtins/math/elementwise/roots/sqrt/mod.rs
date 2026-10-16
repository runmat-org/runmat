mod admission;
mod complex;
mod errors;
mod evaluation;
mod fusion_sign;
mod host;
mod provider;
mod preparation;
mod publication;
mod specs;
#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;

pub(super) const BUILTIN_NAME: &str = "sqrt";

#[runtime_builtin(
    name = "sqrt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::roots::sqrt"
)]
async fn sqrt_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SQRT_DESCRIPTOR,
        &runmat_builtins::SQRT_ERROR_TOO_MANY_OUTPUTS,
        "sqrt",
    )?;
    if let Some(symbolic) = admission::admit(&value).await? {
        return Ok(symbolic);
    }
    preparation::execute(runtime.as_ref(), value).await
}

#[cfg(test)]
use complex::parts_f32 as sqrt_complex_parts_f32;
