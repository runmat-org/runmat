mod errors;
mod evaluation;
mod host;
mod provider;
mod preparation;
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

pub(super) const BUILTIN_NAME: &str = "realsqrt";

#[runtime_builtin(
    name = "realsqrt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::elementwise::roots::realsqrt"
)]
async fn realsqrt_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<RootedValue> {
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::REALSQRT_DESCRIPTOR,
        &runmat_builtins::REALSQRT_ERROR_TOO_MANY_OUTPUTS,
        "realsqrt",
    )?;
    preparation::execute(runtime.as_ref(), value).await
}
