mod errors;
mod evaluation;
mod host;
mod provider;
mod specs;
#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
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
    crate::builtins::common::real_unary_preparation::execute(
        runtime.as_ref(),
        &runmat_builtins::REALSQRT_CATALOG_ENTRY,
        runmat_types::ScalarUnaryMath::Sqrt,
        crate::context::preparation::RealUnaryPublication::Restore,
        crate::builtins::common::real_unary_preparation::RealUnaryAdapter {
            internal_error: &runmat_builtins::REALSQRT_ERROR_INTERNAL,
            before_host: |_: &GpuTensorHandle, classification| {
                if classification == runmat_types::RealUnaryClassification::Outside {
                    return Err(errors::with_detail(
                        &runmat_builtins::REALSQRT_ERROR_DOMAIN,
                        "gpuArray contains negative values",
                    ));
                }
                Ok(())
            },
            publish_host: |provider: &'static dyn AccelProvider,
                           input: &GpuTensorHandle,
                           output| {
                super::super::unary_provider::restore_output(provider, input, output, BUILTIN_NAME)
            },
            publish_real: |provider: &'static dyn AccelProvider,
                           input: &GpuTensorHandle,
                           output| {
                super::super::unary_provider::validate_real_unary_output(
                    provider,
                    input,
                    output,
                    BUILTIN_NAME,
                )
            },
        },
        value,
        |value| async move {
            let output = evaluation::evaluate_admitted(value).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
