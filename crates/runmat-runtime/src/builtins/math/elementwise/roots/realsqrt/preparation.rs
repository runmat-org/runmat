//! Exact real-unary preparation and publication for this root operation.

use super::*;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    value: Value,
) -> BuiltinResult<RootedValue> {
    crate::builtins::common::real_unary_preparation::execute(
        runtime,
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
                super::super::super::unary_provider::restore_output(
                    provider,
                    input,
                    output,
                    BUILTIN_NAME,
                )
            },
            publish_real: |provider: &'static dyn AccelProvider,
                           input: &GpuTensorHandle,
                           output| {
                super::super::super::unary_provider::validate_real_unary_output(
                    provider,
                    input,
                    output,
                    BUILTIN_NAME,
                )
            },
        },
        value,
        false,
        |value: f32| host::admitted_scalar_f32(value).map(|real| (real, 0.0)),
        |value: f64| host::admitted_scalar_f64(value).map(|real| (real, 0.0)),
        |value| async move {
            let output = evaluation::evaluate_admitted(value).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
