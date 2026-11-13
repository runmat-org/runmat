//! Exact real-unary preparation and publication for this root operation.

use super::*;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    value: Value,
) -> BuiltinResult<RootedValue> {
    crate::builtins::common::real_unary_preparation::execute(
        runtime,
        &runmat_builtins::SQRT_CATALOG_ENTRY,
        runmat_types::ScalarUnaryMath::Sqrt,
        crate::context::preparation::RealUnaryPublication::Restore,
        crate::builtins::common::real_unary_preparation::RealUnaryAdapter {
            internal_error: &runmat_builtins::SQRT_ERROR_INTERNAL,
            before_host: |_: &GpuTensorHandle, _| Ok(()),
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
        true,
        |value: f32| Ok(host::real_parts_f32(value)),
        |value: f64| Ok(host::real_parts_f64(value)),
        |value| async move {
            let output = evaluation::evaluate_admitted(value).await?;
            RootedValue::new(output).map_err(|cause| cause.into_runtime_error())
        },
    )
    .await
}
