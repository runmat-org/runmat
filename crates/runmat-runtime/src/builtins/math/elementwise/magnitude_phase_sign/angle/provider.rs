use super::*;

pub(super) async fn evaluate(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        OPERATION.error(&ANGLE_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(OPERATION.error(
            &ANGLE_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        return Err(OPERATION.error(
            &ANGLE_ERROR_INVALID_INPUT,
            "integer gpuArray input is not supported",
        ));
    }
    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Err(OPERATION.error(
            &ANGLE_ERROR_INVALID_INPUT,
            "logical gpuArray input is not supported",
        ));
    }
    if runmat_accelerate_api::handle_precision(&handle) == Some(provider.precision()) {
        let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
        let result = provider.unary_angle(&handle).await;
        match result {
            Ok(output) if valid_angle_gpu_output(&output, &handle, provider) => {
                let output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
                return gpu_helpers::resident_gpu_value(output);
            }
            Ok(output) => {
                let mut error = OPERATION.error(
                    &ANGLE_ERROR_INTERNAL,
                    "provider unary_angle returned malformed output",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => {
                let mut failure = OPERATION.error(
                    &ANGLE_ERROR_INTERNAL,
                    format!("provider unary_angle failed: {error}"),
                );
                failure =
                    crate::runtime_error::with_provider_execution_source(failure, error.into());
                failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(failure);
            }
        }
    }
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    let gathered = gathered_result?;
    let host = match gathered {
        Value::Complex(re, im) => Ok(Value::Num(host::angle_scalar(re, im))),
        Value::ComplexTensor(ct) => host::angle_complex_tensor(ct),
        other => host::angle_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
}

fn valid_angle_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}
