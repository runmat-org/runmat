use super::*;

pub(super) async fn evaluate(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        OPERATION.error(&SIGN_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(OPERATION.error(
            &SIGN_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let storage = runmat_accelerate_api::handle_storage(&handle);
    let integer = runmat_accelerate_api::handle_integer_type(&handle);
    if storage == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved && integer.is_some() {
        return Err(OPERATION.error(
            &SIGN_ERROR_INVALID_INPUT,
            "typed complex integer input is not supported",
        ));
    }
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    let precision = runmat_accelerate_api::handle_precision(&handle);
    let kernel_compatible =
        !logical && (integer.is_some() || precision == Some(provider.precision()));
    if kernel_compatible {
        let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
        let result = provider.unary_sign(&handle).await;
        match result {
            Ok(output) if valid_sign_gpu_output(&output, &handle, provider) => {
                let output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
                if storage == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved {
                    return gpu_helpers::complex_gpu_value(output);
                }
                return gpu_helpers::resident_gpu_value(output);
            }
            Ok(output) => {
                let mut error = OPERATION.error(
                    &SIGN_ERROR_INTERNAL,
                    "provider unary_sign returned malformed output",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => {
                let mut failure = OPERATION.error(
                    &SIGN_ERROR_INTERNAL,
                    format!("provider unary_sign failed: {error}"),
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
        Value::Complex(re, im) => {
            let (re_out, im_out) = host::sign_complex(re, im);
            Ok(Value::Complex(re_out, im_out))
        }
        Value::ComplexTensor(ct) => host::sign_complex_tensor(ct),
        other => host::sign_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
}

fn valid_sign_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::handle_storage(input),
            precision: runmat_accelerate_api::handle_precision(input),
            integer: runmat_accelerate_api::handle_integer_type(input),
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}
