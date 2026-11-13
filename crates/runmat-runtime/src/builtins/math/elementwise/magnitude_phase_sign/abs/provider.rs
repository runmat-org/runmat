use super::*;

pub(super) async fn evaluate(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        OPERATION.error(&ABS_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(OPERATION.error(
            &ABS_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }

    let integer = runmat_accelerate_api::handle_integer_type(&handle);
    let logical = runmat_accelerate_api::handle_is_logical(&handle);
    let precision = runmat_accelerate_api::handle_precision(&handle);
    if integer.is_none() && !logical && precision == Some(provider.precision()) {
        let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
        let result = provider.unary_abs(&handle).await;
        match result {
            Ok(output) if valid_output(&output, &handle, provider) => {
                let output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
                return gpu_helpers::resident_gpu_value(output);
            }
            Ok(output) => {
                let mut error = OPERATION.error(
                    &ABS_ERROR_INTERNAL,
                    "provider unary_abs returned malformed output",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => {
                let mut failure = OPERATION.error(
                    &ABS_ERROR_INTERNAL,
                    format!("provider unary_abs failed: {error}"),
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
    let host = host::abs_host_value(gathered)?;
    let host = crate::builtins::math::resident_real_unary::preserve_elementwise_scalar_shape(
        host,
        handle.shape(),
        BUILTIN_NAME,
    )?;
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, host)
        .map_err(|error| OPERATION.error(&ABS_ERROR_INTERNAL, error))?;
    let output = gpu_helpers::upload_tensor_protecting(provider, &tensor, &[&handle])?;
    let expected_precision = if integer.is_some() {
        None
    } else if logical {
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    } else {
        precision
    };
    if !valid_real_output(&output, &handle, provider, expected_precision, integer) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            OPERATION.error(
                &ABS_ERROR_INTERNAL,
                "provider upload returned malformed fallback output",
            ),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [&handle])?;
    gpu_helpers::resident_gpu_value(output)
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    valid_real_output(
        output,
        input,
        provider,
        runmat_accelerate_api::handle_precision(input),
        None,
    )
}

fn valid_real_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    precision: Option<runmat_accelerate_api::ProviderPrecision>,
    integer: Option<runmat_accelerate_api::IntegerElementType>,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision,
            integer,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}
