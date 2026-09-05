use super::*;

pub(super) async fn evaluate(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
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
        let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
        let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let result = provider.unary_abs(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match result {
            Ok(mut output) if valid_output(&output, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut output, input_provenance);
                return Ok(gpu_helpers::resident_gpu_value(output));
            }
            Ok(output) => {
                gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
                return Err(OPERATION.error(
                    &ABS_ERROR_INTERNAL,
                    "provider unary_abs returned malformed output",
                ));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                return Err(OPERATION.error(
                    &ABS_ERROR_INTERNAL,
                    format!("provider unary_abs failed: {error}"),
                ));
            }
        }
    }

    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered =
        gathered_result.map_err(|error| OPERATION.error(&ABS_ERROR_INTERNAL, error.message()))?;
    let host = host::abs_host_value(gathered)?;
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, host)
        .map_err(|error| OPERATION.error(&ABS_ERROR_INTERNAL, error))?;
    let mut output = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|error| OPERATION.error(&ABS_ERROR_INTERNAL, error))?;
    let expected_precision = if integer.is_some() {
        None
    } else if logical {
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    } else {
        precision
    };
    if !valid_real_output(&output, &handle, provider, expected_precision, integer) {
        gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
        return Err(OPERATION.error(
            &ABS_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    runmat_accelerate_api::set_handle_provenance(&mut output, input_provenance);
    Ok(gpu_helpers::resident_gpu_value(output))
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
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}
