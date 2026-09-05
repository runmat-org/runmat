use super::*;

pub(super) async fn evaluate(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
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
        let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
        let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let result = provider.unary_sign(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match result {
            Ok(mut output) if valid_sign_gpu_output(&output, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut output, input_provenance);
                if storage == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved {
                    return Ok(gpu_helpers::complex_gpu_value(output));
                }
                return Ok(gpu_helpers::resident_gpu_value(output));
            }
            Ok(output) => {
                gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
                return Err(OPERATION.error(
                    &SIGN_ERROR_INTERNAL,
                    "provider unary_sign returned malformed output",
                ));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                return Err(OPERATION.error(
                    &SIGN_ERROR_INTERNAL,
                    format!("provider unary_sign failed: {error}"),
                ));
            }
        }
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered_result.map_err(|err| OPERATION.error(&SIGN_ERROR_INTERNAL, err))?;
    let host = match gathered {
        Value::Complex(re, im) => {
            let (re_out, im_out) = host::sign_complex(re, im);
            Ok(Value::Complex(re_out, im_out))
        }
        Value::ComplexTensor(ct) => host::sign_complex_tensor(ct),
        other => host::sign_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
        .map_err(|err| OPERATION.error(&SIGN_ERROR_INTERNAL, err.message()))
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
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}
