use super::*;

pub(super) async fn evaluate(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
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
        let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
        let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let result = provider.unary_angle(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match result {
            Ok(mut output) if valid_angle_gpu_output(&output, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut output, input_provenance);
                return Ok(gpu_helpers::resident_gpu_value(output));
            }
            Ok(output) => {
                gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
                return Err(OPERATION.error(
                    &ANGLE_ERROR_INTERNAL,
                    "provider unary_angle returned malformed output",
                ));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                return Err(OPERATION.error(
                    &ANGLE_ERROR_INTERNAL,
                    format!("provider unary_angle failed: {error}"),
                ));
            }
        }
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered =
        gathered_result.map_err(|err| OPERATION.error(&ANGLE_ERROR_INTERNAL, err.to_string()))?;
    let host = match gathered {
        Value::Complex(re, im) => Ok(Value::Num(host::angle_scalar(re, im))),
        Value::ComplexTensor(ct) => host::angle_complex_tensor(ct),
        other => host::angle_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
        .map_err(|err| OPERATION.error(&ANGLE_ERROR_INTERNAL, err.message()))
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
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}
