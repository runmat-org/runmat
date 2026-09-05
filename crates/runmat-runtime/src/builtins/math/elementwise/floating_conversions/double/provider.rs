use super::double_error_with_detail;
use super::host::double_from_gathered;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderPrecision,
};
use runmat_builtins::DOUBLE_ERROR_INTERNAL;
use runmat_value::Value;

pub(super) async fn double_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        double_error_with_detail(&DOUBLE_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let input_element =
        gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
            double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                "GPU input class metadata contradicts its physical storage",
            )
        })?;

    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let direct_eligible = provider.precision() == ProviderPrecision::F64
        && input_element == NumericElementType::F64
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::Real;
    if direct_eligible {
        let direct = provider.unary_double(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match direct {
            Ok(mut output) if valid_output(&output, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut output, input_provenance);
                return Ok(gpu_helpers::resident_gpu_value(output));
            }
            Ok(output) => {
                gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
                return Err(double_error_with_detail(
                    &DOUBLE_ERROR_INTERNAL,
                    "provider unary_double returned malformed output",
                ));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                return Err(double_error_with_detail(
                    &DOUBLE_ERROR_INTERNAL,
                    format!("provider unary_double failed: {error}"),
                ));
            }
        }
    }

    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered_result
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error.message()))?;
    let converted = double_from_gathered(gathered)?;
    if provider.precision() != ProviderPrecision::F64 {
        return Ok(converted);
    }

    let mut output = match &converted {
        Value::Tensor(tensor) => gpu_helpers::upload_tensor(provider, tensor)
            .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?,
        Value::ComplexTensor(tensor) => gpu_helpers::upload_complex_tensor(provider, tensor)?,
        other => {
            return Err(double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                format!("double fallback produced unsupported value {other:?}"),
            ));
        }
    };
    if !valid_output(&output, &handle, provider) {
        gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
        return Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            "provider upload returned malformed double fallback output",
        ));
    }
    runmat_accelerate_api::set_handle_provenance(&mut output, input_provenance);
    if runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::ComplexInterleaved {
        Ok(gpu_helpers::complex_gpu_value(output))
    } else {
        Ok(gpu_helpers::resident_gpu_value(output))
    }
}

pub(super) fn valid_double_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    complex: bool,
) -> bool {
    let expected_storage = if complex {
        GpuTensorStorage::ComplexInterleaved
    } else {
        GpuTensorStorage::Real
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: expected_storage,
            precision: Some(ProviderPrecision::F64),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    valid_double_gpu_output(
        output,
        input,
        provider,
        runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::ComplexInterleaved,
    )
}
