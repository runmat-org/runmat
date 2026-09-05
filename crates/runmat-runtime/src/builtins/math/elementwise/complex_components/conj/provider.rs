use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_provenance, handle_storage,
    set_handle_provenance, AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, GpuOutputAliasPolicy, UnaryGpuOutputContract};
use crate::BuiltinResult;

use super::{builtin_error_with_detail, host, BUILTIN_NAME, CONJ_ERROR_INTERNAL};

pub(super) async fn execute(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let storage = handle_storage(&handle);
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(builtin_error_with_detail(
            &CONJ_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    if storage == GpuTensorStorage::Real {
        if handle_integer_type(&handle).is_some() {
            return Ok(gpu_helpers::resident_gpu_value(handle));
        }
        if handle_is_logical(&handle) {
            return Ok(gpu_helpers::logical_gpu_value(handle));
        }
    }

    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        builtin_error_with_detail(&CONJ_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let kernel_compatible = handle_integer_type(&handle).is_none()
        && !handle_is_logical(&handle)
        && handle_precision(&handle) == Some(provider.precision());
    if kernel_compatible {
        if let Some(output) = try_provider(&handle, provider).await? {
            return Ok(output);
        }
    }
    gather_conjugate_restore(handle, provider).await
}

async fn try_provider(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Value>> {
    let storage = handle_storage(input);
    let metadata = gpu_helpers::snapshot_handle_metadata(input);
    let provenance = handle_provenance(input).unwrap_or(GpuHandleProvenance::Automatic);
    let result = provider.unary_conj(input).await;
    gpu_helpers::restore_handle_metadata(input, &metadata);
    match result {
        Ok(mut output) if valid_output(&output, input, provider) => {
            set_handle_provenance(&mut output, provenance);
            let value = if storage == GpuTensorStorage::ComplexInterleaved {
                gpu_helpers::complex_gpu_value(output)
            } else {
                gpu_helpers::resident_gpu_value(output)
            };
            Ok(Some(value))
        }
        Ok(output) => {
            gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
            Err(builtin_error_with_detail(
                &CONJ_ERROR_INTERNAL,
                "provider unary_conj returned malformed output",
            ))
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(builtin_error_with_detail(
            &CONJ_ERROR_INTERNAL,
            format!("provider unary_conj failed: {error}"),
        )),
    }
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    let storage = handle_storage(input);
    let alias = if storage == GpuTensorStorage::Real {
        GpuOutputAliasPolicy::AllowInput
    } else {
        GpuOutputAliasPolicy::RequireDistinct
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        UnaryGpuOutputContract {
            storage,
            precision: handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    )
}

async fn gather_conjugate_restore(
    handle: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &metadata);
    let host_value = gathered
        .map_err(|error| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, error.to_string()))?;
    let conjugated = host::execute(host_value)?;
    gpu_helpers::restore_class_preserving_value(&handle, conjugated, BUILTIN_NAME)
        .map_err(|error| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, error.message()))
}
