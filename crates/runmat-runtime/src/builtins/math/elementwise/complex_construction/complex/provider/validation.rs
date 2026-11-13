//! Resident input metadata and exact-owner admission before provider effects.

use runmat_accelerate_api::{handle_storage, AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{COMPLEX_ERROR_INTERNAL, COMPLEX_ERROR_INVALID_INPUT};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::error;

pub(super) fn validate_binary_resident_inputs(
    real: &Value,
    imaginary: &Value,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<()> {
    for value in [real, imaginary] {
        let Value::GpuTensor(handle) = value else {
            continue;
        };
        validate_real_handle(handle)?;
        let actual_owner = gpu_helpers::require_resident_compute_owner(handle, || {
            error(
                &COMPLEX_ERROR_INTERNAL,
                "GPU provider unavailable for input",
            )
        })?;
        if !runmat_accelerate_api::same_provider_instance(actual_owner, owner)
            || handle.device_id() != owner.device_id()
        {
            return Err(error(
                &COMPLEX_ERROR_INVALID_INPUT,
                "GPU inputs must belong to the same provider",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_real_or_complex_handle(handle: &GpuTensorHandle) -> BuiltinResult<()> {
    gpu_helpers::expected_handle_numeric_element_type(handle)
        .map(|_| ())
        .map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))
}

fn validate_real_handle(handle: &GpuTensorHandle) -> BuiltinResult<()> {
    validate_real_or_complex_handle(handle)?;
    if handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(error(&COMPLEX_ERROR_INVALID_INPUT, "inputs must be real"));
    }
    Ok(())
}
