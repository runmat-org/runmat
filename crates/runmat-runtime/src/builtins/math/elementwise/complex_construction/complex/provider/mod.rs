mod operand;
mod output;

use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_provenance, handle_storage,
    set_handle_provenance, AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
};
use runmat_builtins::{COMPLEX_ERROR_INTERNAL, COMPLEX_ERROR_INVALID_INPUT};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::{error, host, BUILTIN_NAME};

pub(super) async fn unary(value: Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return host::unary(value);
    };
    let owner = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    validate_real_or_complex_handle(&handle)?;
    if handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return Ok(gpu_helpers::resident_gpu_value(handle));
    }
    if handle_integer_type(&handle).is_none()
        && !handle_is_logical(&handle)
        && handle_precision(&handle) == Some(owner.precision())
    {
        if let Some(output) = try_unary_provider(&handle, owner).await? {
            return Ok(output);
        }
    }
    unary_fallback(handle, owner).await
}

pub(super) async fn binary(real: Value, imaginary: Value) -> BuiltinResult<Value> {
    if !matches!(real, Value::GpuTensor(_)) && !matches!(imaginary, Value::GpuTensor(_)) {
        return host::binary(real, imaginary);
    }
    let source = gpu_helpers::select_resident_output_source(
        [&real, &imaginary]
            .into_iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
        BUILTIN_NAME,
    )?
    .ok_or_else(|| {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "resident construction had no output owner",
        )
    })?;
    let owner = gpu_helpers::exact_provider_for_handle(&source).ok_or_else(|| {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    validate_binary_resident_inputs(&real, &imaginary, owner)?;
    if let Some(output) = try_binary_provider(&real, &imaginary, &source, owner).await? {
        return Ok(output);
    }

    let real = gather(&real).await?;
    let imaginary = gather(&imaginary).await?;
    let output = host::binary(real, imaginary)?;
    gpu_helpers::restore_class_preserving_value(&source, output, BUILTIN_NAME)
}

async fn try_unary_provider(
    input: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Value>> {
    let metadata = gpu_helpers::snapshot_handle_metadata(input);
    let provenance = handle_provenance(input).unwrap_or(GpuHandleProvenance::Automatic);
    let result = owner.complex_from_real(input).await;
    gpu_helpers::restore_handle_metadata(input, &metadata);
    match result {
        Ok(mut output) if output::valid_unary(&output, input, owner) => {
            set_handle_provenance(&mut output, provenance);
            Ok(Some(gpu_helpers::resident_gpu_value(output)))
        }
        Ok(output) => {
            gpu_helpers::free_rejected_provider_output(&output, &[input], owner);
            Err(error(
                &COMPLEX_ERROR_INTERNAL,
                "provider complex_from_real returned malformed output",
            ))
        }
        Err(detail) if gpu_helpers::provider_hook_is_unsupported(&detail) => Ok(None),
        Err(detail) => Err(error(
            &COMPLEX_ERROR_INTERNAL,
            format!("provider complex_from_real failed: {detail}"),
        )),
    }
}

async fn try_binary_provider(
    real: &Value,
    imaginary: &Value,
    source: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Value>> {
    if operand::requires_exact_host_path(real) || operand::requires_exact_host_path(imaginary) {
        return Ok(None);
    }
    let expected_precision = operand::floating_result_precision(real, imaginary);
    if expected_precision != Some(owner.precision()) {
        return Ok(None);
    }
    let real = operand::RealGpuOperand::from_value(real, owner)?;
    let imaginary = operand::RealGpuOperand::from_value(imaginary, owner)?;
    let Some(shape) = output::compatible_shape(&real.handle.shape, &imaginary.handle.shape) else {
        return Ok(None);
    };
    let real_metadata = gpu_helpers::snapshot_handle_metadata(&real.handle);
    let imaginary_metadata = gpu_helpers::snapshot_handle_metadata(&imaginary.handle);
    let result = owner
        .complex_from_real_imag(&real.handle, &imaginary.handle)
        .await;
    gpu_helpers::restore_handle_metadata(&real.handle, &real_metadata);
    gpu_helpers::restore_handle_metadata(&imaginary.handle, &imaginary_metadata);
    match result {
        Ok(mut output)
            if output::valid_binary(
                &output,
                &real.handle,
                &imaginary.handle,
                shape,
                expected_precision,
                owner,
            ) =>
        {
            let provenance = handle_provenance(source).unwrap_or(GpuHandleProvenance::Automatic);
            set_handle_provenance(&mut output, provenance);
            Ok(Some(gpu_helpers::resident_gpu_value(output)))
        }
        Ok(output) => {
            gpu_helpers::free_rejected_provider_output(
                &output,
                &[&real.handle, &imaginary.handle],
                owner,
            );
            Err(error(
                &COMPLEX_ERROR_INTERNAL,
                "provider complex_from_real_imag returned malformed output",
            ))
        }
        Err(detail) if gpu_helpers::provider_hook_is_unsupported(&detail) => Ok(None),
        Err(detail) => Err(error(
            &COMPLEX_ERROR_INTERNAL,
            format!("provider complex_from_real_imag failed: {detail}"),
        )),
    }
}

async fn unary_fallback(
    handle: GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered = gpu_helpers::download_value_preserving_residency_async(owner, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &metadata);
    let output = host::unary(gathered.map_err(|detail| error(&COMPLEX_ERROR_INTERNAL, detail))?)?;
    gpu_helpers::restore_class_preserving_value(&handle, output, BUILTIN_NAME)
}

async fn gather(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value).await,
        value => Ok(value.clone()),
    }
}

fn validate_binary_resident_inputs(
    real: &Value,
    imaginary: &Value,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<()> {
    for value in [real, imaginary] {
        let Value::GpuTensor(handle) = value else {
            continue;
        };
        validate_real_handle(handle)?;
        let actual_owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
            error(
                &COMPLEX_ERROR_INTERNAL,
                "GPU provider unavailable for input",
            )
        })?;
        if !std::ptr::eq(actual_owner, owner) || handle.device_id != owner.device_id() {
            return Err(error(
                &COMPLEX_ERROR_INVALID_INPUT,
                "GPU inputs must belong to the same provider",
            ));
        }
    }
    Ok(())
}

fn validate_real_or_complex_handle(handle: &GpuTensorHandle) -> BuiltinResult<()> {
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
