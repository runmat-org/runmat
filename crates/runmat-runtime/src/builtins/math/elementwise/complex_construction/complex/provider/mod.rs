mod operand;
mod output;
mod validation;

use validation::{validate_binary_resident_inputs, validate_real_or_complex_handle};

use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, AccelProvider,
    GpuTensorHandle, GpuTensorStorage,
};
use runmat_builtins::COMPLEX_ERROR_INTERNAL;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::{error, host, BUILTIN_NAME};

pub(super) async fn unary(value: Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return host::unary(value);
    };
    let owner = gpu_helpers::require_resident_compute_owner(&handle, || {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    validate_real_or_complex_handle(&handle)?;
    if handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return gpu_helpers::resident_gpu_value(handle);
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
    gpu_helpers::require_resident_compute_values(&[&real, &imaginary], true, || {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
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
    let owner = gpu_helpers::require_resident_compute_owner(&source, || {
        error(
            &COMPLEX_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    validate_binary_resident_inputs(&real, &imaginary, owner)?;
    if let Some(output) = try_binary_provider(&real, &imaginary, &source, owner).await? {
        return Ok(output);
    }

    let protected = resident_inputs(&real, &imaginary);
    let real_host = gather(&real).await?;
    let imaginary_host = gather(&imaginary).await?;
    let output = host::binary(real_host, imaginary_host)?;
    gpu_helpers::restore_class_preserving_value_protecting(
        &source,
        output,
        BUILTIN_NAME,
        &protected,
    )
}

async fn try_unary_provider(
    input: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Value>> {
    let metadata = gpu_helpers::ResidencyRestoreGuard::new(input);
    let result = owner.complex_from_real(input).await;
    drop(metadata);
    match result {
        Ok(output) if output::valid_unary(&output, input, owner) => {
            let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
            gpu_helpers::complex_gpu_value(output).map(Some)
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            error(
                &COMPLEX_ERROR_INTERNAL,
                "provider complex_from_real returned malformed output",
            ),
        )),
        Err(detail) if gpu_helpers::provider_hook_allows_host_fallback(&detail) => Ok(None),
        Err(detail) => Err(provider_error("complex_from_real", detail)),
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
    let admitted_real = operand::RealOperand::admit(real)?;
    let admitted_imaginary = operand::RealOperand::admit(imaginary)?;
    let shape = host::compatible_shapes(admitted_real.shape(), admitted_imaginary.shape())?;
    let mut protected = resident_inputs(real, imaginary);
    let real = operand::RealGpuOperand::from_admitted(admitted_real, owner, &protected)?;
    protected.push(&real.handle);
    let imaginary = operand::RealGpuOperand::from_admitted(admitted_imaginary, owner, &protected)?;
    let real_metadata = gpu_helpers::ResidencyRestoreGuard::new(&real.handle);
    let imaginary_metadata = gpu_helpers::ResidencyRestoreGuard::new(&imaginary.handle);
    let result = owner
        .complex_from_real_imag(&real.handle, &imaginary.handle)
        .await;
    drop(real_metadata);
    drop(imaginary_metadata);
    match result {
        Ok(output)
            if output::valid_binary(
                &output,
                &real.handle,
                &imaginary.handle,
                shape,
                expected_precision,
                owner,
            ) =>
        {
            let output = gpu_helpers::propagate_owned_output_provenance(output, [source])?;
            gpu_helpers::complex_gpu_value(output).map(Some)
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            error(
                &COMPLEX_ERROR_INTERNAL,
                "provider complex_from_real_imag returned malformed output",
            ),
        )),
        Err(detail) if gpu_helpers::provider_hook_allows_host_fallback(&detail) => Ok(None),
        Err(detail) => Err(provider_error("complex_from_real_imag", detail)),
    }
}

fn provider_error(operation: &str, source: anyhow::Error) -> crate::RuntimeError {
    let mut failure = error(
        &COMPLEX_ERROR_INTERNAL,
        format!("provider {operation} failed: {source}"),
    );
    failure = crate::runtime_error::with_provider_execution_source(failure, source.into());
    failure
}

fn resident_inputs<'a>(real: &'a Value, imaginary: &'a Value) -> Vec<&'a GpuTensorHandle> {
    [real, imaginary]
        .into_iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect()
}

async fn unary_fallback(
    handle: GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let metadata = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let gathered = gpu_helpers::download_value_preserving_residency_async(owner, &handle).await;
    drop(metadata);
    let output = host::unary(gathered.map_err(|source| {
        let mut failure = error(&COMPLEX_ERROR_INTERNAL, source.to_string());
        failure.source = Some(Box::new(source));
        failure
    })?)?;
    gpu_helpers::restore_class_preserving_value(&handle, output, BUILTIN_NAME)
}

async fn gather(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value).await,
        value => Ok(value.clone()),
    }
}
