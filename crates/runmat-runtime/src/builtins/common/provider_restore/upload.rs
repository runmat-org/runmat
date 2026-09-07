use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::{ComplexTensor, IntegerStorage, Tensor, Value};

use super::super::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult};

pub(crate) fn upload_value_like(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    upload_value_like_protected(provider, value, builtin, prototype, &[])
}

pub(crate) fn upload_value_like_protected(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
    prototype: &GpuTensorHandle,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let output = upload_value_protected(provider, value, builtin, protected)?;
    let Value::GpuTensor(mut handle) = output else {
        unreachable!("upload_value always returns a resident value")
    };
    if handle.device_id != prototype.device_id {
        free_unless_protected(provider, &handle, protected);
        return Err(build_runtime_error(format!(
            "{builtin}: provider restored the result on the wrong device"
        ))
        .with_builtin(builtin)
        .build());
    }
    if let Some(provenance) = runmat_accelerate_api::handle_provenance(prototype) {
        runmat_accelerate_api::set_handle_provenance(&mut handle, provenance);
    }
    Ok(Value::GpuTensor(handle))
}

pub(crate) fn upload_value_protected(
    provider: &dyn AccelProvider,
    value: Value,
    builtin: &str,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let (expected_shape, expected_storage, expected_integer_type, expected_precision) = match &value
    {
        Value::Num(_) => (
            vec![1, 1],
            runmat_accelerate_api::GpuTensorStorage::Real,
            None,
            Some(runmat_accelerate_api::ProviderPrecision::F64),
        ),
        Value::Int(value) => (
            vec![1, 1],
            runmat_accelerate_api::GpuTensorStorage::Real,
            Some(value.integer_class().into()),
            None,
        ),
        Value::Tensor(tensor) => (
            tensor.shape.clone(),
            runmat_accelerate_api::GpuTensorStorage::Real,
            tensor
                .integer_storage()
                .map(|storage| storage.integer_class().into()),
            if tensor.integer_storage().is_some() {
                None
            } else if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            } else {
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            },
        ),
        Value::Complex(_, _) => (
            vec![1, 1],
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            None,
            Some(runmat_accelerate_api::ProviderPrecision::F64),
        ),
        Value::ComplexTensor(tensor) => (
            tensor.shape.clone(),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            None,
            Some(
                if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
                    runmat_accelerate_api::ProviderPrecision::F32
                } else {
                    runmat_accelerate_api::ProviderPrecision::F64
                },
            ),
        ),
        other => {
            return Err(build_runtime_error(format!(
                "{builtin}: cannot restore unsupported result {other:?} to provider"
            ))
            .with_builtin(builtin)
            .build())
        }
    };
    let handle = match value {
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
            gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                build_runtime_error(format!(
                    "{builtin}: failed to restore result to input provider: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?
        }
        Value::Int(value) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_builtin(builtin)
                        .build()
                })?;
            gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                build_runtime_error(format!(
                    "{builtin}: failed to restore result to input provider: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?
        }
        Value::Tensor(tensor) => {
            gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
                build_runtime_error(format!(
                    "{builtin}: failed to restore result to input provider: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?;
            upload_complex_without_precision_override(provider, &tensor, builtin)?
        }
        Value::ComplexTensor(tensor) => {
            upload_complex_without_precision_override(provider, &tensor, builtin)?
        }
        other => unreachable!("validated restore value {other:?}"),
    };
    let valid = handle.shape == expected_shape
        && runmat_accelerate_api::handle_storage(&handle) == expected_storage
        && runmat_accelerate_api::handle_integer_type(&handle) == expected_integer_type
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && (expected_integer_type.is_some()
            || runmat_accelerate_api::handle_precision(&handle) == expected_precision)
        && gpu_helpers::exact_provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && protected
            .iter()
            .all(|input| !gpu_helpers::same_gpu_handle(&handle, input));
    if !valid {
        free_unless_protected(provider, &handle, protected);
        return Err(build_runtime_error(format!(
            "{builtin}: provider returned an incompatible restored result"
        ))
        .with_builtin(builtin)
        .build());
    }
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn free_unless_protected(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    protected: &[GpuTensorHandle],
) {
    if protected.iter().any(|candidate| {
        candidate.device_id == handle.device_id && candidate.buffer_id == handle.buffer_id
    }) {
        return;
    }
    let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(provider);
    let _ = owner.free(handle);
}

fn upload_complex_without_precision_override(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
    builtin: &str,
) -> BuiltinResult<GpuTensorHandle> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(format!(
            "{builtin}: typed complex integer GPU buffers are not supported"
        ))
        .with_builtin(builtin)
        .build());
    }
    let handle = gpu_helpers::upload_complex_tensor(provider, tensor).map_err(|error| {
        build_runtime_error(format!(
            "{builtin}: failed to restore result to input provider: {error}"
        ))
        .with_builtin(builtin)
        .build()
    })?;
    runmat_accelerate_api::set_handle_logical(&handle, false);
    Ok(handle)
}
