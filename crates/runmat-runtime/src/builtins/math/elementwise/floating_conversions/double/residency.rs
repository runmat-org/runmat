use super::double_error_with_detail;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;
use log::trace;
use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    DOUBLE_ERROR_GPU_UNSUPPORTED, DOUBLE_ERROR_INTERNAL, DOUBLE_ERROR_INVALID_INPUT,
};
use runmat_value::{ComplexTensor, Tensor, Value};

pub(super) async fn convert_to_gpu(
    value: Value,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = resolved_actual_double_owner(prototype).ok_or_else(|| {
        double_error_with_detail(
            &DOUBLE_ERROR_GPU_UNSUPPORTED,
            "GPU output requested via 'like' but the prototype owner is unavailable",
        )
    })?;
    if provider.precision() != ProviderPrecision::F64 {
        return Err(double_error_with_detail(
            &DOUBLE_ERROR_GPU_UNSUPPORTED,
            "active acceleration provider does not support float64 storage",
        ));
    }
    let value = match value {
        Value::GpuTensor(handle) => {
            let same_owner = resolved_actual_double_owner(&handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider));
            if same_owner
                && handle.device_id == prototype.device_id
                && runmat_accelerate_api::handle_precision(&handle) == Some(ProviderPrecision::F64)
                && runmat_accelerate_api::handle_integer_type(&handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(&handle)
            {
                return Ok(Value::GpuTensor(handle));
            }
            convert_to_host_like(Value::GpuTensor(handle)).await?
        }
        other => other,
    };
    match value {
        Value::Tensor(tensor) => upload_double_like_real(provider, prototype, &tensor),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            upload_double_like_real(provider, prototype, &tensor)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))?;
            upload_double_like_complex(provider, prototype, &tensor)
        }
        Value::ComplexTensor(tensor) => upload_double_like_complex(provider, prototype, &tensor),
        other => Err(double_error_with_detail(
            &DOUBLE_ERROR_INVALID_INPUT,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

fn upload_double_like_real(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    tensor: &Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, tensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?;
    if valid_double_like_output(&handle, prototype, provider, &tensor.shape, false) {
        Ok(Value::GpuTensor(handle))
    } else {
        free_rejected_double_handle(&handle, &[prototype]);
        Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            "provider returned malformed GPU output for 'like'",
        ))
    }
}

fn upload_double_like_complex(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    tensor: &ComplexTensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_complex_tensor(provider, tensor)
        .map_err(|error| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error))?;
    if valid_double_like_output(&handle, prototype, provider, &tensor.shape, true) {
        Ok(gpu_helpers::complex_gpu_value(handle))
    } else {
        free_rejected_double_handle(&handle, &[prototype]);
        Err(double_error_with_detail(
            &DOUBLE_ERROR_INTERNAL,
            "provider returned malformed complex GPU output for 'like'",
        ))
    }
}

pub(super) fn valid_double_like_output(
    output: &GpuTensorHandle,
    prototype: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
    complex: bool,
) -> bool {
    let expected_storage = if complex {
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    } else {
        runmat_accelerate_api::GpuTensorStorage::Real
    };
    output.shape == expected_shape
        && output.device_id == prototype.device_id
        && !gpu_handles_alias(output, prototype)
        && runmat_accelerate_api::handle_precision(output) == Some(ProviderPrecision::F64)
        && runmat_accelerate_api::handle_storage(output) == expected_storage
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && resolved_actual_double_owner(output).is_some_and(|owner| std::ptr::eq(owner, provider))
}

pub(super) fn resolved_actual_double_owner(
    handle: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id)
}

pub(super) fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

pub(super) fn free_rejected_double_handle(
    handle: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
) {
    if protected
        .iter()
        .any(|protected| gpu_handles_alias(handle, protected))
    {
        trace!("double: rejected handle aliases a caller-owned input; not freeing it");
        return;
    }
    if let Some(owner) = resolved_actual_double_owner(handle) {
        if let Err(error) = owner.free(handle) {
            trace!("double: failed to free rejected handle through its owner ({error})");
        }
    } else {
        trace!("double: rejected handle has no resolvable owner; leaving cleanup to its producer");
    }
}

pub(super) async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy)
                .await
                .map_err(|e| double_error_with_detail(&DOUBLE_ERROR_INTERNAL, e))
        }
        other => Ok(other),
    }
}
