use super::single_error_with_detail;
use super::storage::{int_value_to_f32, single_tensor_to_host};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;
use log::trace;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    SINGLE_ERROR_GPU_UNSUPPORTED, SINGLE_ERROR_INTERNAL, SINGLE_ERROR_INVALID_INPUT,
};
use runmat_value::{NumericStorage, Tensor, Value};

pub(super) async fn convert_to_gpu(
    value: Value,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(prototype).ok_or_else(|| {
        single_error_with_detail(
            &SINGLE_ERROR_GPU_UNSUPPORTED,
            "GPU output requested via 'like' but the prototype owner is unavailable",
        )
    })?;
    let value = match value {
        Value::GpuTensor(handle)
            if valid_single_resident_value(&handle, prototype, provider, &handle.shape) =>
        {
            return Ok(Value::GpuTensor(handle));
        }
        Value::GpuTensor(handle) => convert_to_host_like(Value::GpuTensor(handle)).await?,
        other => other,
    };
    match value {
        Value::Tensor(tensor) => upload_single_like(provider, prototype, &tensor),
        Value::Num(n) => {
            let tensor =
                Tensor::from_numeric_storage(NumericStorage::F32(vec![n as f32]), vec![1, 1])
                    .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
            upload_single_like(provider, prototype, &tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::from_numeric_storage(
                NumericStorage::F32(vec![int_value_to_f32(&i)]),
                vec![1, 1],
            )
            .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
            upload_single_like(provider, prototype, &tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::from_numeric_storage(
                NumericStorage::F32(vec![if b { 1.0 } else { 0.0 }]),
                vec![1, 1],
            )
            .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
            upload_single_like(provider, prototype, &tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = single_tensor_to_host(tensor::logical_to_tensor(&logical)?)?;
            upload_single_like(provider, prototype, &tensor)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(single_error_with_detail(
            &SINGLE_ERROR_INVALID_INPUT,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(single_error_with_detail(
            &SINGLE_ERROR_INVALID_INPUT,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

fn upload_single_like(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prototype: &GpuTensorHandle,
    tensor: &Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, tensor)
        .map_err(|error| single_error_with_detail(&SINGLE_ERROR_INTERNAL, error))?;
    if valid_single_like_output(&handle, prototype, provider, &tensor.shape) {
        Ok(Value::GpuTensor(handle))
    } else {
        free_rejected_single_handle(&handle, &[prototype]);
        Err(single_error_with_detail(
            &SINGLE_ERROR_INTERNAL,
            "provider returned malformed GPU output for 'like'",
        ))
    }
}

pub(super) fn valid_single_like_output(
    output: &GpuTensorHandle,
    prototype: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
) -> bool {
    !single_gpu_handles_alias(output, prototype)
        && valid_single_resident_value(output, prototype, provider, expected_shape)
}

fn valid_single_resident_value(
    output: &GpuTensorHandle,
    prototype: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
) -> bool {
    output.shape == expected_shape
        && output.device_id == prototype.device_id
        && runmat_accelerate_api::handle_precision(output)
            == Some(runmat_accelerate_api::ProviderPrecision::F32)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn single_gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

pub(super) fn free_rejected_single_handle(
    handle: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
) {
    if protected
        .iter()
        .any(|protected| single_gpu_handles_alias(handle, protected))
    {
        trace!("single: rejected handle aliases a caller-owned prototype; not freeing it");
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) {
        if let Err(error) = owner.free(handle) {
            trace!("single: failed to free rejected handle through its owner ({error})");
        }
    } else {
        trace!("single: rejected handle has no resolvable owner; leaving cleanup to its producer");
    }
}

pub(super) async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy)
                .await
                .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))
        }
        other => Ok(other),
    }
}
