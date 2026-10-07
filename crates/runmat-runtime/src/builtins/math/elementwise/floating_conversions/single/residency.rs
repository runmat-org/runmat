use super::single_error_with_detail;
use super::storage::{int_value_to_f32, single_tensor_to_host};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    SINGLE_ERROR_GPU_UNSUPPORTED, SINGLE_ERROR_INTERNAL, SINGLE_ERROR_INVALID_INPUT,
};
use runmat_value::{NumericStorage, Tensor, Value};

pub(super) async fn convert_to_gpu(
    value: Value,
    prototype: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        single_error_with_detail(
            &SINGLE_ERROR_GPU_UNSUPPORTED,
            "GPU output requested via 'like' but the prototype owner is unavailable",
        )
    })?;
    let value = match value {
        Value::GpuTensor(handle)
            if valid_single_resident_value(&handle, prototype, provider, handle.shape()) =>
        {
            return Ok(Value::GpuTensor(handle));
        }
        Value::GpuTensor(handle) => convert_to_host_like(Value::GpuTensor(handle)).await?,
        other => other,
    };
    match value {
        Value::Tensor(tensor) => upload_single_like(provider, prototype, &tensor, protected),
        Value::Num(n) => {
            let tensor =
                Tensor::from_numeric_storage(NumericStorage::F32(vec![n as f32]), vec![1, 1])
                    .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
            upload_single_like(provider, prototype, &tensor, protected)
        }
        Value::Int(i) => {
            let tensor = Tensor::from_numeric_storage(
                NumericStorage::F32(vec![int_value_to_f32(&i)]),
                vec![1, 1],
            )
            .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
            upload_single_like(provider, prototype, &tensor, protected)
        }
        Value::Bool(b) => {
            let tensor = Tensor::from_numeric_storage(
                NumericStorage::F32(vec![if b { 1.0 } else { 0.0 }]),
                vec![1, 1],
            )
            .map_err(|e| single_error_with_detail(&SINGLE_ERROR_INTERNAL, e))?;
            upload_single_like(provider, prototype, &tensor, protected)
        }
        Value::LogicalArray(logical) => {
            let tensor = single_tensor_to_host(tensor::logical_to_tensor(&logical)?)?;
            upload_single_like(provider, prototype, &tensor, protected)
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
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let handle =
        gpu_helpers::upload_tensor_protecting(provider, tensor, protected).map_err(|source| {
            let mut error = single_error_with_detail(&SINGLE_ERROR_INTERNAL, &source);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error.source = Some(Box::new(source));
            error
        })?;
    if valid_single_like_output(&handle, prototype, provider, tensor.shape()) {
        Ok(Value::GpuTensor(handle))
    } else {
        let mut error = single_error_with_detail(
            &SINGLE_ERROR_INTERNAL,
            "provider returned malformed GPU output for 'like'",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(gpu_helpers::reject_gpu_output(handle, error))
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
    output.shape() == expected_shape
        && output.device_id() == prototype.device_id()
        && runmat_accelerate_api::handle_precision(output)
            == Some(runmat_accelerate_api::ProviderPrecision::F32)
        && gpu_helpers::numeric_descriptor_matches(
            output,
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn single_gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id() == rhs.device_id() && lhs.buffer_id() == rhs.buffer_id()
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
