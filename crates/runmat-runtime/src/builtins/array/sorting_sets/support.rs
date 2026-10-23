use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, IntegerElementType};
use runmat_value::{ComplexTensor, IntegerStorage, Tensor, Value};

use crate::builtins::common::gpu_helpers;

pub(super) fn is_unsupported_set_gpu_integer(handle: &GpuTensorHandle) -> bool {
    matches!(
        runmat_accelerate_api::handle_integer_type(handle),
        Some(IntegerElementType::I64 | IntegerElementType::U64)
    )
}

pub(super) fn set_output_provider(a: &Value, b: &Value) -> Option<&'static dyn AccelProvider> {
    output_provider(a).or_else(|| output_provider(b))
}

pub(super) fn output_provider(value: &Value) -> Option<&'static dyn AccelProvider> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    runmat_accelerate_api::provider_for_handle(handle).or_else(runmat_accelerate_api::provider)
}

pub(super) fn restore_set_outputs(
    provider: Option<&'static dyn AccelProvider>,
    builtin: &str,
    outputs: Vec<Value>,
    internal_error: fn(String) -> crate::RuntimeError,
) -> crate::BuiltinResult<Vec<Value>> {
    let Some(provider) = provider else {
        return Ok(outputs);
    };
    outputs
        .into_iter()
        .map(|value| upload_set_output(provider, builtin, value, internal_error))
        .collect()
}

fn upload_set_output(
    provider: &'static dyn AccelProvider,
    builtin: &str,
    value: Value,
    internal_error: fn(String) -> crate::RuntimeError,
) -> crate::BuiltinResult<Value> {
    let upload_tensor = |tensor: Tensor| -> crate::BuiltinResult<Value> {
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|error| internal_error(format!("{builtin}: GPU upload failed: {error}")))?;
        gpu_helpers::resident_gpu_value(handle)
    };
    match value {
        Value::Tensor(tensor) => upload_tensor(tensor),
        Value::Num(number) => upload_tensor(
            Tensor::new(vec![number], vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?,
        ),
        Value::Int(integer) => upload_tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(integer), vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?,
        ),
        Value::Bool(logical) => {
            upload_logical_output(provider, &[u8::from(logical)], &[1, 1], builtin)
        }
        Value::LogicalArray(logical) => {
            upload_logical_output(provider, &logical.data, &logical.shape, builtin)
        }
        Value::Complex(real, imaginary) => {
            let tensor = ComplexTensor::new(vec![(real, imaginary)], vec![1, 1])
                .map_err(|error| internal_error(format!("{builtin}: {error}")))?;
            let handle =
                gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|error| {
                    internal_error(format!("{builtin}: GPU upload failed: {}", error.message()))
                })?;
            gpu_helpers::complex_gpu_value(handle)
        }
        Value::ComplexTensor(tensor) => {
            let handle =
                gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|error| {
                    internal_error(format!("{builtin}: GPU upload failed: {}", error.message()))
                })?;
            gpu_helpers::complex_gpu_value(handle)
        }
        other => Err(internal_error(format!(
            "{builtin}: cannot restore resident output {other:?}"
        ))),
    }
}

/// Upload canonical host logical bytes through their producer-owned API.
pub(super) fn upload_logical_output(
    provider: &dyn AccelProvider,
    data: &[u8],
    shape: &[usize],
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let output = provider
        .upload_logical(&runmat_accelerate_api::HostLogicalTensorView { data, shape })
        .map_err(|cause| {
            crate::build_runtime_error(format!("{builtin}: logical upload failed: {cause}"))
                .with_builtin(builtin)
                .with_source(cause)
                .build()
        })?;
    gpu_helpers::logical_gpu_value(output)
}

#[cfg(test)]
mod owning_upload_tests {
    use super::*;

    #[test]
    fn set_outputs_are_genuinely_logical_before_adoption() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let output =
                upload_logical_output(provider, &[1, 0, 1, 0], &[2, 2], "ismember").unwrap();
            let Value::GpuTensor(handle) = &output else {
                panic!("resident logical output")
            };
            assert!(handle.view().is_logical());
            assert_eq!(handle.shape(), &[2, 2]);
            let host =
                futures::executor::block_on(gpu_helpers::gather_value_async(&output)).unwrap();
            let Value::LogicalArray(array) = host else {
                panic!("logical host class")
            };
            assert_eq!(array.data.as_slice(), &[1, 0, 1, 0]);
            assert_eq!(array.shape, vec![2, 2]);
        });
    }
}
