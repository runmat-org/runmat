use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult};

use super::super::{
    resident_output_from_sources, valid_real_binary_output, ArithmeticProviderOperation,
};

pub(super) fn accept_output(
    operation: ArithmeticProviderOperation,
    output: GpuTensorHandle,
    class_source: &GpuTensorHandle,
    other_source: Option<&GpuTensorHandle>,
    provider: &dyn AccelProvider,
    expected_shape: &[usize],
) -> BuiltinResult<Option<Value>> {
    let sources =
        other_source.map_or_else(|| vec![class_source], |other| vec![class_source, other]);
    if valid_real_binary_output(
        &output,
        class_source,
        other_source,
        provider,
        expected_shape,
    ) {
        Ok(Some(resident_output_from_sources(output, sources)))
    } else {
        gpu_helpers::free_rejected_provider_output(&output, &sources, provider);
        Err(provider_contract_error(
            operation,
            "provider returned invalid element-wise output metadata",
        ))
    }
}

pub(super) fn accepted_output(result: &BuiltinResult<Option<Value>>) -> Option<&GpuTensorHandle> {
    result
        .as_ref()
        .ok()
        .and_then(Option::as_ref)
        .and_then(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
}

pub(super) fn common_owner(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let left_owner = gpu_helpers::exact_provider_for_handle(left)?;
    let right_owner = gpu_helpers::exact_provider_for_handle(right)?;
    (std::ptr::eq(left_owner, right_owner) && left.device_id == right.device_id)
        .then_some(left_owner)
}

pub(super) fn compatible_inputs(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    if runmat_accelerate_api::handle_storage(left) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_storage(right) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(left)
        || runmat_accelerate_api::handle_is_logical(right)
    {
        return false;
    }
    match (
        runmat_accelerate_api::handle_integer_type(left),
        runmat_accelerate_api::handle_integer_type(right),
    ) {
        (Some(left), Some(right)) => left == right,
        (None, None) => {
            runmat_accelerate_api::handle_precision(left)
                == runmat_accelerate_api::handle_precision(right)
                && runmat_accelerate_api::handle_precision(left).is_some()
        }
        _ => false,
    }
}

pub(super) fn float_handle_supported(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle).is_some()
}

pub(super) fn provider_error(
    operation: ArithmeticProviderOperation,
    error: impl std::fmt::Display,
) -> crate::RuntimeError {
    build_runtime_error(format!("{}: {error}", operation.name()))
        .with_builtin(operation.name())
        .build()
}

pub(super) fn provider_contract_error(
    operation: ArithmeticProviderOperation,
    message: &'static str,
) -> crate::RuntimeError {
    build_runtime_error(format!("{}: {message}", operation.name()))
        .with_builtin(operation.name())
        .build()
}
