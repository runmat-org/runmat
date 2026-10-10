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
        resident_output_from_sources(output, sources).map(Some)
    } else {
        Err(gpu_helpers::reject_gpu_output(
            output,
            provider_contract_error(
                operation,
                "provider returned invalid element-wise output metadata",
            ),
        ))
    }
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn common_owner(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let left_owner = gpu_helpers::exact_provider_for_handle(left)?;
    let right_owner = gpu_helpers::exact_provider_for_handle(right)?;
    (runmat_accelerate_api::same_provider_instance(left_owner, right_owner)
        && left.device_id() == right.device_id())
    .then_some(left_owner)
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn compatible_inputs(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> bool {
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

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn float_handle_supported(
    handle: &GpuTensorHandle,
) -> bool {
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle).is_some()
}

pub(super) fn provider_error(
    operation: ArithmeticProviderOperation,
    error: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> crate::RuntimeError {
    let error = error.into();
    let result = build_runtime_error(format!("{}: {error}", operation.name()))
        .with_builtin(operation.name())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    crate::runtime_error::with_provider_execution_source(result, error)
}

pub(super) fn provider_error_after_effect(
    operation: ArithmeticProviderOperation,
    error: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> crate::RuntimeError {
    let error = error.into().after_effect();
    let diagnostic = build_runtime_error(format!("{}: {error}", operation.name()))
        .with_builtin(operation.name())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    crate::runtime_error::with_provider_execution_source(diagnostic, error)
}

pub(super) fn provider_contract_error(
    operation: ArithmeticProviderOperation,
    message: &'static str,
) -> crate::RuntimeError {
    build_runtime_error(format!("{}: {message}", operation.name()))
        .with_builtin(operation.name())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
