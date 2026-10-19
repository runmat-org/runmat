//! Resident comparison execution and explicit-residency restoration.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::{LogicalArray, Value};

use crate::builtins::common::{broadcast::broadcast_shapes, gpu_helpers};

use super::IntegerComparisonOp;

#[cfg(test)]
#[path = "gpu_owned_tests.rs"]
mod owned_tests;

pub(crate) async fn try_gpu_equality_comparison(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    operation: IntegerComparisonOp,
) -> Option<crate::BuiltinResult<Value>> {
    if lhs.device_id() != rhs.device_id() {
        return None;
    }
    let provider = resolved_actual_owner(lhs)?;
    let rhs_owner = resolved_actual_owner(rhs)?;
    if !runmat_accelerate_api::same_provider_instance(provider, rhs_owner)
        || runmat_accelerate_api::handle_precision(lhs)
            != runmat_accelerate_api::handle_precision(rhs)
    {
        return None;
    }
    let result = match operation {
        IntegerComparisonOp::Eq => provider.elem_eq(lhs, rhs).await,
        IntegerComparisonOp::Ne => provider.elem_ne(lhs, rhs).await,
        _ => unreachable!("equality helper only supports eq and ne"),
    };
    match result {
        Ok(handle) if valid_equality_output(&handle, lhs, rhs, provider) => Some(
            gpu_helpers::propagate_owned_output_provenance(handle, [lhs, rhs])
                .and_then(gpu_helpers::logical_gpu_value),
        ),
        Ok(handle) => Some(Err(rejected(handle, operation))),
        Err(error) => provider_failure(error, operation),
    }
}

pub(crate) fn select_comparison_output_source(
    lhs: &Value,
    rhs: &Value,
    builtin: &str,
) -> crate::BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        builtin,
    )
}

pub(crate) fn restore_explicit_comparison_result(
    value: Value,
    source: Option<&GpuTensorHandle>,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    let value = match value {
        Value::Bool(bit) => Value::LogicalArray(
            LogicalArray::new(vec![u8::from(bit)], vec![1, 1]).map_err(|error| {
                crate::build_runtime_error(format!(
                    "{builtin}: invalid scalar logical result: {error}"
                ))
                .with_builtin(builtin)
                .build()
            })?,
        ),
        value => value,
    };
    let restored = gpu_helpers::restore_class_preserving_value(source, value, builtin)?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(crate::build_runtime_error(format!(
            "{builtin}: provider cannot preserve explicit gpuArray output residency"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:GpuUploadFailed"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(restored)
}

/// Executes a resident ordering comparison, extracting real lanes first when
/// either input is complex-interleaved. The logical result remains resident.
pub(crate) async fn try_gpu_ordering_comparison(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    operation: IntegerComparisonOp,
) -> Option<crate::BuiltinResult<Value>> {
    if lhs.device_id() != rhs.device_id() {
        return None;
    }
    let provider = resolved_actual_owner(lhs)?;
    let rhs_owner = resolved_actual_owner(rhs)?;
    if !runmat_accelerate_api::same_provider_instance(provider, rhs_owner) {
        return None;
    }
    let temporary_lhs;
    let temporary_rhs;
    let lhs_projected =
        runmat_accelerate_api::handle_storage(lhs) == GpuTensorStorage::ComplexInterleaved;
    let lhs_real =
        if runmat_accelerate_api::handle_storage(lhs) == GpuTensorStorage::ComplexInterleaved {
            match provider.unary_real(lhs).await {
                Ok(handle)
                    if valid_real_projection(&handle, lhs, provider)
                        && !handle.shares_allocation_with(rhs) =>
                {
                    temporary_lhs = handle;
                    &temporary_lhs
                }
                Ok(handle) => {
                    return Some(Err(rejected(handle, operation)));
                }
                Err(error) => return provider_failure(error, operation),
            }
        } else {
            lhs
        };
    let rhs_real =
        if runmat_accelerate_api::handle_storage(rhs) == GpuTensorStorage::ComplexInterleaved {
            match provider.unary_real(rhs).await {
                Ok(handle)
                    if valid_real_projection(&handle, rhs, provider)
                        && !handle.shares_allocation_with(lhs)
                        && !handle.shares_allocation_with(lhs_real) =>
                {
                    temporary_rhs = handle;
                    &temporary_rhs
                }
                Ok(handle) => {
                    return Some(Err(rejected(handle, operation)));
                }
                Err(error) => {
                    let typed = runmat_accelerate_api::ProviderExecutionError::from(error);
                    return provider_failure(
                        if lhs_projected {
                            typed.after_effect()
                        } else {
                            typed
                        },
                        operation,
                    );
                }
            }
        } else {
            rhs
        };
    if runmat_accelerate_api::handle_precision(lhs_real)
        != runmat_accelerate_api::handle_precision(rhs_real)
    {
        if lhs_projected
            || runmat_accelerate_api::handle_storage(rhs) == GpuTensorStorage::ComplexInterleaved
        {
            return provider_failure(
                runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                    "comparison projection precision changed after provider output"
                ))
                .after_effect(),
                operation,
            );
        }
        return None;
    }
    let result = match operation {
        IntegerComparisonOp::Lt => provider.elem_lt(lhs_real, rhs_real).await,
        IntegerComparisonOp::Le => provider.elem_le(lhs_real, rhs_real).await,
        IntegerComparisonOp::Gt => provider.elem_gt(lhs_real, rhs_real).await,
        IntegerComparisonOp::Ge => provider.elem_ge(lhs_real, rhs_real).await,
        IntegerComparisonOp::Eq | IntegerComparisonOp::Ne => {
            unreachable!("resident complex ordering helper only supports lt/le/gt/ge")
        }
    };
    match result {
        Ok(handle) if valid_ordering_output(&handle, lhs_real, rhs_real, provider) => Some(
            gpu_helpers::propagate_owned_output_provenance(handle, [lhs, rhs])
                .and_then(gpu_helpers::logical_gpu_value),
        ),
        Ok(handle) => Some(Err(rejected(handle, operation))),
        Err(error) => provider_failure(
            if lhs_projected
                || runmat_accelerate_api::handle_storage(rhs)
                    == GpuTensorStorage::ComplexInterleaved
            {
                error.after_effect()
            } else {
                error
            },
            operation,
        ),
    }
}

fn resolved_actual_owner(
    handle: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id())
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.shares_allocation_with(rhs)
}

fn valid_real_projection(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape() == input.shape()
        && output.device_id() == input.device_id()
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && resolved_actual_owner(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn valid_ordering_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let expected_shape = broadcast_shapes("comparison", lhs.shape(), rhs.shape()).ok();
    expected_shape.as_deref() == Some(output.shape())
        && output.device_id() == lhs.device_id()
        && !gpu_handles_alias(output, lhs)
        && !gpu_handles_alias(output, rhs)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_is_logical(output)
        && matches!(
            output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::U8
                | runmat_accelerate_api::NumericElementType::F32
                | runmat_accelerate_api::NumericElementType::F64
        )
        && resolved_actual_owner(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn valid_equality_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let expected_shape = broadcast_shapes("comparison", lhs.shape(), rhs.shape()).ok();
    expected_shape.as_deref() == Some(output.shape())
        && output.device_id() == lhs.device_id()
        && !gpu_handles_alias(output, lhs)
        && !gpu_handles_alias(output, rhs)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_is_logical(output)
        && matches!(
            output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::U8
                | runmat_accelerate_api::NumericElementType::F32
                | runmat_accelerate_api::NumericElementType::F64
        )
        && resolved_actual_owner(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn rejected(handle: GpuTensorHandle, operation: IntegerComparisonOp) -> crate::RuntimeError {
    gpu_helpers::reject_gpu_output(
        handle,
        crate::build_runtime_error(format!(
            "{}: provider returned an invalid or aliased output",
            operation.builtin_name()
        ))
        .with_builtin(operation.builtin_name())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build(),
    )
}

fn provider_failure(
    error: impl Into<runmat_accelerate_api::ProviderExecutionError>,
    operation: IntegerComparisonOp,
) -> Option<crate::BuiltinResult<Value>> {
    let error = error.into();
    if runmat_accelerate_api::is_unsupported_provider_error(&error) {
        return None;
    }
    let contextual =
        crate::build_runtime_error(format!("comparison: provider execution failed: {error}"))
            .with_builtin(operation.builtin_name())
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
    Some(Err(crate::runtime_error::with_provider_execution_source(
        contextual, error,
    )))
}
