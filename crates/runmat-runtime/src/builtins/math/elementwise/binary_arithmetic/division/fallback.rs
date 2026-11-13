//! Integer admission and residency-preserving host fallback for division.

use super::{DivisionContext, Value};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult};

pub(super) fn operand_is_scalar(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => tensor::is_scalar_tensor(value),
        Value::GpuTensor(value) => value.shape().iter().all(|&extent| extent == 1),
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) | Value::Bool(_) => true,
        _ => false,
    }
}

pub(super) fn validate_integer_admission(
    context: DivisionContext,
    numerator: &Value,
    denominator: &Value,
) -> BuiltinResult<()> {
    crate::builtins::math::elementwise::integer_arithmetic::validate_integer_binary_inputs(
        numerator,
        denominator,
        operand_is_scalar(numerator),
        operand_is_scalar(denominator),
        context.identity.name,
    )
    .map_err(|detail| {
        build_runtime_error(detail)
            .with_builtin(context.identity.name)
            .build()
    })
}

pub(super) fn source_handle(value: &Value) -> Option<&runmat_accelerate_api::GpuTensorHandle> {
    match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }
}

pub(super) fn preferred_source(
    numerator: &Value,
    denominator: &Value,
) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let handles = [source_handle(numerator), source_handle(denominator)];
    handles
        .iter()
        .flatten()
        .find(|handle| runmat_accelerate_api::handle_is_explicit(handle))
        .or_else(|| handles.iter().flatten().next())
        .map(|handle| (*handle).clone())
}

pub(super) async fn gather_operand(context: DivisionContext, value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, context.identity.name)),
        _ => Ok(value),
    }
}

pub(super) fn restore_residency(
    context: DivisionContext,
    source: &runmat_accelerate_api::GpuTensorHandle,
    mut host: Value,
    originals: &[runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    if matches!(host, Value::Num(_) | Value::Int(_) | Value::Bool(_)) {
        host = Value::Tensor(
            tensor::value_into_tensor_for(context.identity.name, host)
                .map_err(|error| context.internal_error(error))?,
        );
    }
    let protected = originals.iter().collect::<Vec<_>>();
    let restored = gpu_helpers::restore_class_preserving_value_protecting(
        source,
        host,
        context.identity.name,
        &protected,
    )?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(context.internal_error("provider cannot preserve explicit gpuArray output"));
    }
    Ok(restored)
}
