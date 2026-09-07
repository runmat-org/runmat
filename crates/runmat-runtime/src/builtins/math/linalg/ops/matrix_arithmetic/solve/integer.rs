use runmat_value::{IntegerStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::BuiltinResult;

use super::{errors, host, SolveOrientation};

pub(super) fn contains(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

pub(super) async fn evaluate(
    orientation: SolveOrientation,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<Value> {
    let restore_source = gpu_helpers::select_resident_output_source(
        [lhs, rhs].into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        orientation.name(),
    )
    .map_err(|error| errors::map_control_flow(orientation, error))?;
    let lhs = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(|error| errors::map_control_flow(orientation, error))?;
    let rhs = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(|error| errors::map_control_flow(orientation, error))?;
    let result = host_integer(orientation, lhs, rhs)?;
    let Some(source) = restore_source else {
        return Ok(result);
    };
    let result = match result {
        Value::Int(value) => Value::Tensor(
            Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|error| errors::internal(orientation, error))?,
        ),
        other => other,
    };
    let provider = gpu_helpers::exact_provider_for_handle(&source).ok_or_else(|| {
        errors::internal(
            orientation,
            format!("{}: resident input owner was lost", orientation.name()),
        )
    })?;
    crate::builtins::common::provider_restore::upload_value_like(
        provider,
        result,
        orientation.name(),
        &source,
    )
    .map_err(|error| errors::map_control_flow(orientation, error))
}

fn host_integer(orientation: SolveOrientation, lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let scalar = match orientation {
        SolveOrientation::Left => &lhs,
        SolveOrientation::Right => &rhs,
    };
    if !is_scalar(scalar) {
        return Err(errors::invalid_input(
            orientation,
            format!(
                "{}: integer inputs are only supported for scalar {} division",
                orientation.name(),
                match orientation {
                    SolveOrientation::Left => "left",
                    SolveOrientation::Right => "right",
                }
            ),
        ));
    }
    let (numerator, denominator) = match orientation {
        SolveOrientation::Left => (&rhs, &lhs),
        SolveOrientation::Right => (&lhs, &rhs),
    };
    if is_complex(numerator) || is_complex(denominator) {
        return host::evaluate(orientation, lhs, rhs);
    }
    if let Some(result) = try_integer_binary(
        numerator,
        denominator,
        IntegerBinaryOp::Divide,
        orientation.name(),
    )
    .map_err(|error| errors::invalid_input(orientation, error))?
    {
        Ok(result)
    } else {
        host::evaluate(orientation, lhs, rhs)
    }
}

fn is_scalar(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

fn is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(..) | Value::ComplexTensor(_))
}
