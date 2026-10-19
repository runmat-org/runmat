use super::{invalid, FitCooperation, Value};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{IntValue, LogicalArray, NumericScalar, Tensor};

pub(super) async fn numeric_tensor(
    label: &str,
    value: Value,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Tensor> {
    let tensor = match value {
        Value::LogicalArray(array) => {
            logical_tensor_cooperatively(array, label, cooperation).await?
        }
        other => tensor::value_into_tensor_for(label, other)
            .map_err(|_| invalid(format!("mnrfit: {label} must be numeric")))?,
    };
    ensure_exact_integer_tensor(&tensor, label, cooperation).await?;
    Ok(tensor)
}

pub(super) async fn admitted_host_predictor(
    value: Value,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::LogicalArray(array) => logical_tensor_cooperatively(array, "X", cooperation).await,
        scalar => numeric_tensor("X", scalar, cooperation).await,
    }
}

async fn logical_tensor_cooperatively(
    array: LogicalArray,
    label: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Tensor> {
    let values = logical_values_cooperatively(&array, label, cooperation).await?;
    Tensor::new(values, array.shape)
        .map_err(|_| invalid(format!("mnrfit: {label} must be numeric")))
}

async fn logical_values_cooperatively(
    array: &LogicalArray,
    label: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(array.data.len())
        .map_err(|_| invalid(format!("mnrfit: {label} exceeds available host resources")))?;
    for (index, value) in array.data.iter().enumerate() {
        values.push(f64::from(*value != 0));
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((array.data.len() % 16_384) as u64)
        .await?;
    Ok(values)
}

pub(super) async fn numeric_vector(
    value: Value,
    label: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    let value = match value {
        Value::LogicalArray(array) => {
            if array.shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
                return Err(invalid(format!("mnrfit: {label} must be a vector")));
            }
            return logical_values_cooperatively(&array, label, cooperation).await;
        }
        other => other,
    };
    let tensor = numeric_tensor(label, value, cooperation).await?;
    if tensor
        .shape()
        .iter()
        .copied()
        .filter(|dim| *dim > 1)
        .count()
        > 1
    {
        return Err(invalid(format!("mnrfit: {label} must be a vector")));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(tensor.len())
        .map_err(|_| invalid(format!("mnrfit: {label} exceeds available host resources")))?;
    for index in 0..tensor.len() {
        let value = tensor::tensor_value_f64(&tensor, index);
        if !value.is_finite() {
            return Err(invalid(format!(
                "mnrfit: {label} must contain finite values"
            )));
        }
        values.push(value);
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((tensor.len() % 16_384) as u64)
        .await?;
    Ok(values)
}

fn numeric_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => exact_integer_as_f64(value, label),
        Value::Bool(value) => Ok(f64::from(*value)),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor
                .numeric_value_at(0)
                .and_then(NumericScalar::into_int_value)
            {
                exact_integer_as_f64(&integer, label)?;
            }
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            Ok(f64::from(data[0] != 0))
        }
        _ => Err(invalid(format!("mnrfit: {label} must be a scalar"))),
    }
}

pub(super) fn exact_integer_as_f64(value: &IntValue, label: &str) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(invalid(format!(
            "mnrfit: integer {label} values must be exactly representable as double"
        )))
    }
}

pub(super) async fn ensure_exact_integer_tensor(
    tensor: &Tensor,
    label: &str,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for index in 0..tensor.len() {
        if let Some(value) = tensor
            .numeric_value_at(index)
            .and_then(NumericScalar::into_int_value)
        {
            exact_integer_as_f64(&value, label)?;
        }
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((tensor.len() % 16_384) as u64)
        .await?;
    Ok(())
}

pub(super) fn positive_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    let scalar = numeric_scalar(value, label)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(invalid(format!(
            "mnrfit: {label} must be positive and finite"
        )));
    }
    Ok(scalar)
}

pub(super) fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    let scalar = positive_scalar(value, label)?;
    if scalar.fract() != 0.0
        || scalar > usize::MAX as f64
        || (usize::BITS == 64 && scalar == usize::MAX as f64)
    {
        return Err(invalid(format!(
            "mnrfit: {label} must be a positive integer"
        )));
    }
    Ok(scalar as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::IntegerStorage;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    #[test]
    fn large_host_integer_admission_yields_and_cancels_before_completion() {
        let source = CancellationSource::new();
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1; 20_000]), vec![20_000, 1])
            .expect("valid integer predictor");
        let budget = crate::execution::CpuCooperationBudget::with_test_turn(
            crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(std::future::pending()),
        );
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let mut cooperation = FitCooperation::with_budget(&runtime, budget);
        let mut admission = Box::pin(ensure_exact_integer_tensor(&tensor, "X", &mut cooperation));
        let waker = noop_waker();
        assert!(admission
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        source.cancel(CancellationReason::User);
        let Poll::Ready(Err(error)) = admission.as_mut().poll(&mut Context::from_waker(&waker))
        else {
            panic!("cancellation must stop the pending exactness scan");
        };
        assert!(error.message().contains("cancelled"));
    }
}
