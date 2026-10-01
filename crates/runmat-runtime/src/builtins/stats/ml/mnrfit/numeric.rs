use super::{invalid, Value};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{IntValue, LogicalArray, NumericScalar, Tensor};

pub(super) fn numeric_tensor(label: &str, value: Value) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_into_tensor_for(label, value)
        .map_err(|_| invalid(format!("mnrfit: {label} must be numeric")))?;
    ensure_exact_integer_tensor(&tensor, label)?;
    Ok(tensor)
}

pub(super) fn numeric_vector(value: Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = numeric_tensor(label, value)?;
    if tensor.shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(format!("mnrfit: {label} must be a vector")));
    }
    let values = tensor::tensor_values_f64(&tensor);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "mnrfit: {label} must contain finite values"
        )));
    }
    Ok(values)
}

fn numeric_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => exact_integer_as_f64(value, label),
        Value::Bool(value) => Ok(f64::from(*value)),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, label)?;
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            Ok(f64::from(data[0] != 0))
        }
        _ => Err(invalid(format!("mnrfit: {label} must be a scalar"))),
    }
}

fn exact_integer_as_f64(value: &IntValue, label: &str) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(invalid(format!(
            "mnrfit: integer {label} values must be exactly representable as double"
        )))
    }
}

pub(super) fn ensure_exact_integer_tensor(tensor: &Tensor, label: &str) -> BuiltinResult<()> {
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
    }
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
