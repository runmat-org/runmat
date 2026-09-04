use runmat_value::{IntegerStorage, LogicalArray, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::error;

pub(super) fn column(value: Value, rows: usize) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => Tensor::new(vec![value; rows], vec![rows, 1])
            .map(Value::Tensor)
            .map_err(error::invalid),
        Value::Int(value) => integer_column(IntegerStorage::from_scalar(value), rows),
        Value::Bool(value) => LogicalArray::new(vec![u8::from(value); rows], vec![rows, 1])
            .map(Value::LogicalArray)
            .map_err(error::invalid),
        Value::Tensor(tensor) => tensor_column(tensor, rows),
        Value::LogicalArray(array) => logical_column(array, rows),
        other => Err(error::invalid(format!(
            "accumarray: unsupported data input {other:?}"
        ))),
    }
}

fn tensor_column(tensor: Tensor, rows: usize) -> BuiltinResult<Value> {
    let len = tensor_utils::tensor_element_len(&tensor);
    if len != 1 && len != rows {
        return Err(error::invalid(
            "accumarray: data must be scalar or match subscript row count",
        ));
    }
    if len == rows {
        return tensor
            .reshape(vec![rows, 1])
            .map(Value::Tensor)
            .map_err(error::invalid);
    }
    if let Some(storage) = tensor.integer_storage() {
        return integer_column(storage.clone(), rows);
    }
    Tensor::new_with_dtype(
        vec![tensor_utils::tensor_value_f64(&tensor, 0); rows],
        vec![rows, 1],
        tensor.numeric_dtype(),
    )
    .map(Value::Tensor)
    .map_err(error::invalid)
}

fn integer_column(storage: IntegerStorage, rows: usize) -> BuiltinResult<Value> {
    let value = storage
        .value_at(0)
        .ok_or_else(|| error::invalid("accumarray: scalar integer data is empty"))?;
    let repeated = storage
        .from_exact_values_like(vec![value; rows])
        .map_err(error::invalid)?;
    Tensor::new_integer(repeated, vec![rows, 1])
        .map(Value::Tensor)
        .map_err(error::invalid)
}

fn logical_column(array: LogicalArray, rows: usize) -> BuiltinResult<Value> {
    if array.data.len() != 1 && array.data.len() != rows {
        return Err(error::invalid(
            "accumarray: data must be scalar or match subscript row count",
        ));
    }
    let values = if array.data.len() == 1 {
        vec![array.data[0]; rows]
    } else {
        array.data.into_vec()
    };
    LogicalArray::new(values, vec![rows, 1])
        .map(Value::LogicalArray)
        .map_err(error::invalid)
}

pub(super) fn default_sum_values(value: Value, rows: usize) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![value; rows]),
        Value::Int(value) => Ok(vec![value.to_f64(); rows]),
        Value::Bool(value) => Ok(vec![f64::from(value); rows]),
        Value::Tensor(tensor) => {
            let values = tensor_utils::tensor_into_values_f64(tensor);
            expand_or_validate(values, rows)
        }
        Value::LogicalArray(array) => expand_or_validate(
            array
                .data
                .into_iter()
                .map(|flag| f64::from(flag != 0))
                .collect(),
            rows,
        ),
        other => Err(error::invalid(format!(
            "accumarray: unsupported data input {other:?}"
        ))),
    }
}

fn expand_or_validate(values: Vec<f64>, rows: usize) -> BuiltinResult<Vec<f64>> {
    match values.len() {
        1 => Ok(vec![values[0]; rows]),
        len if len == rows => Ok(values),
        _ => Err(error::invalid(
            "accumarray: data must be scalar or match subscript row count",
        )),
    }
}

pub(super) fn is_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

pub(super) fn is_double(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F64)
}

pub(super) fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    as_numeric_scalar(value)
        .ok_or_else(|| error::invalid(format!("{context}: expected numeric scalar")))
}

pub(super) fn as_numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(f64::from(*value)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Some(f64::from(array.data[0] != 0)),
        _ => None,
    }
}
