//! Set equality is deliberately independent of the ordering relation. NaNs sort
//! last but never match, including repeated occurrences in the same operand.
use super::float_order::SetFloat;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{SharedShape, Value};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Operand {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum ElementKey {
    Real(u64),
    Complex(u64, u64),
    Missing { operand: Operand, record: usize },
}
impl ElementKey {
    pub(super) fn real<T: SetFloat>(value: T, operand: Operand, record: usize) -> Self {
        if value.is_nan() {
            Self::Missing { operand, record }
        } else {
            Self::Real(value.canonical_key())
        }
    }
    pub(super) fn complex<T: SetFloat>(value: (T, T), operand: Operand, record: usize) -> Self {
        if value.0.is_nan() || value.1.is_nan() {
            Self::Missing { operand, record }
        } else {
            Self::Complex(value.0.canonical_key(), value.1.canonical_key())
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum ElementOrientation {
    BothRow,
    ARow,
}
impl ElementOrientation {
    pub(super) fn row(self, a: &[usize], b: &[usize]) -> bool {
        is_row(a) && (matches!(self, Self::ARow) || is_row(b))
    }
    pub(super) fn values_row(self, a: &Value, b: &Value) -> bool {
        value_is_row(a) && (matches!(self, Self::ARow) || value_is_row(b))
    }
}
fn is_row(shape: &[usize]) -> bool {
    shape.len() == 1
        || (!shape.is_empty() && shape[0] == 1 && shape.iter().skip(2).all(|extent| *extent == 1))
}
fn value_is_row(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => is_row(value.shape()),
        Value::ComplexTensor(value) => is_row(value.shape()),
        Value::LogicalArray(value) => is_row(&value.shape),
        Value::CharArray(value) => is_row(value.shape()),
        Value::StringArray(value) => is_row(value.shape()),
        Value::GpuTensor(value) => is_row(value.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(..) | Value::String(_) => {
            true
        }
        _ => false,
    }
}
/// Only the newly produced element result changes geometry. Tensor payloads and
/// text/character owners move unchanged; no workspace payload is copied.
pub(super) fn orient(
    value: Value,
    row: bool,
    error: fn(String) -> RuntimeError,
) -> BuiltinResult<Value> {
    if !row {
        return Ok(value);
    }
    Ok(match value {
        Value::Tensor(value) => {
            let count = value.len();
            Value::Tensor(value.reshape(vec![1, count]).map_err(error)?)
        }
        Value::ComplexTensor(mut value) => {
            value
                .reshape_in_place(vec![1, value.len()])
                .map_err(error)?;
            Value::ComplexTensor(value)
        }
        Value::CharArray(mut value) => {
            value.validate_layout().map_err(error)?;
            let count = value
                .rows
                .checked_mul(value.cols)
                .ok_or_else(|| error("set result shape overflow".into()))?;
            value.shape = SharedShape::new_unmanaged(vec![1, count]);
            value.rows = 1;
            value.cols = count;
            Value::CharArray(value)
        }
        Value::StringArray(mut value) => {
            let count = value
                .rows
                .checked_mul(value.cols)
                .ok_or_else(|| error("set result shape overflow".into()))?;
            value.shape = SharedShape::new_unmanaged(vec![1, count]);
            value.rows = 1;
            value.cols = count;
            Value::StringArray(value)
        }
        scalar => scalar,
    })
}

pub(super) fn logical(value: Value, error: fn(String) -> RuntimeError) -> BuiltinResult<Value> {
    let tensor =
        crate::builtins::common::tensor::value_into_tensor_for("setdiff", value).map_err(error)?;
    let shape = tensor.shape().to_vec();
    let runmat_value::NumericStorage::F64(values) = tensor.into_numeric_storage().map_err(error)?
    else {
        return Err(error(
            "setdiff logical result lost its comparison class".into(),
        ));
    };
    if values.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(error(
            "setdiff logical result contains a nonlogical value".into(),
        ));
    }
    if values.len() == 1 {
        return Ok(Value::Bool(values[0] != 0.0));
    }
    runmat_value::LogicalArray::new(
        values
            .into_iter()
            .map(|value| u8::from(value != 0.0))
            .collect(),
        shape,
    )
    .map(Value::LogicalArray)
    .map_err(error)
}
