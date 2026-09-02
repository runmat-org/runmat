use runmat_builtins::RelationalOperator;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::tensor;

use super::errors::{runtime_error, ComparisonError};

pub(super) enum Operand {
    Numeric(Buffer<f64>),
    Complex(Buffer<(f64, f64)>),
    Text(Buffer<String>),
}

pub(super) struct Buffer<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl Operand {
    pub(super) fn from_value(
        value: Value,
        operator: RelationalOperator,
    ) -> crate::BuiltinResult<Self> {
        let operand = match value {
            Value::Num(value) => Self::Numeric(Buffer::scalar(value)),
            Value::Bool(value) => Self::Numeric(Buffer::scalar(f64::from(u8::from(value)))),
            Value::Int(value) => Self::Numeric(Buffer::scalar(value.to_f64())),
            Value::Tensor(value) => Self::Numeric(Buffer::from_tensor(value)),
            Value::LogicalArray(value) => Self::Numeric(Buffer::from_logical(value)),
            Value::CharArray(value) => Self::Numeric(Buffer::from_chars(value)),
            Value::Complex(real, imaginary) => Self::Complex(Buffer::scalar((real, imaginary))),
            Value::ComplexTensor(value) => Self::Complex(Buffer::from_complex(value)),
            Value::String(value) => Self::Text(Buffer::scalar(value)),
            Value::StringArray(value) => Self::Text(Buffer::from_strings(value)),
            _ => return Err(runtime_error(operator, ComparisonError::InvalidInput)),
        };
        Ok(operand)
    }
}

impl Buffer<f64> {
    fn from_tensor(value: Tensor) -> Self {
        let shape = value.shape.clone();
        Self {
            data: tensor::tensor_into_values_f64(value),
            shape,
        }
    }

    fn from_logical(value: LogicalArray) -> Self {
        Self {
            shape: value.shape,
            data: value
                .data
                .into_iter()
                .map(|item| f64::from(u8::from(item != 0)))
                .collect(),
        }
    }

    fn from_chars(value: CharArray) -> Self {
        let CharArray {
            data, rows, cols, ..
        } = value;
        let mut ordered = Vec::with_capacity(data.len());
        for column in 0..cols {
            for row in 0..rows {
                ordered.push(data[row * cols + column] as u32 as f64);
            }
        }
        Self {
            data: ordered,
            shape: vec![rows, cols],
        }
    }
}

impl Buffer<(f64, f64)> {
    fn from_complex(value: ComplexTensor) -> Self {
        let shape = value.shape.clone();
        Self {
            data: value.materialize_f64(),
            shape,
        }
    }
}

impl Buffer<String> {
    fn from_strings(value: StringArray) -> Self {
        Self {
            data: value.data,
            shape: value.shape,
        }
    }
}

impl<T> Buffer<T> {
    fn scalar(value: T) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }
}

pub(super) fn compare(
    lhs: Operand,
    rhs: Operand,
    operator: RelationalOperator,
) -> crate::BuiltinResult<Value> {
    let (data, shape) = match (lhs, rhs) {
        (Operand::Numeric(lhs), Operand::Numeric(rhs)) => {
            broadcast_compare(&lhs, &rhs, operator, |lhs, rhs| {
                apply_real(operator, *lhs, *rhs)
            })?
        }
        (Operand::Complex(lhs), Operand::Complex(rhs)) => {
            broadcast_compare(&lhs, &rhs, operator, |lhs, rhs| {
                apply_complex(operator, *lhs, *rhs)
            })?
        }
        (Operand::Numeric(lhs), Operand::Complex(rhs)) if operator.is_equality() => {
            let lhs = promote_real(lhs);
            broadcast_compare(&lhs, &rhs, operator, |lhs, rhs| {
                apply_complex(operator, *lhs, *rhs)
            })?
        }
        (Operand::Complex(lhs), Operand::Numeric(rhs)) if operator.is_equality() => {
            let rhs = promote_real(rhs);
            broadcast_compare(&lhs, &rhs, operator, |lhs, rhs| {
                apply_complex(operator, *lhs, *rhs)
            })?
        }
        (Operand::Text(lhs), Operand::Text(rhs)) => {
            broadcast_compare(&lhs, &rhs, operator, |lhs, rhs| {
                apply_ordering(operator, lhs.cmp(rhs))
            })?
        }
        _ => return Err(runtime_error(operator, ComparisonError::InvalidInput)),
    };
    logical_value(data, shape, operator)
}

fn broadcast_compare<T>(
    lhs: &Buffer<T>,
    rhs: &Buffer<T>,
    operator: RelationalOperator,
    predicate: impl Fn(&T, &T) -> bool,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(operator.name(), &lhs.shape, &rhs.shape)
        .map_err(|_| runtime_error(operator, ComparisonError::SizeMismatch))?;
    let total = tensor::element_count(&shape);
    let lhs_strides = compute_strides(&lhs.shape);
    let rhs_strides = compute_strides(&rhs.shape);
    let mut data = Vec::with_capacity(total);
    for index in 0..total {
        let lhs_index = broadcast_index(index, &shape, &lhs.shape, &lhs_strides);
        let rhs_index = broadcast_index(index, &shape, &rhs.shape, &rhs_strides);
        data.push(u8::from(predicate(
            &lhs.data[lhs_index],
            &rhs.data[rhs_index],
        )));
    }
    Ok((data, shape))
}

fn logical_value(
    data: Vec<u8>,
    shape: Vec<usize>,
    operator: RelationalOperator,
) -> crate::BuiltinResult<Value> {
    if data.len() == 1 {
        return Ok(Value::Bool(data[0] != 0));
    }
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(|_| runtime_error(operator, ComparisonError::InvalidInput))
}

fn promote_real(value: Buffer<f64>) -> Buffer<(f64, f64)> {
    Buffer {
        data: value.data.into_iter().map(|real| (real, 0.0)).collect(),
        shape: value.shape,
    }
}

fn apply_real(operator: RelationalOperator, lhs: f64, rhs: f64) -> bool {
    match operator {
        RelationalOperator::Equal => lhs == rhs,
        RelationalOperator::NotEqual => lhs != rhs,
        RelationalOperator::LessThan => lhs < rhs,
        RelationalOperator::LessThanOrEqual => lhs <= rhs,
        RelationalOperator::GreaterThan => lhs > rhs,
        RelationalOperator::GreaterThanOrEqual => lhs >= rhs,
    }
}

fn apply_complex(operator: RelationalOperator, lhs: (f64, f64), rhs: (f64, f64)) -> bool {
    match operator {
        RelationalOperator::Equal => lhs == rhs,
        RelationalOperator::NotEqual => lhs != rhs,
        _ => apply_real(operator, lhs.0, rhs.0),
    }
}

fn apply_ordering(operator: RelationalOperator, ordering: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::{Equal, Greater, Less};
    match operator {
        RelationalOperator::Equal => ordering == Equal,
        RelationalOperator::NotEqual => ordering != Equal,
        RelationalOperator::LessThan => ordering == Less,
        RelationalOperator::LessThanOrEqual => ordering != Greater,
        RelationalOperator::GreaterThan => ordering == Greater,
        RelationalOperator::GreaterThanOrEqual => ordering != Less,
    }
}
