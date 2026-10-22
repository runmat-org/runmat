//! Borrowed exact `dateshift` weekday/rule integers; no tagged temporary Vec.
use crate::builtins::common::tensor;
use crate::builtins::datetime::datetime_error;
use crate::BuiltinResult;
use runmat_value::{IntValue, Tensor, Value};

pub(super) enum IntegerView<'a> {
    Scalar(i64),
    Tensor(&'a Tensor),
}

impl<'a> IntegerView<'a> {
    pub(super) fn borrow(value: &'a Value, context: &str) -> BuiltinResult<Self> {
        match value {
            Value::Int(value) => value.try_to_i64().map(Self::Scalar).ok_or_else(|| {
                datetime_error(format!("dateshift: {context} is outside int64 range"))
            }),
            Value::Num(value) => super::super::super::dateshift_model::exact_f64_to_i64(*value)
                .map(Self::Scalar)
                .ok_or_else(|| {
                    datetime_error(format!(
                        "dateshift: {context} must be a representable integer"
                    ))
                }),
            Value::Tensor(tensor) => Ok(Self::Tensor(tensor)),
            Value::Bool(_) | Value::LogicalArray(_) => Err(datetime_error(format!(
                "dateshift: {context} does not accept logical values"
            ))),
            _ => Err(datetime_error(format!(
                "dateshift: {context} must be an integer scalar or array"
            ))),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.len(),
        }
    }

    pub(super) fn shape(&self) -> &'a [usize] {
        const SCALAR: &[usize] = &[1, 1];
        match self {
            Self::Scalar(_) => SCALAR,
            Self::Tensor(tensor) => tensor.shape(),
        }
    }

    pub(super) fn at(&self, index: usize, context: &str) -> BuiltinResult<i64> {
        match self {
            Self::Scalar(value) => Ok(*value),
            Self::Tensor(tensor) => {
                if let Some(storage) = tensor.integer_storage() {
                    let value: IntValue = storage
                        .value_at(index)
                        .expect("admitted integer index is within payload");
                    value.try_to_i64().ok_or_else(|| {
                        datetime_error(format!("dateshift: {context} is outside int64 range"))
                    })
                } else {
                    super::super::super::dateshift_model::exact_f64_to_i64(
                        tensor::tensor_value_f64(tensor, index),
                    )
                    .ok_or_else(|| {
                        datetime_error(format!(
                            "dateshift: {context} values must be representable integers"
                        ))
                    })
                }
            }
        }
    }
}
