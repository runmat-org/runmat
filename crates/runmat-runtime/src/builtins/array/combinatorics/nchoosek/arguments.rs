use runmat_builtins::NCHOOSEK_ERROR_INVALID_INPUT;
use runmat_value::{Tensor, Value};

use crate::BuiltinResult;

use super::error;
use super::numeric::{self, CoefficientClass};

#[derive(Debug, Copy, Clone)]
pub(super) struct ScalarCoefficient {
    pub n: usize,
    pub class: CoefficientClass,
}

#[derive(Debug, Copy, Clone)]
pub(super) struct Selection {
    pub value: usize,
    pub class: CoefficientClass,
}

pub(super) fn scalar_coefficient(value: &Value) -> Option<ScalarCoefficient> {
    match value {
        Value::Num(value) => numeric::nonnegative_f64(*value).map(|n| ScalarCoefficient {
            n,
            class: CoefficientClass::Double,
        }),
        Value::Int(value) => numeric::nonnegative_int(value).map(|n| ScalarCoefficient {
            n,
            class: numeric::int_class(value),
        }),
        Value::Tensor(tensor) if numeric::tensor_element_len(tensor) == 1 => {
            let scalar = tensor.numeric_value_at(0)?;
            if let Some(value) = scalar.into_int_value() {
                return numeric::nonnegative_int(&value).map(|n| ScalarCoefficient {
                    n,
                    class: numeric::int_class(&value),
                });
            }
            numeric::nonnegative_scalar(scalar).map(|n| ScalarCoefficient {
                n,
                class: numeric::tensor_class(tensor.numeric_dtype()),
            })
        }
        _ => None,
    }
}

pub(super) fn selection(value: &Value) -> BuiltinResult<Selection> {
    let parsed = match value {
        Value::Num(value) => numeric::nonnegative_f64(*value).map(|value| Selection {
            value,
            class: CoefficientClass::Double,
        }),
        Value::Int(value) => numeric::nonnegative_int(value).map(|parsed| Selection {
            value: parsed,
            class: numeric::int_class(value),
        }),
        Value::Tensor(tensor) if numeric::tensor_element_len(tensor) == 1 => {
            tensor_selection(tensor)
        }
        _ => None,
    };
    parsed.ok_or_else(|| {
        error::with_message(
            &NCHOOSEK_ERROR_INVALID_INPUT,
            "nchoosek: k must be a nonnegative integer scalar",
        )
    })
}

fn tensor_selection(tensor: &Tensor) -> Option<Selection> {
    let scalar = tensor.numeric_value_at(0)?;
    if let Some(value) = scalar.into_int_value() {
        let class = numeric::int_class(&value);
        return numeric::nonnegative_int(&value).map(|value| Selection { value, class });
    }
    numeric::nonnegative_scalar(scalar).map(|value| Selection {
        value,
        class: numeric::tensor_class(tensor.numeric_dtype()),
    })
}

pub(super) fn is_numeric_scalar(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if numeric::tensor_element_len(tensor) == 1)
}

pub(super) fn vector_len(shape: &[usize]) -> BuiltinResult<usize> {
    match shape {
        [] => Ok(1),
        [n] => Ok(*n),
        [0, _] | [_, 0] => Ok(0),
        [1, n] | [n, 1] => Ok(*n),
        _ => Err(error::with_message(
            &NCHOOSEK_ERROR_INVALID_INPUT,
            "nchoosek: first input must be a scalar or vector",
        )),
    }
}
