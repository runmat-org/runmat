use crate::BuiltinResult;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

use super::super::error::arrayfun_flow;

pub(in crate::builtins::acceleration::gpu::arrayfun) enum ArrayData {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
    String(StringArray),
    Scalar(Value),
}

impl ArrayData {
    pub(in crate::builtins::acceleration::gpu::arrayfun) fn from_value(
        value: Value,
    ) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(value) => Ok(Self::Tensor(value)),
            Value::LogicalArray(value) => Ok(Self::Logical(value)),
            Value::ComplexTensor(value) => Ok(Self::Complex(value)),
            Value::CharArray(value) => Ok(Self::Char(value)),
            Value::StringArray(value) => Ok(Self::String(value)),
            Value::Num(_)
            | Value::Bool(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::String(_) => Ok(Self::Scalar(value)),
            other => Err(arrayfun_flow(format!(
                "arrayfun: unsupported input type {other:?} (expected numeric, logical, complex, char, or string arrays)"
            ))),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Tensor(value) => value.len(),
            Self::Logical(value) => value.data.len(),
            Self::Complex(value) => value.len(),
            Self::Char(value) => value.rows * value.cols,
            Self::String(value) => value.data.len(),
            Self::Scalar(_) => 1,
        }
    }

    pub(in crate::builtins::acceleration::gpu::arrayfun) fn shape_vec(&self) -> Vec<usize> {
        match self {
            Self::Tensor(value) => normalized_shape(&value.shape),
            Self::Logical(value) => normalized_shape(&value.shape),
            Self::Complex(value) => normalized_shape(&value.shape),
            Self::Char(value) => vec![value.rows, value.cols],
            Self::String(value) => normalized_shape(&value.shape),
            Self::Scalar(_) => vec![1, 1],
        }
    }

    pub(super) fn scalar_fact(&self) -> runmat_types::ValueFact {
        match self {
            Self::Tensor(value) => crate::value_fact::numeric_storage_scalar_fact(
                value.numeric_dtype(),
                runmat_types::NumericDomain::Real,
            ),
            Self::Logical(_) => {
                runmat_types::ValueFact::scalar(runmat_types::ValueKindFact::Logical)
            }
            Self::Complex(value) => crate::value_fact::numeric_storage_scalar_fact(
                value.numeric_dtype(),
                runmat_types::NumericDomain::Complex,
            ),
            Self::Char(_) => {
                runmat_types::ValueFact::scalar(runmat_types::ValueKindFact::Character)
            }
            Self::String(_) => runmat_types::ValueFact::scalar(runmat_types::ValueKindFact::String),
            Self::Scalar(value) => crate::call::catalog::scalarized_fact(value),
        }
    }
}

fn normalized_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}
