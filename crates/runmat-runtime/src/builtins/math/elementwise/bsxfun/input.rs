use runmat_builtins::BSXFUN_ERROR_INVALID_INPUT;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, NumericScalar, Tensor, Value};

pub(super) struct ArrayInput {
    data: ArrayData,
    shape: Vec<usize>,
}

impl ArrayInput {
    pub(super) fn from_value(value: Value) -> crate::BuiltinResult<Self> {
        let data = ArrayData::from_value(value)?;
        let shape = data.shape();
        Ok(Self { data, shape })
    }

    pub(super) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(super) fn value_at(&self, index: usize) -> crate::BuiltinResult<Value> {
        self.data.value_at(index)
    }

    pub(super) fn scalar_fact(&self) -> runmat_types::ValueFact {
        self.data.scalar_fact()
    }
}

enum ArrayData {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Character(CharArray),
    Scalar(Value),
}

impl ArrayData {
    fn from_value(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Tensor(value) => Ok(Self::Tensor(value)),
            Value::LogicalArray(value) => Ok(Self::Logical(value)),
            Value::ComplexTensor(value) => Ok(Self::Complex(value)),
            Value::CharArray(value) => Ok(Self::Character(value)),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
                Ok(Self::Scalar(value))
            }
            other => Err(super::error::detail(
                &BSXFUN_ERROR_INVALID_INPUT,
                Some(format!("unsupported input type {other:?}")),
            )),
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Tensor(value) => normalize_shape(&value.shape),
            Self::Logical(value) => normalize_shape(&value.shape),
            Self::Complex(value) => normalize_shape(&value.shape),
            Self::Character(value) => vec![value.rows, value.cols],
            Self::Scalar(_) => vec![1, 1],
        }
    }

    fn value_at(&self, index: usize) -> crate::BuiltinResult<Value> {
        match self {
            Self::Tensor(value) => value
                .numeric_value_at(index)
                .map(numeric_value)
                .transpose()?
                .ok_or_else(|| super::error::internal("numeric input index is out of bounds")),
            Self::Logical(value) => value
                .data
                .get(index)
                .map(|bit| Value::Bool(*bit != 0))
                .ok_or_else(|| super::error::internal("logical input index is out of bounds")),
            Self::Complex(value) => value
                .numeric_value_at(index)
                .map(complex_value)
                .transpose()?
                .ok_or_else(|| super::error::internal("complex input index is out of bounds")),
            Self::Character(value) => character_value(value, index),
            Self::Scalar(value) => Ok(value.clone()),
        }
    }

    fn scalar_fact(&self) -> runmat_types::ValueFact {
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
            Self::Character(_) => {
                runmat_types::ValueFact::scalar(runmat_types::ValueKindFact::Character)
            }
            Self::Scalar(value) => crate::call::catalog::scalarized_fact(value),
        }
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn numeric_value(value: NumericScalar) -> crate::BuiltinResult<Value> {
    match value {
        NumericScalar::F64(value) => Ok(Value::Num(value)),
        NumericScalar::F32(value) => Tensor::from_f32(vec![value], vec![1, 1])
            .map(Value::Tensor)
            .map_err(super::error::internal),
        value => value
            .into_int_value()
            .map(Value::Int)
            .ok_or_else(|| super::error::internal("numeric scalar class is inconsistent")),
    }
}

fn complex_value((real, imaginary): (NumericScalar, NumericScalar)) -> crate::BuiltinResult<Value> {
    match (real, imaginary) {
        (NumericScalar::F64(real), NumericScalar::F64(imaginary)) => {
            Ok(Value::Complex(real, imaginary))
        }
        (NumericScalar::F32(real), NumericScalar::F32(imaginary)) => {
            ComplexTensor::from_f32(vec![(real, imaginary)], vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(super::error::internal)
        }
        (real, imaginary) => {
            let real = real
                .into_int_value()
                .ok_or_else(|| super::error::internal("complex component classes do not match"))?;
            let imaginary = imaginary
                .into_int_value()
                .ok_or_else(|| super::error::internal("complex component classes do not match"))?;
            let storage = runmat_value::IntegerComplexStorage::new(
                runmat_value::IntegerStorage::from_scalar(real),
                runmat_value::IntegerStorage::from_scalar(imaginary),
            )
            .map_err(super::error::internal)?;
            ComplexTensor::new_integer(storage, vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(super::error::internal)
        }
    }
}

fn character_value(array: &CharArray, index: usize) -> crate::BuiltinResult<Value> {
    if array.rows == 0 || array.cols == 0 {
        return CharArray::new(Vec::new(), 0, 0)
            .map(Value::CharArray)
            .map_err(super::error::internal);
    }
    let row = index % array.rows;
    let column = index / array.rows;
    let source = row * array.cols + column;
    let character = array
        .data
        .get(source)
        .copied()
        .ok_or_else(|| super::error::internal("character input index is out of bounds"))?;
    CharArray::new(vec![character], 1, 1)
        .map(Value::CharArray)
        .map_err(super::error::internal)
}
