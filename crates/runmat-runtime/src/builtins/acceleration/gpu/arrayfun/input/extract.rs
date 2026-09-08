use crate::BuiltinResult;
use runmat_value::{
    CharArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericScalar, Tensor, Value,
};

use super::super::error::{arrayfun_flow, arrayfun_internal};
use super::data::ArrayData;

impl ArrayData {
    pub(in crate::builtins::acceleration::gpu::arrayfun) fn value_at(
        &self,
        index: usize,
    ) -> BuiltinResult<Value> {
        match self {
            Self::Tensor(value) => numeric(value, index),
            Self::Logical(value) => Ok(Value::Bool(
                *value
                    .data
                    .get(index)
                    .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?
                    != 0,
            )),
            Self::Complex(value) => complex(value, index),
            Self::Char(value) => character(value, index),
            Self::String(value) => {
                Ok(Value::String(value.data.get(index).cloned().ok_or_else(
                    || arrayfun_flow("arrayfun: index out of bounds"),
                )?))
            }
            Self::Scalar(value) => Ok(value.clone()),
        }
    }
}

fn numeric(value: &Tensor, index: usize) -> BuiltinResult<Value> {
    match value
        .numeric_value_at(index)
        .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?
    {
        NumericScalar::F64(value) => Ok(Value::Num(value)),
        NumericScalar::F32(value) => Tensor::from_f32(vec![value], vec![1, 1])
            .map(Value::Tensor)
            .map_err(arrayfun_internal),
        value => value
            .into_int_value()
            .map(Value::Int)
            .ok_or_else(|| arrayfun_internal("arrayfun: integer scalar classification failed")),
    }
}

fn complex(value: &ComplexTensor, index: usize) -> BuiltinResult<Value> {
    let (real, imaginary) = value
        .numeric_value_at(index)
        .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?;
    match (real, imaginary) {
        (NumericScalar::F64(real), NumericScalar::F64(imaginary)) => {
            Ok(Value::Complex(real, imaginary))
        }
        (NumericScalar::F32(real), NumericScalar::F32(imaginary)) => {
            ComplexTensor::from_f32(vec![(real, imaginary)], vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(arrayfun_internal)
        }
        (real, imaginary) => integer_complex(real, imaginary),
    }
}

fn integer_complex(real: NumericScalar, imaginary: NumericScalar) -> BuiltinResult<Value> {
    let real = real.into_int_value().ok_or_else(|| {
        arrayfun_internal("arrayfun: complex scalar components have inconsistent classes")
    })?;
    let imaginary = imaginary.into_int_value().ok_or_else(|| {
        arrayfun_internal("arrayfun: complex scalar components have inconsistent classes")
    })?;
    IntegerComplexStorage::new(
        IntegerStorage::from_scalar(real),
        IntegerStorage::from_scalar(imaginary),
    )
    .and_then(|storage| ComplexTensor::new_integer(storage, vec![1, 1]))
    .map(Value::ComplexTensor)
    .map_err(arrayfun_internal)
}

fn character(value: &CharArray, index: usize) -> BuiltinResult<Value> {
    if value.rows == 0 || value.cols == 0 {
        return CharArray::new(Vec::new(), 0, 0)
            .map(Value::CharArray)
            .map_err(arrayfun_internal);
    }
    let row = index % value.rows;
    let column = index / value.rows;
    let character = *value
        .data
        .get(row * value.cols + column)
        .ok_or_else(|| arrayfun_flow("arrayfun: index out of bounds"))?;
    CharArray::new(vec![character], 1, 1)
        .map(Value::CharArray)
        .map_err(arrayfun_internal)
}
