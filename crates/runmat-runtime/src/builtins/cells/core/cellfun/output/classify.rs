use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::{IntValue, NumericScalar, Value};

use super::super::error;

pub(super) enum ClassifiedValue {
    Logical(bool),
    F64(f64),
    F32(f32),
    Integer(IntValue),
    ComplexF64((f64, f64)),
    ComplexF32((f32, f32)),
    IntegerComplex(IntValue, IntValue),
    Char(char),
}

pub(super) fn value(value: &Value) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(value) => Ok(ClassifiedValue::Logical(*value)),
        Value::LogicalArray(value) if value.len() == 1 => {
            Ok(ClassifiedValue::Logical(value.data[0] != 0))
        }
        Value::Num(value) => Ok(ClassifiedValue::F64(*value)),
        Value::Int(value) => Ok(ClassifiedValue::Integer(value.clone())),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            numeric(value.numeric_value_at(0).ok_or_else(|| {
                error::internal("cellfun: scalar tensor has no numeric storage value")
            })?)
        }
        Value::Complex(real, imaginary) => {
            Ok(ClassifiedValue::ComplexF64((*real, *imaginary)))
        }
        Value::ComplexTensor(value) if tensor::is_scalar_complex_tensor(value) => {
            let (real, imaginary) = value.numeric_value_at(0).ok_or_else(|| {
                error::internal("cellfun: scalar complex tensor has no storage value")
            })?;
            complex(real, imaginary)
        }
        Value::CharArray(value) if value.rows * value.cols == 1 => {
            Ok(ClassifiedValue::Char(value.data.first().copied().unwrap_or('\0')))
        }
        value => Err(error::uniform(format!(
            "cellfun: callback must return scalar numeric, logical, character, or complex values when UniformOutput is true, got {value:?}"
        ))),
    }
}

fn numeric(value: NumericScalar) -> BuiltinResult<ClassifiedValue> {
    match value {
        NumericScalar::F64(value) => Ok(ClassifiedValue::F64(value)),
        NumericScalar::F32(value) => Ok(ClassifiedValue::F32(value)),
        value => value
            .into_int_value()
            .map(ClassifiedValue::Integer)
            .ok_or_else(|| error::internal("cellfun: integer scalar classification failed")),
    }
}

fn complex(real: NumericScalar, imaginary: NumericScalar) -> BuiltinResult<ClassifiedValue> {
    match (real, imaginary) {
        (NumericScalar::F64(real), NumericScalar::F64(imaginary)) => {
            Ok(ClassifiedValue::ComplexF64((real, imaginary)))
        }
        (NumericScalar::F32(real), NumericScalar::F32(imaginary)) => {
            Ok(ClassifiedValue::ComplexF32((real, imaginary)))
        }
        (real, imaginary) => Ok(ClassifiedValue::IntegerComplex(
            real.into_int_value().ok_or_else(|| {
                error::internal("cellfun: complex scalar has inconsistent component classes")
            })?,
            imaginary.into_int_value().ok_or_else(|| {
                error::internal("cellfun: complex scalar has inconsistent component classes")
            })?,
        )),
    }
}
