use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_builtins::ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE;
use runmat_value::{IntValue, NumericScalar, Value};

use super::super::error::{arrayfun_error_with_detail, arrayfun_internal};

pub(in crate::builtins::acceleration::gpu::arrayfun) enum ClassifiedValue {
    Logical(bool),
    F64(f64),
    F32(f32),
    Integer(IntValue),
    ComplexF64((f64, f64)),
    ComplexF32((f32, f32)),
    IntegerComplex(IntValue, IntValue),
    Char(char),
}

pub(in crate::builtins::acceleration::gpu::arrayfun) fn classify_value(
    value: &Value,
) -> BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(b) => Ok(ClassifiedValue::Logical(*b)),
        Value::LogicalArray(la) if la.len() == 1 => Ok(ClassifiedValue::Logical(la.data[0] != 0)),
        Value::Int(value) => Ok(ClassifiedValue::Integer(value.clone())),
        Value::Num(n) => Ok(ClassifiedValue::F64(*n)),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            match t.numeric_value_at(0).ok_or_else(|| {
                arrayfun_internal("arrayfun: scalar tensor has no numeric storage value")
            })? {
                NumericScalar::F64(value) => Ok(ClassifiedValue::F64(value)),
                NumericScalar::F32(value) => Ok(ClassifiedValue::F32(value)),
                value => Ok(ClassifiedValue::Integer(
                    value.into_int_value().ok_or_else(|| {
                        arrayfun_internal("arrayfun: integer scalar classification failed")
                    })?,
                )),
            }
        }
        Value::Complex(re, im) => Ok(ClassifiedValue::ComplexF64((*re, *im))),
        Value::ComplexTensor(t) if tensor::is_scalar_complex_tensor(t) => {
            let (real, imag) = t.numeric_value_at(0).ok_or_else(|| {
                arrayfun_internal("arrayfun: scalar complex tensor has no storage value")
            })?;
            match (real, imag) {
                (NumericScalar::F64(real), NumericScalar::F64(imag)) => {
                    Ok(ClassifiedValue::ComplexF64((real, imag)))
                }
                (NumericScalar::F32(real), NumericScalar::F32(imag)) => {
                    Ok(ClassifiedValue::ComplexF32((real, imag)))
                }
                (real, imag) => Ok(ClassifiedValue::IntegerComplex(
                    real.into_int_value().ok_or_else(|| {
                        arrayfun_internal(
                            "arrayfun: complex callback result has inconsistent component classes",
                        )
                    })?,
                    imag.into_int_value().ok_or_else(|| {
                        arrayfun_internal(
                            "arrayfun: complex callback result has inconsistent component classes",
                        )
                    })?,
                )),
            }
        }
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            let ch = ca.data.first().copied().unwrap_or('\0');
            Ok(ClassifiedValue::Char(ch))
        }
        other => Err(arrayfun_error_with_detail(
            &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
            format!(
                "callback must return scalar numeric, logical, character, or complex values for UniformOutput=true (got {other:?})"
            ),
        )),
    }
}
