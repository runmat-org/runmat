use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE;
use runmat_value::{IntValue, IntegerStorage};

use super::super::super::error::arrayfun_error_with_detail;
use super::super::classify::ClassifiedValue;

fn heterogeneous() -> RuntimeError {
    arrayfun_error_with_detail(
        &ARRAYFUN_ERROR_UNIFORM_OUTPUT_TYPE,
        "callback outputs with UniformOutput=true must have the same data type on every invocation",
    )
}

pub(super) fn logical(values: &mut Vec<u8>, value: ClassifiedValue) -> BuiltinResult<()> {
    match value {
        ClassifiedValue::Logical(value) => values.push(value as u8),
        _ => return Err(heterogeneous()),
    }
    Ok(())
}

pub(super) fn f64(
    values: &mut Vec<f64>,
    value: ClassifiedValue,
) -> BuiltinResult<Option<Vec<(f64, f64)>>> {
    match value {
        ClassifiedValue::F64(value) => values.push(value),
        ClassifiedValue::ComplexF64(value) => {
            let mut complex = std::mem::take(values)
                .into_iter()
                .map(|value| (value, 0.0))
                .collect::<Vec<_>>();
            complex.push(value);
            return Ok(Some(complex));
        }
        _ => return Err(heterogeneous()),
    }
    Ok(None)
}

pub(super) fn f32(
    values: &mut Vec<f32>,
    value: ClassifiedValue,
) -> BuiltinResult<Option<Vec<(f32, f32)>>> {
    match value {
        ClassifiedValue::F32(value) => values.push(value),
        ClassifiedValue::ComplexF32(value) => {
            let mut complex = std::mem::take(values)
                .into_iter()
                .map(|value| (value, 0.0))
                .collect::<Vec<_>>();
            complex.push(value);
            return Ok(Some(complex));
        }
        _ => return Err(heterogeneous()),
    }
    Ok(None)
}

pub(super) fn integer(
    prototype: &IntegerStorage,
    values: &mut Vec<IntValue>,
    value: ClassifiedValue,
) -> BuiltinResult<()> {
    let ClassifiedValue::Integer(value) = value else {
        return Err(heterogeneous());
    };
    if IntegerStorage::from_scalar(value.clone()).numeric_dtype() != prototype.numeric_dtype() {
        return Err(heterogeneous());
    }
    values.push(value);
    Ok(())
}

pub(super) fn complex_f64(
    values: &mut Vec<(f64, f64)>,
    value: ClassifiedValue,
) -> BuiltinResult<()> {
    match value {
        ClassifiedValue::F64(value) => values.push((value, 0.0)),
        ClassifiedValue::ComplexF64(value) => values.push(value),
        _ => return Err(heterogeneous()),
    }
    Ok(())
}

pub(super) fn complex_f32(
    values: &mut Vec<(f32, f32)>,
    value: ClassifiedValue,
) -> BuiltinResult<()> {
    match value {
        ClassifiedValue::F32(value) => values.push((value, 0.0)),
        ClassifiedValue::ComplexF32(value) => values.push(value),
        _ => return Err(heterogeneous()),
    }
    Ok(())
}

pub(super) fn integer_complex(
    real_prototype: &IntegerStorage,
    imag_prototype: &IntegerStorage,
    real_values: &mut Vec<IntValue>,
    imag_values: &mut Vec<IntValue>,
    value: ClassifiedValue,
) -> BuiltinResult<()> {
    let ClassifiedValue::IntegerComplex(real, imag) = value else {
        return Err(heterogeneous());
    };
    if IntegerStorage::from_scalar(real.clone()).numeric_dtype() != real_prototype.numeric_dtype()
        || IntegerStorage::from_scalar(imag.clone()).numeric_dtype()
            != imag_prototype.numeric_dtype()
    {
        return Err(heterogeneous());
    }
    real_values.push(real);
    imag_values.push(imag);
    Ok(())
}

pub(super) fn character(values: &mut Vec<char>, value: ClassifiedValue) -> BuiltinResult<()> {
    let ClassifiedValue::Char(value) = value else {
        return Err(heterogeneous());
    };
    values.push(value);
    Ok(())
}
