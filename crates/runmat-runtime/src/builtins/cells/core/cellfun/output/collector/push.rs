use crate::BuiltinResult;
use runmat_value::{IntValue, IntegerStorage};

use super::super::super::error;
use super::super::classify::ClassifiedValue;

fn heterogeneous() -> crate::RuntimeError {
    error::uniform(
        "cellfun: callback outputs with UniformOutput=true must have the same data type on every invocation",
    )
}

pub(super) enum LogicalPromotion {
    F64(Vec<f64>),
    ComplexF64(Vec<(f64, f64)>),
}

pub(super) fn logical(
    values: &mut Vec<u8>,
    value: ClassifiedValue,
) -> BuiltinResult<Option<LogicalPromotion>> {
    match value {
        ClassifiedValue::Logical(value) => values.push(value as u8),
        ClassifiedValue::F64(value) => {
            let mut promoted = std::mem::take(values)
                .into_iter()
                .map(|value| if value == 0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>();
            promoted.push(value);
            return Ok(Some(LogicalPromotion::F64(promoted)));
        }
        ClassifiedValue::ComplexF64(value) => {
            let mut promoted = std::mem::take(values)
                .into_iter()
                .map(|value| (if value == 0 { 0.0 } else { 1.0 }, 0.0))
                .collect::<Vec<_>>();
            promoted.push(value);
            return Ok(Some(LogicalPromotion::ComplexF64(promoted)));
        }
        _ => return Err(heterogeneous()),
    }
    Ok(None)
}

pub(super) fn f64(
    values: &mut Vec<f64>,
    value: ClassifiedValue,
) -> BuiltinResult<Option<Vec<(f64, f64)>>> {
    match value {
        ClassifiedValue::Logical(value) => values.push(if value { 1.0 } else { 0.0 }),
        ClassifiedValue::F64(value) => values.push(value),
        ClassifiedValue::ComplexF64(value) => {
            let mut promoted = std::mem::take(values)
                .into_iter()
                .map(|value| (value, 0.0))
                .collect::<Vec<_>>();
            promoted.push(value);
            return Ok(Some(promoted));
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
            let mut promoted = std::mem::take(values)
                .into_iter()
                .map(|value| (value, 0.0))
                .collect::<Vec<_>>();
            promoted.push(value);
            return Ok(Some(promoted));
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
        ClassifiedValue::Logical(value) => values.push((if value { 1.0 } else { 0.0 }, 0.0)),
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
    prototype: &IntegerStorage,
    real: &mut Vec<IntValue>,
    imaginary: &mut Vec<IntValue>,
    value: ClassifiedValue,
) -> BuiltinResult<()> {
    let ClassifiedValue::IntegerComplex(next_real, next_imaginary) = value else {
        return Err(heterogeneous());
    };
    let real_type = IntegerStorage::from_scalar(next_real.clone()).numeric_dtype();
    let imaginary_type = IntegerStorage::from_scalar(next_imaginary.clone()).numeric_dtype();
    if real_type != prototype.numeric_dtype() || imaginary_type != prototype.numeric_dtype() {
        return Err(heterogeneous());
    }
    real.push(next_real);
    imaginary.push(next_imaginary);
    Ok(())
}

pub(super) fn character(values: &mut Vec<char>, value: ClassifiedValue) -> BuiltinResult<()> {
    let ClassifiedValue::Char(value) = value else {
        return Err(heterogeneous());
    };
    values.push(value);
    Ok(())
}
