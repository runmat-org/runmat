use runmat_value::{IntValue, IntegerStorage};

use super::classify::ClassifiedValue;
use super::UniformScalarError;

pub(super) enum LogicalPromotion {
    F64(Vec<f64>),
    ComplexF64(Vec<(f64, f64)>),
}

pub(super) fn logical(
    values: &mut Vec<u8>,
    value: ClassifiedValue,
) -> Result<Option<LogicalPromotion>, UniformScalarError> {
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
        _ => return Err(UniformScalarError::Heterogeneous),
    }
    Ok(None)
}

pub(super) fn f64(
    values: &mut Vec<f64>,
    value: ClassifiedValue,
) -> Result<Option<Vec<(f64, f64)>>, UniformScalarError> {
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
        _ => return Err(UniformScalarError::Heterogeneous),
    }
    Ok(None)
}

pub(super) fn f32(
    values: &mut Vec<f32>,
    value: ClassifiedValue,
) -> Result<Option<Vec<(f32, f32)>>, UniformScalarError> {
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
        _ => return Err(UniformScalarError::Heterogeneous),
    }
    Ok(None)
}

pub(super) fn integer(
    prototype: &IntegerStorage,
    values: &mut Vec<IntValue>,
    value: ClassifiedValue,
) -> Result<(), UniformScalarError> {
    let ClassifiedValue::Integer(value) = value else {
        return Err(UniformScalarError::Heterogeneous);
    };
    if IntegerStorage::from_scalar(value.clone()).numeric_dtype() != prototype.numeric_dtype() {
        return Err(UniformScalarError::Heterogeneous);
    }
    values.push(value);
    Ok(())
}

pub(super) fn complex_f64(
    values: &mut Vec<(f64, f64)>,
    value: ClassifiedValue,
) -> Result<(), UniformScalarError> {
    match value {
        ClassifiedValue::Logical(value) => values.push((if value { 1.0 } else { 0.0 }, 0.0)),
        ClassifiedValue::F64(value) => values.push((value, 0.0)),
        ClassifiedValue::ComplexF64(value) => values.push(value),
        _ => return Err(UniformScalarError::Heterogeneous),
    }
    Ok(())
}

pub(super) fn complex_f32(
    values: &mut Vec<(f32, f32)>,
    value: ClassifiedValue,
) -> Result<(), UniformScalarError> {
    match value {
        ClassifiedValue::F32(value) => values.push((value, 0.0)),
        ClassifiedValue::ComplexF32(value) => values.push(value),
        _ => return Err(UniformScalarError::Heterogeneous),
    }
    Ok(())
}

pub(super) fn integer_complex(
    prototype: &IntegerStorage,
    real: &mut Vec<IntValue>,
    imaginary: &mut Vec<IntValue>,
    value: ClassifiedValue,
) -> Result<(), UniformScalarError> {
    let ClassifiedValue::IntegerComplex(next_real, next_imaginary) = value else {
        return Err(UniformScalarError::Heterogeneous);
    };
    let real_type = IntegerStorage::from_scalar(next_real.clone()).numeric_dtype();
    let imaginary_type = IntegerStorage::from_scalar(next_imaginary.clone()).numeric_dtype();
    if real_type != prototype.numeric_dtype() || imaginary_type != prototype.numeric_dtype() {
        return Err(UniformScalarError::Heterogeneous);
    }
    real.push(next_real);
    imaginary.push(next_imaginary);
    Ok(())
}

pub(super) fn character(
    values: &mut Vec<char>,
    value: ClassifiedValue,
) -> Result<(), UniformScalarError> {
    let ClassifiedValue::Char(value) = value else {
        return Err(UniformScalarError::Heterogeneous);
    };
    values.push(value);
    Ok(())
}
