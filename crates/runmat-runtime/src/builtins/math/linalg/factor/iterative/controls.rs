//! Structural iterative-solver controls; callers own their error policy.

use runmat_value::{NumericDType, Value};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg::factor) enum GeometryError {
    UnsupportedRightHandSide,
    InvalidRightHandSide,
    UnsupportedCoefficient,
    UnsupportedNativeClass,
}

pub(in crate::builtins::math::linalg::factor) fn rhs_geometry(
    value: &Value,
) -> Result<(usize, NumericDType), GeometryError> {
    if matches!(value, Value::Num(_) | Value::Complex(_, _)) {
        return Ok((1, NumericDType::F64));
    }
    let (shape, dtype, native) = match value {
        Value::Tensor(tensor) => (
            tensor.shape(),
            tensor.numeric_dtype(),
            tensor.as_f32_slice().is_some() || tensor.as_f64_slice().is_some(),
        ),
        Value::ComplexTensor(tensor) => (
            tensor.shape(),
            tensor.numeric_dtype(),
            tensor.as_f32_slice().is_some() || tensor.as_f64_slice().is_some(),
        ),
        _ => return Err(GeometryError::UnsupportedRightHandSide),
    };
    let rows = shape.first().copied().unwrap_or(0);
    if !native || rows == 0 || shape.get(1) != Some(&1) || shape.iter().skip(2).any(|&n| n != 1) {
        return Err(GeometryError::InvalidRightHandSide);
    }
    Ok((rows, dtype))
}

pub(in crate::builtins::math::linalg::factor) fn coefficient_geometry(
    value: &Value,
) -> Result<(usize, usize, NumericDType), GeometryError> {
    if matches!(value, Value::Num(_) | Value::Complex(_, _)) {
        return Ok((1, 1, NumericDType::F64));
    }
    let (rows, cols, single, double) = match value {
        Value::Tensor(matrix) => (
            matrix.rows(),
            matrix.cols(),
            matrix.as_f32_slice().is_some(),
            matrix.as_f64_slice().is_some(),
        ),
        Value::ComplexTensor(matrix) => (
            matrix.rows(),
            matrix.cols(),
            matrix.as_f32_slice().is_some(),
            matrix.as_f64_slice().is_some(),
        ),
        Value::SparseTensor(matrix) => (
            matrix.rows,
            matrix.cols,
            matrix.as_f32_slice().is_some() || matrix.as_complex_f32_slice().is_some(),
            matrix.as_f64_slice().is_some() || matrix.as_complex_f64_slice().is_some(),
        ),
        _ => return Err(GeometryError::UnsupportedCoefficient),
    };
    let dtype = match (single, double) {
        (true, false) => NumericDType::F32,
        (false, true) => NumericDType::F64,
        _ => return Err(GeometryError::UnsupportedNativeClass),
    };
    Ok((rows, cols, dtype))
}

pub(in crate::builtins::math::linalg::factor) fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

pub(in crate::builtins::math::linalg::factor) fn is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.len() == 0,
        Value::ComplexTensor(tensor) => tensor.len() == 0,
        Value::SparseTensor(tensor) => tensor.nnz() == 0 && tensor.rows == 0 && tensor.cols == 0,
        _ => false,
    }
}

pub(in crate::builtins::math::linalg::factor) fn real_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) if crate::builtins::common::validation::integer_is_exact_f64(value) => {
            Some(value.to_f64())
        }
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(values) = tensor.as_f64_slice() {
                Some(values[0])
            } else if let Some(values) = tensor.as_f32_slice() {
                Some(f64::from(values[0]))
            } else {
                let value = tensor.integer_storage()?.value_at(0)?;
                crate::builtins::common::validation::integer_is_exact_f64(&value)
                    .then(|| value.to_f64())
            }
        }
        _ => None,
    }
}

pub(in crate::builtins::math::linalg::factor) fn positive_count(value: &Value) -> Option<usize> {
    if let Value::Int(value) = value {
        return value.try_to_usize().filter(|&count| count > 0);
    }
    if let Value::Tensor(tensor) = value {
        if tensor.len() == 1 {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(0)?
                    .try_to_usize()
                    .filter(|&count| count > 0);
            }
        }
    }
    let value = real_scalar(value)?;
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return None;
    }
    Some(value as usize)
}
