use runmat_builtins::{Tensor, Value};

use crate::builtins::common::gpu_helpers;

/// Materialise a value so indexing helpers can operate on host tensors.
pub(crate) fn materialize_value(value: Value) -> Result<(Value, bool), String> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor(&handle)?;
            Ok((Value::Tensor(gathered), true))
        }
        other => Ok((other, false)),
    }
}

/// Parse a MATLAB-style size vector into concrete dimension extents.
pub(crate) fn parse_dims(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(tensor) => dims_from_tensor(tensor),
        Value::Num(n) => Ok(vec![coerce_positive_int(*n)?]),
        Value::Int(i) => Ok(vec![coerce_positive_int(i.to_f64())?]),
        Value::Cell(ca) => {
            if ca.data.is_empty() {
                return Err("Size vector must have at least one element.".to_string());
            }
            let mut dims = Vec::with_capacity(ca.data.len());
            for cell in &ca.data {
                let coerced = match &**cell {
                    Value::Num(n) => coerce_positive_int(*n)?,
                    Value::Int(i) => coerce_positive_int(i.to_f64())?,
                    _ => return Err("Size vector must contain numeric values.".to_string()),
                };
                dims.push(coerced);
            }
            Ok(dims)
        }
        _ => Err("Size vector must be a numeric vector.".to_string()),
    }
}

fn dims_from_tensor(tensor: &Tensor) -> Result<Vec<usize>, String> {
    if !is_vector_shape(&tensor.shape) {
        return Err("Size vector must be a row vector.".to_string());
    }
    if tensor.data.is_empty() {
        return Err("Size vector must have at least one element.".to_string());
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        dims.push(coerce_positive_int(value)?);
    }
    Ok(dims)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

/// Coerce a floating-point value into a strictly positive integer.
pub(crate) fn coerce_positive_int(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("Size arguments must be positive integers.".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("Size arguments must be positive integers.".to_string());
    }
    if rounded < 1.0 {
        return Err("Size arguments must be positive integers.".to_string());
    }
    Ok(rounded as usize)
}

/// Build column-major strides for the supplied dimensions, checking overflow.
pub(crate) fn build_strides(dims: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.checked_mul(dim).ok_or_else(|| {
            "Size vector elements overflow the maximum supported size.".to_string()
        })?;
    }
    Ok(strides)
}

/// Compute the total number of elements implied by the size vector.
pub(crate) fn total_elements(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| "Size vector elements overflow the maximum supported size.".to_string())
    })
}
