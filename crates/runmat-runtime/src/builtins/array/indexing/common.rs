use runmat_builtins::{Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

/// Materialise a value so indexing helpers can operate on host tensors.
pub(crate) fn materialize_value(value: Value, _builtin: &str) -> BuiltinResult<(Value, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor(&handle)?;
            Ok((Value::Tensor(gathered), true))
        }
        other => Ok((other, false)),
    }
}

/// Parse a MATLAB-style size vector into concrete dimension extents.
pub(crate) fn parse_dims(value: &Value, builtin: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => dims_from_tensor(tensor, builtin),
        Value::Num(n) => Ok(vec![coerce_positive_int(*n, builtin)?]),
        Value::Int(i) => Ok(vec![coerce_positive_int(i.to_f64(), builtin)?]),
        Value::Cell(ca) => {
            if ca.data.is_empty() {
                return Err(indexing_error(
                    builtin,
                    "Size vector must have at least one element.",
                ));
            }
            let mut dims = Vec::with_capacity(ca.data.len());
            for cell in &ca.data {
                let coerced = match &**cell {
                    Value::Num(n) => coerce_positive_int(*n, builtin)?,
                    Value::Int(i) => coerce_positive_int(i.to_f64(), builtin)?,
                    _ => {
                        return Err(indexing_error(
                            builtin,
                            "Size vector must contain numeric values.",
                        ))
                    }
                };
                dims.push(coerced);
            }
            Ok(dims)
        }
        _ => Err(indexing_error(
            builtin,
            "Size vector must be a numeric vector.",
        )),
    }
}

fn dims_from_tensor(tensor: &Tensor, builtin: &str) -> BuiltinResult<Vec<usize>> {
    if !is_vector_shape(&tensor.shape) {
        return Err(indexing_error(builtin, "Size vector must be a row vector."));
    }
    if tensor.data.is_empty() {
        return Err(indexing_error(
            builtin,
            "Size vector must have at least one element.",
        ));
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        dims.push(coerce_positive_int(value, builtin)?);
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
pub(crate) fn coerce_positive_int(value: f64, builtin: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(indexing_error(
            builtin,
            "Size arguments must be positive integers.",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(indexing_error(
            builtin,
            "Size arguments must be positive integers.",
        ));
    }
    if rounded < 1.0 {
        return Err(indexing_error(
            builtin,
            "Size arguments must be positive integers.",
        ));
    }
    Ok(rounded as usize)
}

/// Build column-major strides for the supplied dimensions, checking overflow.
pub(crate) fn build_strides(dims: &[usize], builtin: &str) -> BuiltinResult<Vec<usize>> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.checked_mul(dim).ok_or_else(|| {
            indexing_error(
                builtin,
                "Size vector elements overflow the maximum supported size.",
            )
        })?;
    }
    Ok(strides)
}

/// Compute the total number of elements implied by the size vector.
pub(crate) fn total_elements(dims: &[usize], builtin: &str) -> BuiltinResult<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            indexing_error(
                builtin,
                "Size vector elements overflow the maximum supported size.",
            )
        })
    })
}

fn indexing_error(builtin: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}
