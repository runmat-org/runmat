use runmat_builtins::Value;

use crate::builtins::common::arg_tokens::ArgToken;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

/// Materialise a value so indexing helpers can operate on host tensors.
pub(crate) async fn materialize_value(
    value: Value,
    _builtin: &str,
) -> BuiltinResult<(Value, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
            Ok((Value::Tensor(gathered), true))
        }
        other => Ok((other, false)),
    }
}

/// Parse a MATLAB-style size vector into concrete dimension extents.
pub(crate) async fn parse_dims(value: &Value, builtin: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(_) | Value::Int(_) => parse_scalar_dims(value, builtin).await,
        Value::Tensor(tensor) if tensor.data.len() == 1 => parse_scalar_dims(value, builtin).await,
        Value::GpuTensor(handle) if tensor::element_count(&handle.shape) == 1 => {
            parse_scalar_dims(value, builtin).await
        }
        Value::Tensor(_) | Value::GpuTensor(_) => parse_vector_dims(value, builtin).await,
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

async fn parse_scalar_dims(value: &Value, builtin: &str) -> BuiltinResult<Vec<usize>> {
    let Some(dim) = tensor::dimension_from_value_async(value, builtin, false)
        .await
        .map_err(|_| indexing_error(builtin, "Size arguments must be positive integers."))?
    else {
        return Err(indexing_error(
            builtin,
            "Size vector must be a numeric vector.",
        ));
    };
    Ok(vec![dim])
}

async fn parse_vector_dims(value: &Value, builtin: &str) -> BuiltinResult<Vec<usize>> {
    let dims = tensor::dims_from_value_async(value)
        .await
        .map_err(|_| indexing_error(builtin, "Size arguments must be positive integers."))?
        .ok_or_else(|| indexing_error(builtin, "Size vector must be a row vector."))?;
    if dims.is_empty() {
        return Err(indexing_error(
            builtin,
            "Size vector must have at least one element.",
        ));
    }
    if dims.contains(&0) {
        return Err(indexing_error(
            builtin,
            "Size arguments must be positive integers.",
        ));
    }
    Ok(dims)
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

pub(crate) fn dims_from_tokens(tokens: &[ArgToken]) -> Option<Vec<usize>> {
    let value = tokens.first()?;
    match value {
        ArgToken::Number(num) => coerce_positive_literal(*num).map(|dim| vec![dim]),
        ArgToken::Vector(values) => {
            if values.is_empty() {
                return None;
            }
            let mut dims = Vec::with_capacity(values.len());
            for value in values {
                let dim = match value {
                    ArgToken::Number(num) => coerce_positive_literal(*num)?,
                    _ => return None,
                };
                dims.push(dim);
            }
            Some(dims)
        }
        _ => None,
    }
}

fn coerce_positive_literal(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < 1.0 {
        return None;
    }
    Some(rounded as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_from_tokens_accepts_scalar() {
        let dims = dims_from_tokens(&[ArgToken::Number(3.0)]);
        assert_eq!(dims, Some(vec![3]));
    }

    #[test]
    fn dims_from_tokens_accepts_vector() {
        let dims = dims_from_tokens(&[ArgToken::Vector(vec![
            ArgToken::Number(2.0),
            ArgToken::Number(4.0),
        ])]);
        assert_eq!(dims, Some(vec![2, 4]));
    }

    #[test]
    fn dims_from_tokens_rejects_non_numeric() {
        let dims = dims_from_tokens(&[ArgToken::Vector(vec![ArgToken::String(
            "bad".to_string(),
        )])]);
        assert_eq!(dims, None);
    }
}
