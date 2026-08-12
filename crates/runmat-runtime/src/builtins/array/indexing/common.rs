use runmat_value::{IntValue, Value};

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
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            parse_scalar_dims(value, builtin).await
        }
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
                let coerced = match cell {
                    Value::Num(n) => coerce_positive_int(*n, builtin)?,
                    Value::Int(i) => coerce_positive_integer(i, builtin)?,
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
    if !fits_positive_platform_index(rounded) {
        return Err(indexing_error(
            builtin,
            "Size arguments exceed the maximum supported size.",
        ));
    }
    Ok(rounded as usize)
}

fn coerce_positive_integer(value: &IntValue, builtin: &str) -> BuiltinResult<usize> {
    value.try_to_usize().filter(|&dim| dim >= 1).ok_or_else(|| {
        indexing_error(
            builtin,
            "Size arguments must be positive integers within the supported range.",
        )
    })
}

pub(crate) fn dims_from_tokens(tokens: &[ArgToken]) -> Option<Vec<usize>> {
    let value = tokens.first()?;
    match value {
        ArgToken::Number(num) => coerce_positive_literal(*num).map(|dim| vec![dim]),
        ArgToken::Integer(value) => coerce_positive_integer_literal(value).map(|dim| vec![dim]),
        ArgToken::Vector(values) => {
            if values.is_empty() {
                return None;
            }
            let mut dims = Vec::with_capacity(values.len());
            for value in values {
                let dim = match value {
                    ArgToken::Number(num) => coerce_positive_literal(*num)?,
                    ArgToken::Integer(value) => coerce_positive_integer_literal(value)?,
                    _ => return None,
                };
                dims.push(dim);
            }
            Some(dims)
        }
        _ => None,
    }
}

fn coerce_positive_integer_literal(value: &IntValue) -> Option<usize> {
    value.try_to_usize().filter(|&dim| dim >= 1)
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
    if !fits_positive_platform_index(rounded) {
        return None;
    }
    Some(rounded as usize)
}

pub(crate) fn fits_positive_platform_index(value: f64) -> bool {
    if value > usize::MAX.saturating_sub(1) as f64 {
        return false;
    }
    let parsed = value as usize;
    parsed as f64 == value && parsed != usize::MAX
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
    use futures::executor::block_on;
    use runmat_value::{CellArray, IntegerStorage, Tensor};

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
        let dims = dims_from_tokens(&[ArgToken::Vector(vec![ArgToken::String("bad".to_string())])]);
        assert_eq!(dims, None);
    }

    #[test]
    fn parse_dims_preserves_typed_cell_dimensions_exactly() {
        let dims = block_on(parse_dims(
            &Value::Cell(
                CellArray::new(
                    vec![
                        Value::Int(IntValue::U8(2)),
                        Value::Int(IntValue::U64(9_007_199_254_740_993)),
                    ],
                    1,
                    2,
                )
                .expect("cell"),
            ),
            "zeros",
        ))
        .expect("dimensions");

        assert_eq!(dims, vec![2, 9_007_199_254_740_993]);
        assert!(block_on(parse_dims(
            &Value::Cell(CellArray::new(vec![Value::Int(IntValue::I8(-1))], 1, 1).expect("cell"),),
            "zeros",
        ))
        .is_err());
    }

    #[test]
    fn parse_dims_reads_typed_integer_tensor_storage_exactly() {
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar dim");
        let dims = block_on(parse_dims(&Value::Tensor(scalar), "zeros")).expect("scalar dims");
        assert_eq!(dims, vec![7]);

        let vector =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("vector dims");
        let dims = block_on(parse_dims(&Value::Tensor(vector), "zeros")).expect("vector dims");
        assert_eq!(dims, vec![2, 3]);
    }

    #[test]
    fn numeric_dimension_coercion_rejects_out_of_range_float_values() {
        assert!(coerce_positive_int(usize::MAX as f64, "zeros").is_err());
        assert!(dims_from_tokens(&[ArgToken::Number(usize::MAX as f64)]).is_none());
        assert!(dims_from_tokens(&[ArgToken::Integer(IntValue::I64(-1))]).is_none());
        #[cfg(target_pointer_width = "32")]
        assert!(dims_from_tokens(&[ArgToken::Integer(IntValue::U64(u64::MAX))]).is_none());
        assert_eq!(
            dims_from_tokens(&[ArgToken::Vector(vec![
                ArgToken::Integer(IntValue::U16(2)),
                ArgToken::Integer(IntValue::U32(4)),
            ])]),
            Some(vec![2, 4])
        );
    }
}
