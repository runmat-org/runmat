use super::metadata::MISSING_TEXT;
use super::standardize::numeric_tensor;
use super::{internal_error, invalid_argument, CharArray, IntValue, Tensor, Value};
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

pub(super) fn is_numeric_data_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    )
}

pub(super) fn pairwise_nan_min(left: Value, right: Value) -> BuiltinResult<Value> {
    if crate::builtins::math::reduction::integer_native::value_has_integer_storage(&left)
        || crate::builtins::math::reduction::integer_native::value_has_integer_storage(&right)
    {
        if let Ok(Some(evaluation)) =
            crate::builtins::math::reduction::integer_native::elementwise_value_extrema(
                &left,
                &right,
                crate::builtins::math::reduction::integer_native::ExtremaDirection::Min,
                crate::builtins::math::reduction::integer_native::ExtremaComparison::Natural,
                false,
            )
        {
            return Ok(evaluation.values);
        }
    }
    let left_scalar = numeric_scalar(&left, "nanmin left").ok();
    let right_scalar = numeric_scalar(&right, "nanmin right").ok();
    if let (Some(a), Some(b)) = (left_scalar, right_scalar) {
        return Ok(Value::Num(nan_min_pair(a, b)));
    }
    let left = numeric_tensor(left, "nanmin left")?;
    let right = numeric_tensor(right, "nanmin right")?;
    let (data, shape, dtype) = broadcast_pairwise_numeric(&left, &right, nan_min_pair)?;
    Tensor::new_with_dtype(data, shape, dtype)
        .map(Value::Tensor)
        .map_err(internal_error)
}

pub(super) fn broadcast_pairwise_numeric(
    left: &Tensor,
    right: &Tensor,
    op: impl Fn(f64, f64) -> f64,
) -> BuiltinResult<(Vec<f64>, Vec<usize>, runmat_value::NumericDType)> {
    let left_len = tensor_utils::tensor_element_len(left);
    let right_len = tensor_utils::tensor_element_len(right);
    if left_len == right_len && left.shape == right.shape {
        let left_values = tensor_utils::tensor_values_f64_cow(left);
        let right_values = tensor_utils::tensor_values_f64_cow(right);
        let data = left_values
            .iter()
            .zip(right_values.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();
        return Ok((data, left.shape.clone(), left.numeric_dtype()));
    }
    if left_len == 1 {
        let left_values = tensor_utils::tensor_values_f64_cow(left);
        let right_values = tensor_utils::tensor_values_f64_cow(right);
        let data = right_values
            .iter()
            .map(|b| op(left_values[0], *b))
            .collect();
        return Ok((data, right.shape.clone(), right.numeric_dtype()));
    }
    if right_len == 1 {
        let left_values = tensor_utils::tensor_values_f64_cow(left);
        let right_values = tensor_utils::tensor_values_f64_cow(right);
        let data = left_values
            .iter()
            .map(|a| op(*a, right_values[0]))
            .collect();
        return Ok((data, left.shape.clone(), left.numeric_dtype()));
    }
    Err(invalid_argument(
        "nanmin: pairwise inputs must have the same shape or one scalar input",
    ))
}

pub(super) fn nan_min_pair(a: f64, b: f64) -> f64 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => f64::NAN,
        (true, false) => b,
        (false, true) => a,
        (false, false) => a.min(b),
    }
}

pub(super) fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(f64::from(*b)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        other => Err(invalid_argument(format!(
            "{context}: expected numeric scalar, got {other:?}"
        ))),
    }
}

pub(super) fn first_nonsingleton_dim(rows: usize, cols: usize) -> usize {
    if rows > 1 {
        1
    } else if cols > 1 {
        2
    } else {
        1
    }
}

pub(super) fn validate_matrix_dim(dim: usize, context: &str) -> BuiltinResult<()> {
    if dim == 1 || dim == 2 {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "{context}: dimension must be 1 or 2"
        )))
    }
}

pub(super) fn scalar_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(integer) => return integer_size_to_usize(integer, context),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let integer = storage.value_at(0).ok_or_else(|| {
                    internal_error(format!("{context}: integer scalar storage length mismatch"))
                })?;
                return integer_size_to_usize(&integer, context);
            }
        }
        _ => {}
    }
    let n = numeric_scalar(value, context)?;
    numeric_size_to_usize(n, context)
}

pub(super) fn integer_size_to_usize(value: &IntValue, context: &str) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        invalid_argument(format!("{context}: expected nonnegative platform integer"))
    })
}

pub(super) fn numeric_size_to_usize(n: f64, context: &str) -> BuiltinResult<usize> {
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return Err(invalid_argument(format!(
            "{context}: expected nonnegative integer"
        )));
    }
    if n > usize::MAX as f64 {
        return Err(invalid_argument(format!("{context}: integer too large")));
    }
    if usize::BITS == 64 && n == usize::MAX as f64 {
        return Err(invalid_argument(format!("{context}: integer too large")));
    }
    Ok(n as usize)
}

pub(super) fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
}

pub(super) fn char_rows(array: &CharArray) -> Vec<String> {
    let mut out = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        out.push(array.data[start..start + array.cols].iter().collect());
    }
    out
}

pub(super) fn is_missing_text(text: &str) -> bool {
    text.eq_ignore_ascii_case(MISSING_TEXT)
}
