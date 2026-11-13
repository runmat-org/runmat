//! Numeric content predicates with exact typed storage.

use super::{
    complex_real_values_all, exact_integer_values_all, handle_integer_type, handle_is_logical,
    handle_storage, host_value, int_f64_matches, int_is_negative, int_is_nonnegative,
    int_is_nonpositive, int_is_positive, numeric_values_all, tensor, value_is_empty,
    value_is_ordered_against_zero_async, BuiltinResult, GpuTensorStorage, IntValue,
    IntegerComparisonOp, Ordering, RangeInclusivity, Value,
};

pub fn value_is_finite(value: &Value) -> bool {
    match value {
        Value::Num(v) => v.is_finite(),
        Value::Int(_) | Value::Bool(_) => true,
        Value::Complex(re, im) => re.is_finite() && im.is_finite(),
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t)
            .iter()
            .all(|v| v.is_finite()),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) if t.is_complex() => t
            .materialize_complex_f64()
            .expect("complex sparse storage")
            .iter()
            .all(|value| value.0.is_finite() && value.1.is_finite()),
        Value::SparseTensor(t) => t.materialize_f64().iter().all(|v| v.is_finite()),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite()),
        Value::LogicalArray(_) | Value::CharArray(_) => true,
        Value::GpuTensor(_) => true,
        _ => false,
    }
}

pub fn value_is_real(value: &Value) -> bool {
    if value_is_empty(value) {
        return true;
    }
    match value {
        Value::Complex(_, im) => *im == 0.0,
        Value::ComplexTensor(t) if t.integer_storage().is_some() => t
            .integer_storage()
            .as_ref()
            .expect("checked integer complex storage")
            .imag()
            .exact_values()
            .iter()
            .all(IntValue::is_zero),
        Value::ComplexTensor(t) => t.materialize_f64().iter().all(|(_, im)| *im == 0.0),
        Value::SparseTensor(t) if t.is_complex() => t
            .materialize_complex_f64()
            .expect("complex sparse storage")
            .iter()
            .all(|value| value.1 == 0.0),
        Value::GpuTensor(handle) => handle_storage(handle) == GpuTensorStorage::Real,
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_) => true,
        _ => false,
    }
}

pub async fn value_is_real_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
    {
        return Ok(value_is_real(&host_value(value).await?));
    }
    Ok(value_is_real(value))
}

pub fn value_is_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Bool(_) | Value::LogicalArray(_) | Value::CharArray(_) => true,
        Value::Num(v) => v.is_finite() && v.fract() == 0.0,
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t)
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0.0),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) if t.is_complex() => t
            .materialize_complex_f64()
            .expect("complex sparse storage")
            .iter()
            .all(|value| {
                value.0.is_finite()
                    && value.0.fract() == 0.0
                    && value.1.is_finite()
                    && value.1.fract() == 0.0
            }),
        Value::SparseTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|v| v.is_finite() && v.fract() == 0.0),
        Value::Complex(re, im) => {
            re.is_finite() && re.fract() == 0.0 && im.is_finite() && im.fract() == 0.0
        }
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t.materialize_f64().iter().all(|(re, im)| {
            re.is_finite() && re.fract() == 0.0 && im.is_finite() && im.fract() == 0.0
        }),
        Value::GpuTensor(handle) => {
            handle_is_logical(handle) || handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

pub fn value_is_non_nan(value: &Value) -> bool {
    match value {
        Value::Num(v) => !v.is_nan(),
        Value::Complex(re, im) => !re.is_nan() && !im.is_nan(),
        Value::Tensor(t) if t.integer_storage().is_some() => true,
        Value::Tensor(t) => tensor::tensor_values_f64_cow(t).iter().all(|v| !v.is_nan()),
        Value::SparseTensor(t) if t.integer_storage().is_some() => true,
        Value::SparseTensor(t) if t.is_complex() => t
            .materialize_complex_f64()
            .expect("complex sparse storage")
            .iter()
            .all(|value| !value.0.is_nan() && !value.1.is_nan()),
        Value::SparseTensor(t) => t.materialize_f64().iter().all(|v| !v.is_nan()),
        Value::ComplexTensor(t) if t.integer_storage().is_some() => true,
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| !re.is_nan() && !im.is_nan()),
        Value::Cell(c) => c.data().iter().all(value_is_non_nan),
        _ => true,
    }
}

pub fn value_is_nonmissing(value: &Value) -> bool {
    value_is_non_nan(value)
}

pub fn value_is_positive(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v > 0.0, int_is_positive) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_positive) {
        return result;
    }
    numeric_values_all(value, |v| v > 0.0)
}

pub fn value_is_negative(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v < 0.0, int_is_negative) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_negative) {
        return result;
    }
    numeric_values_all(value, |v| v < 0.0)
}

pub fn value_is_nonnegative(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v >= 0.0, int_is_nonnegative) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_nonnegative) {
        return result;
    }
    numeric_values_all(value, |v| v >= 0.0)
}

pub fn value_is_nonpositive(value: &Value) -> bool {
    if let Some(result) = complex_real_values_all(value, |v| v <= 0.0, int_is_nonpositive) {
        return result;
    }
    if let Some(result) = exact_integer_values_all(value, int_is_nonpositive) {
        return result;
    }
    numeric_values_all(value, |v| v <= 0.0)
}

pub fn value_is_nonzero(value: &Value) -> bool {
    match value {
        Value::Complex(re, im) => *re != 0.0 || *im != 0.0,
        Value::ComplexTensor(t) if t.integer_storage().is_some() => {
            let integer_data = t.integer_storage().expect("checked integer data");
            (0..integer_data.len()).all(|index| integer_data.is_nonzero_at(index).unwrap_or(false))
        }
        Value::ComplexTensor(t) => t
            .materialize_f64()
            .iter()
            .all(|(re, im)| *re != 0.0 || *im != 0.0),
        _ => {
            if let Some(result) = exact_integer_values_all(value, |integer| !integer.is_zero()) {
                return result;
            }
            numeric_values_all(value, |v| v != 0.0)
        }
    }
}

pub fn value_is_greater_than_or_equal(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering >= Ordering::Equal)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v >= threshold)
}

pub fn value_is_less_than_or_equal(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering <= Ordering::Equal)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v <= threshold)
}

pub fn value_is_greater_than(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering == Ordering::Greater)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v > threshold)
}

pub fn value_is_less_than(value: &Value, threshold: f64) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        int_f64_matches(integer, threshold, |ordering| ordering == Ordering::Less)
    }) {
        return result;
    }
    numeric_values_all(value, |v| v.is_finite() && v < threshold)
}

pub fn value_is_in_range(
    value: &Value,
    lower: f64,
    upper: f64,
    inclusivity: RangeInclusivity,
) -> bool {
    if let Some(result) = exact_integer_values_all(value, |integer| {
        let lower_ok = int_f64_matches(integer, lower, |ordering| {
            if inclusivity.lower {
                ordering >= Ordering::Equal
            } else {
                ordering == Ordering::Greater
            }
        });
        let upper_ok = int_f64_matches(integer, upper, |ordering| {
            if inclusivity.upper {
                ordering <= Ordering::Equal
            } else {
                ordering == Ordering::Less
            }
        });
        lower_ok && upper_ok
    }) {
        return result;
    }
    numeric_values_all(value, |v| {
        v.is_finite()
            && if inclusivity.lower {
                v >= lower
            } else {
                v > lower
            }
            && if inclusivity.upper {
                v <= upper
            } else {
                v < upper
            }
    })
}

pub async fn value_is_finite_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
    {
        return Ok(true);
    }
    let value = host_value(value).await?;
    Ok(value_is_finite(&value))
}

pub async fn value_is_integer_async(value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
    {
        return Ok(true);
    }
    let value = host_value(value).await?;
    Ok(value_is_integer(&value))
}

pub async fn value_is_non_nan_async(value: &Value) -> BuiltinResult<bool> {
    for leaf in super::cell_leaves::CellLeaves::new(value) {
        super::validate_resident_metadata(leaf)?;
        if matches!(leaf, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
        {
            continue;
        }
        if !value_is_non_nan(&host_value(leaf).await?) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub async fn value_is_nonmissing_async(value: &Value) -> BuiltinResult<bool> {
    value_is_non_nan_async(value).await
}

pub async fn value_is_positive_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Gt).await
}

pub async fn value_is_negative_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Lt).await
}

pub async fn value_is_nonnegative_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Ge).await
}

pub async fn value_is_nonpositive_async(value: &Value) -> BuiltinResult<bool> {
    value_is_ordered_against_zero_async(value, IntegerComparisonOp::Le).await
}

pub async fn value_is_nonzero_async(value: &Value) -> BuiltinResult<bool> {
    let value = host_value(value).await?;
    Ok(value_is_nonzero(&value))
}
