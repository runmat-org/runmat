//! Borrowed scalar/count controls; no temporary numeric vector.
use super::*;

fn numeric_at(value: &Value, index: usize) -> Option<NumericScalar> {
    match value {
        Value::Tensor(tensor) => tensor.numeric_value_at(index),
        Value::LogicalArray(logical) => logical
            .data
            .get(index)
            .map(|bit| NumericScalar::F64(if *bit == 0 { 0.0 } else { 1.0 })),
        other if index == 0 => scalar_numeric(other),
        _ => None,
    }
}

pub(in crate::builtins::math::reduction::moving) fn parse_window(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<WindowSpec> {
    let len = match value {
        Value::Tensor(tensor) => {
            validate_vector_shape(name, tensor.shape(), tensor.len())?;
            tensor.len()
        }
        Value::LogicalArray(logical) => {
            validate_vector_shape(name, &logical.shape, logical.data.len())?;
            logical.data.len()
        }
        _ if scalar_numeric(value).is_some() => 1,
        _ => {
            return Err(invalid_argument(
                name,
                format!("{name}: window must be numeric"),
            ))
        }
    };
    let parse = |index, positive, what| -> BuiltinResult<usize> {
        let scalar = numeric_at(value, index)
            .ok_or_else(|| internal(name, format!("{name}: missing numeric window element")))?;
        if let Some(integer) = scalar.into_int_value() {
            if positive {
                positive_integer_value(name, &integer, what)
            } else {
                nonnegative_integer_value(name, &integer, what)
            }
        } else if positive {
            positive_integer(name, scalar.materialize_f64(), what)
        } else {
            nonnegative_integer(name, scalar.materialize_f64(), what)
        }
    };
    let (before, after) = match len {
        1 => {
            let k = parse(0, true, "window length")?;
            (k / 2, (k - 1) / 2)
        }
        2 => (
            parse(0, false, "window backward length")?,
            parse(1, false, "window forward length")?,
        ),
        _ => {
            return Err(invalid_argument(
                name,
                format!("{name}: window must be a scalar or a two-element vector"),
            ))
        }
    };
    let window = WindowSpec::Counts { before, after };
    window
        .checked_count_len(name)?
        .ok_or_else(|| internal(name, format!("{name}: invalid window")))?;
    Ok(window)
}

/// Window controls are scalar or two-dimensional vectors, not flattened
/// matrices/pages. Ignore only trailing singleton axes, as MATLAB does.
pub(super) fn validate_vector_shape(
    name: &'static str,
    shape: &[usize],
    len: usize,
) -> BuiltinResult<()> {
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let valid = (rows == 1 || cols == 1)
        && shape.iter().skip(2).all(|extent| *extent == 1)
        && shape
            .iter()
            .try_fold(1usize, |n, extent| n.checked_mul(*extent))
            == Some(len);
    if valid {
        Ok(())
    } else {
        Err(invalid_argument(
            name,
            format!(
                "{name}: window and SamplePoints controls must be scalars or row/column vectors"
            ),
        ))
    }
}

pub(in crate::builtins::math::reduction::moving) fn parse_variance_normalization(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Option<VarianceNormalization>> {
    if matches!(value, Value::Tensor(tensor) if tensor.is_empty())
        || matches!(value, Value::LogicalArray(logical) if logical.data.is_empty())
    {
        return Ok(Some(VarianceNormalization::Sample));
    }
    let Some(scalar) = scalar_numeric(value) else {
        return if matches!(value, Value::Tensor(_) | Value::LogicalArray(_)) {
            Err(invalid_argument(
                name,
                format!("{name}: normalization flag must be 0 or 1"),
            ))
        } else {
            Ok(None)
        };
    };
    let normalization = if let Some(integer) = scalar.into_int_value() {
        match integer.try_to_i64() {
            Some(0) => Some(VarianceNormalization::Sample),
            Some(1) => Some(VarianceNormalization::Population),
            _ => None,
        }
    } else {
        let raw = scalar.materialize_f64();
        if raw.abs() <= f64::EPSILON {
            Some(VarianceNormalization::Sample)
        } else if (raw - 1.0).abs() <= f64::EPSILON {
            Some(VarianceNormalization::Population)
        } else {
            None
        }
    };
    normalization
        .map(Some)
        .ok_or_else(|| invalid_argument(name, format!("{name}: normalization flag must be 0 or 1")))
}
