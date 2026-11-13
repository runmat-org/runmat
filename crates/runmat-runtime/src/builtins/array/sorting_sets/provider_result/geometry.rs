use super::{invalid, Error};
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorStorage, HostTensorOwned};

pub(super) fn cardinality(shape: &[usize], error: Error) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, extent| n.checked_mul(*extent))
        .ok_or_else(|| invalid(error, "provider result cardinality overflow"))
}
pub(super) fn real(value: &HostTensorOwned, error: Error) -> BuiltinResult<()> {
    if value.storage != GpuTensorStorage::Real
        || cardinality(&value.shape, error)? != value.data.len()
    {
        return Err(invalid(
            error,
            "provider result has invalid real storage or cardinality",
        ));
    }
    Ok(())
}
pub(super) fn vector(
    value: &HostTensorOwned,
    length: Option<usize>,
    error: Error,
) -> BuiltinResult<()> {
    real(value, error)?;
    if value.shape.as_slice() != [value.data.len(), 1]
        || length.is_some_and(|length| value.data.len() != length)
    {
        return Err(invalid(
            error,
            "provider index output must be a column of the required length",
        ));
    }
    Ok(())
}
pub(super) fn range(values: &[f64], maximum: usize, zero: bool, error: Error) -> BuiltinResult<()> {
    if maximum as u128 > (1u128 << 53)
        || values.iter().any(|value| {
            !value.is_finite()
                || value.fract() != 0.0
                || *value < (if zero { 0.0 } else { 1.0 })
                || *value > maximum as f64
        })
    {
        return Err(invalid(
            error,
            "provider indices are not exact one-based positions within the input",
        ));
    }
    Ok(())
}
pub(super) fn records(shape: &[usize], rows: bool, error: Error) -> BuiltinResult<usize> {
    if rows {
        Ok(shape.first().copied().unwrap_or(1))
    } else {
        cardinality(shape, error)
    }
}
pub(super) fn row_vector(shape: &[usize]) -> bool {
    shape.len() == 1 || (shape.first() == Some(&1) && shape.iter().skip(2).all(|n| *n == 1))
}
pub(super) fn set_shape(
    value: &HostTensorOwned,
    input: &[usize],
    rows: bool,
    maximum: usize,
    row: bool,
    error: Error,
) -> BuiltinResult<usize> {
    real(value, error)?;
    let count = if rows {
        value.shape.first().copied().unwrap_or(1)
    } else {
        value.data.len()
    };
    let columns = if rows {
        input.get(1).copied().unwrap_or(1)
    } else {
        1
    };
    let expected = if row { [1, count] } else { [count, columns] };
    if value.shape.as_slice() != expected || count > maximum {
        return Err(invalid(
            error,
            "provider set values have invalid orientation, columns or cardinality",
        ));
    }
    Ok(count)
}

/// Validate private producer-owned permutation storage in place, then restore
/// every original index. No workspace alias or count-proportional scratch copy.
pub(super) fn permutation(
    values: &mut [f64],
    shape: &[usize],
    axis: usize,
    error: Error,
) -> BuiltinResult<()> {
    let length = shape.get(axis).copied().unwrap_or(1);
    range(values, length, false, error)?;
    if values.is_empty() {
        return Ok(());
    }
    let stride = cardinality(&shape[..axis.min(shape.len())], error)?;
    let span = stride
        .checked_mul(length)
        .filter(|span| *span > 0)
        .ok_or_else(|| invalid(error, "provider permutation slice overflow"))?;
    if !values.len().is_multiple_of(span) {
        return Err(invalid(
            error,
            "provider permutation slice geometry mismatch",
        ));
    }
    for outer in (0..values.len()).step_by(span) {
        for lane in 0..stride {
            for position in 0..length {
                let offset = outer + lane + position * stride;
                let target = outer + lane + (values[offset].abs() as usize - 1) * stride;
                if values[target].is_sign_negative() {
                    return Err(invalid(error, "provider permutation repeats an index"));
                }
                values[target] = -values[target];
            }
        }
    }
    for value in values {
        *value = value.abs();
    }
    Ok(())
}
