use super::ParameterRangeV2;
use crate::model::GeometryContractError;

// Dimensionless serialization-integrity threshold for normalized frame data.
// This is not used for geometric equivalence or any combinatorial decision.
const FRAME_NORMALIZATION_ERROR: f64 = 1.0e-12;

pub(super) fn validate_ranges(
    field: &str,
    ranges: &[ParameterRangeV2; 2],
) -> Result<(), GeometryContractError> {
    validate_range(field, &ranges[0])?;
    validate_range(field, &ranges[1])
}

pub(super) fn validate_range(
    field: &str,
    range: &ParameterRangeV2,
) -> Result<(), GeometryContractError> {
    if !range.start.is_finite() || !range.end.is_finite() || range.start >= range.end {
        return Err(invalid(
            field,
            "parameter range must be finite and increasing",
        ));
    }
    Ok(())
}

pub(super) fn positive(field: &str, value: f64) -> Result<(), GeometryContractError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(field, "must be finite and positive"));
    }
    Ok(())
}

pub(super) fn finite_vector<const N: usize>(
    field: &str,
    value: &[f64; N],
) -> Result<(), GeometryContractError> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(invalid(field, "all components must be finite"));
    }
    Ok(())
}

pub(super) fn nonzero_vector<const N: usize>(
    field: &str,
    value: &[f64; N],
) -> Result<(), GeometryContractError> {
    finite_vector(field, value)?;
    if squared_norm(value) == 0.0 {
        return Err(invalid(field, "must be nonzero"));
    }
    Ok(())
}

pub(super) fn orthonormal_pair<const N: usize>(
    field: &str,
    first: &[f64; N],
    second: &[f64; N],
) -> Result<(), GeometryContractError> {
    finite_vector(field, first)?;
    finite_vector(field, second)?;
    if !approximately_one(squared_norm(first))
        || !approximately_one(squared_norm(second))
        || !approximately_zero(dot(first, second))
    {
        return Err(invalid(field, "axes must form an orthonormal pair"));
    }
    Ok(())
}

pub(super) fn orthonormal_triple(
    field: &str,
    first: &[f64; 3],
    second: &[f64; 3],
    third: &[f64; 3],
) -> Result<(), GeometryContractError> {
    orthonormal_pair(field, first, second)?;
    let cross = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    if !approximately_one(squared_norm(third))
        || !approximately_zero(dot(first, third))
        || !approximately_zero(dot(second, third))
        || !approximately_one(dot(&cross, third))
    {
        return Err(invalid(
            field,
            "axes must form a right-handed orthonormal triple",
        ));
    }
    Ok(())
}

pub(super) fn independent_pair(
    field: &str,
    first: &[f64; 3],
    second: &[f64; 3],
) -> Result<(), GeometryContractError> {
    nonzero_vector(field, first)?;
    nonzero_vector(field, second)?;
    let cross = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    if squared_norm(&cross) == 0.0 {
        return Err(invalid(
            field,
            "parameter axes must be linearly independent",
        ));
    }
    Ok(())
}

fn squared_norm<const N: usize>(value: &[f64; N]) -> f64 {
    dot(value, value)
}

fn dot<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn approximately_one(value: f64) -> bool {
    (value - 1.0).abs() <= FRAME_NORMALIZATION_ERROR
}

fn approximately_zero(value: f64) -> bool {
    value.abs() <= FRAME_NORMALIZATION_ERROR
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
