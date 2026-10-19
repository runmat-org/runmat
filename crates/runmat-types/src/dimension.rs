//! Shared numeric admission for shape and dimension-vector elements.
//! Scalar selectors and dimension vectors retain their distinct established tolerances.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimensionValueError {
    NonFinite,
    Negative,
    Fractional,
    OutsidePlatformRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarDimensionValueError {
    NonFinite,
    Fractional,
    BelowMinimum,
    OutsidePlatformRange,
}

/// Existing scalar-selector admission, shared with structural literal facts.
/// Native integers are decoded separately and never cross this floating path.
pub fn decode_scalar_dimension(
    value: f64,
    allow_zero: bool,
) -> Result<usize, ScalarDimensionValueError> {
    if !value.is_finite() {
        return Err(ScalarDimensionValueError::NonFinite);
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-6 {
        return Err(ScalarDimensionValueError::Fractional);
    }
    let minimum = if allow_zero { 0.0 } else { 1.0 };
    if rounded < minimum {
        return Err(ScalarDimensionValueError::BelowMinimum);
    }
    if !dimension_fits_platform(rounded) {
        return Err(ScalarDimensionValueError::OutsidePlatformRange);
    }
    Ok(rounded as usize)
}

/// Check the platform range without converting through a saturating cast.
/// Binary64 rounds the 64-bit usize maximum upward, but represents the
/// 32-bit maximum exactly; equality is valid only in the latter case.
pub fn dimension_fits_platform(value: f64) -> bool {
    value.is_finite()
        && value >= 0.0
        && (value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64))
}

/// Decode a nonnegative shape dimension using the established absolute
/// machine-epsilon tolerance. Owners enforce their minimum and role rules.
pub fn decode_shape_dimension(value: f64) -> Result<usize, DimensionValueError> {
    if !value.is_finite() {
        return Err(DimensionValueError::NonFinite);
    }
    if value < 0.0 {
        return Err(DimensionValueError::Negative);
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(DimensionValueError::Fractional);
    }
    if !dimension_fits_platform(rounded) {
        return Err(DimensionValueError::OutsidePlatformRange);
    }
    Ok(rounded as usize)
}

/// Established tiling-factor policy. Its relative tolerance is distinct from
/// scalar dimension selectors and from shape-vector elements.
pub fn decode_replication_factor(value: f64) -> Result<usize, DimensionValueError> {
    if !value.is_finite() {
        return Err(DimensionValueError::NonFinite);
    }
    let rounded = value.round();
    let tolerance = (f64::EPSILON * value.abs().max(1.0)).min(1e-9);
    if (rounded - value).abs() > tolerance {
        return Err(DimensionValueError::Fractional);
    }
    if rounded < 0.0 {
        return Err(DimensionValueError::Negative);
    }
    if !dimension_fits_platform(rounded) {
        return Err(DimensionValueError::OutsidePlatformRange);
    }
    Ok(rounded as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_reasons_and_tolerance_are_explicit() {
        assert_eq!(decode_scalar_dimension(1.0 + 1e-7, true), Ok(1));
        assert_eq!(
            decode_scalar_dimension(0.0, false),
            Err(ScalarDimensionValueError::BelowMinimum)
        );
        assert_eq!(
            decode_replication_factor(1_000_000.0 + 1e-10),
            Ok(1_000_000)
        );
        assert_eq!(
            decode_replication_factor(-1.0),
            Err(DimensionValueError::Negative)
        );
        assert_eq!(
            decode_shape_dimension(f64::NAN),
            Err(DimensionValueError::NonFinite)
        );
        assert_eq!(
            decode_shape_dimension(-1.0),
            Err(DimensionValueError::Negative)
        );
        assert_eq!(
            decode_shape_dimension(1.25),
            Err(DimensionValueError::Fractional)
        );
        assert_eq!(decode_shape_dimension(-0.0), Ok(0));
        assert_eq!(decode_shape_dimension(1.0 + f64::EPSILON), Ok(1));
        assert_eq!(
            decode_shape_dimension(1.0 + 2.0 * f64::EPSILON),
            Err(DimensionValueError::Fractional)
        );
    }

    #[test]
    fn maximum_dimension_uses_the_actual_platform_range() {
        let maximum = usize::MAX as f64;
        if usize::BITS < 64 {
            assert_eq!(decode_shape_dimension(maximum), Ok(usize::MAX));
            assert_eq!(
                decode_shape_dimension(maximum + 1.0),
                Err(DimensionValueError::OutsidePlatformRange)
            );
        } else {
            assert_eq!(
                decode_shape_dimension(maximum),
                Err(DimensionValueError::OutsidePlatformRange)
            );
            assert!(decode_shape_dimension(f64::from_bits(maximum.to_bits() - 1)).is_ok());
        }
    }
}
