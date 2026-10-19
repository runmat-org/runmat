//! Finite complex coefficient ratios at the transfer-function boundary.
//!
//! `Complex64` division forms a squared denominator norm. That intermediate
//! can overflow or underflow even when the quotient is representable. Scale
//! both finite operands independently before the complex division; only the
//! final real components are rescaled to the original magnitude.

use num_complex::Complex64;

fn finite_scale(value: Complex64) -> Option<f64> {
    if value.re.is_finite() && value.im.is_finite() {
        Some(value.re.abs().max(value.im.abs()))
    } else {
        None
    }
}

/// A ratio of finite complex values with a nonzero denominator. This does not
/// claim to resolve cancellation or overflow in polynomial evaluation itself.
pub(super) fn finite_complex_ratio(numerator: Complex64, denominator: Complex64) -> Complex64 {
    let (Some(numerator_scale), Some(denominator_scale)) =
        (finite_scale(numerator), finite_scale(denominator))
    else {
        return numerator / denominator;
    };
    if denominator_scale == 0.0 {
        return numerator / denominator;
    }
    if numerator_scale == 0.0 {
        return Complex64::new(0.0, 0.0);
    }

    let unit_numerator = numerator / numerator_scale;
    let unit_denominator = denominator / denominator_scale;
    let unit_ratio = unit_numerator / unit_denominator;
    let magnitude = numerator_scale / denominator_scale;
    let component = |unit: f64| {
        if unit == 0.0 {
            unit
        } else if magnitude.is_finite() && magnitude != 0.0 {
            unit * magnitude
        } else {
            // Avoid a premature infinity or zero in the scale quotient. When
            // that quotient overflows, denominator_scale < 1; if the first
            // product overflows too, the represented result must overflow.
            (unit * numerator_scale) / denominator_scale
        }
    };
    Complex64::new(component(unit_ratio.re), component(unit_ratio.im))
}

/// The direction alone is sufficient at an exact pole. Discard the magnitude
/// *before* division so a tiny/huge ratio cannot underflow to zero and erase
/// an otherwise nonzero signed or complex infinity axis.
pub(super) fn finite_complex_ratio_direction(
    numerator: Complex64,
    denominator: Complex64,
) -> Complex64 {
    let (Some(numerator_scale), Some(denominator_scale)) =
        (finite_scale(numerator), finite_scale(denominator))
    else {
        return numerator / denominator;
    };
    if numerator_scale == 0.0 || denominator_scale == 0.0 {
        return numerator / denominator;
    }
    (numerator / numerator_scale) / (denominator / denominator_scale)
}
