//! Finite-interval geometry shared by the scalar quadrature traversals.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuadratureNumericalError {
    NonrepresentableEstimate,
}

pub(crate) fn midpoint(a: f64, b: f64) -> f64 {
    // Opposite signs make the sum safe; same signs make the difference safe.
    // This also retains a subnormal midpoint where halving both operands first
    // would round them away.
    if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
        (a + b) * 0.5
    } else {
        a + (b - a) * 0.5
    }
}

pub(crate) fn simpson(
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
) -> Result<f64, QuadratureNumericalError> {
    // The convex average cannot overflow for finite samples. Multiplying the
    // interval width by the unscaled weighted sum can overflow prematurely.
    let average = (fa / 6.0 + fb / 6.0) + (fm * (2.0 / 3.0));
    if average == 0.0 {
        return Ok(0.0);
    }

    let width = b - a;
    let (width_fraction, width_exponent) = if width.is_finite() {
        libm::frexp(width)
    } else {
        // Only opposite-sign finite endpoints can overflow their difference.
        let half_width = b * 0.5 - a * 0.5;
        let (fraction, exponent) = libm::frexp(half_width);
        (fraction, exponent + 1)
    };
    let (average_fraction, average_exponent) = libm::frexp(average);
    let value = libm::ldexp(
        width_fraction * average_fraction,
        width_exponent + average_exponent,
    );
    if value.is_finite() {
        Ok(value)
    } else {
        Err(QuadratureNumericalError::NonrepresentableEstimate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_extreme_midpoints_do_not_overflow() {
        assert_eq!(midpoint(-f64::MAX, f64::MAX), 0.0);
        assert_eq!(midpoint(0.0, f64::MAX), f64::MAX * 0.5);
        assert_eq!(midpoint(-f64::MAX, 0.0), -f64::MAX * 0.5);
        assert_eq!(midpoint(0.0, f64::from_bits(1)), 0.0);
    }

    #[test]
    fn simpson_preserves_representable_extreme_integrals() {
        let huge = 1.0e308;
        let constant = simpson(0.0, huge, 1.0, 1.0, 1.0).unwrap();
        assert!((constant - huge).abs() / huge < 1.0e-15);
        assert_eq!(simpson(-f64::MAX, f64::MAX, 0.0, 0.0, 0.0), Ok(0.0));
        assert_eq!(simpson(-f64::MAX, f64::MAX, -1.0, 0.0, 1.0), Ok(0.0));
        assert_eq!(
            simpson(-f64::MAX, f64::MAX, 1.0, 1.0, 1.0),
            Err(QuadratureNumericalError::NonrepresentableEstimate)
        );
    }
}
