//! Project the homogeneous EVD result without an absolute zero threshold.
use num_complex::Complex64;

pub(crate) fn homogeneous_eigenvalue(alpha: Complex64, beta: Complex64) -> Complex64 {
    if beta.re == 0.0 && beta.im == 0.0 {
        return if alpha.re == 0.0 && alpha.im == 0.0 {
            Complex64::new(f64::NAN, f64::NAN)
        } else {
            // Preserve eig's existing signed-component infinity convention.
            fn infinite(value: f64) -> f64 {
                if value == 0.0 {
                    value
                } else {
                    value.signum() * f64::INFINITY
                }
            }
            Complex64::new(infinite(alpha.re), infinite(alpha.im))
        };
    }
    match super::super::complex_division::divide_finite_components(
        alpha.re, alpha.im, beta.re, beta.im,
    ) {
        Some((real, imag)) => Complex64::new(real, imag),
        // Coefficients are checked finite at the numerical boundary. Keep
        // existing IEEE propagation for a nonfinite numerical solver result.
        None => alpha / beta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn homogeneous_projection_preserves_scaled_regular_and_singular_classes() {
        for scale in [1e-200, 1e-20, 1.0, 1e200, f64::from_bits(2)] {
            let value = homogeneous_eigenvalue(
                Complex64::new(0.5 * scale, 0.0),
                Complex64::new(scale, 0.0),
            );
            assert!(value.re.is_finite() && (value.re - 0.5).abs() <= 2.0 * f64::EPSILON);
            assert_eq!(value.im, 0.0);
        }
        for scale in [1e-200, 1.0, 1e200] {
            let value =
                homogeneous_eigenvalue(Complex64::new(scale, scale), Complex64::new(scale, -scale));
            assert!((value.re).abs() < 1e-14);
            assert!((value.im - 1.0).abs() < 1e-14);
        }
        let extreme = 0.75 * f64::MAX;
        let extreme_ratio = homogeneous_eigenvalue(
            Complex64::new(extreme, extreme),
            Complex64::new(extreme, extreme),
        );
        assert!(
            extreme_ratio.re.is_finite() && (extreme_ratio.re - 1.0).abs() <= 2.0 * f64::EPSILON
        );
        assert_eq!(extreme_ratio.im, 0.0);
        // Halving an entire large numerator before division would erase this
        // still-representable minimum-subnormal component.
        let tiny = f64::from_bits(1);
        assert_eq!(
            homogeneous_eigenvalue(Complex64::new(f64::MAX, tiny), Complex64::new(0.0, 1.0)),
            Complex64::new(tiny, -f64::MAX),
        );
        let infinite =
            homogeneous_eigenvalue(Complex64::new(-1.0, -0.0), Complex64::new(0.0, -0.0));
        assert_eq!(infinite.re, f64::NEG_INFINITY);
        assert_eq!(infinite.im.to_bits(), (-0.0f64).to_bits());
        let complex_infinite =
            homogeneous_eigenvalue(Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0));
        assert_eq!(complex_infinite.re, 0.0);
        assert_eq!(complex_infinite.im, f64::NEG_INFINITY);
        let indeterminate =
            homogeneous_eigenvalue(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
        assert!(indeterminate.re.is_nan() && indeterminate.im.is_nan());
    }
}
