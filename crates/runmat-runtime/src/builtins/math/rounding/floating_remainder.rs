//! Quotient-free reduction with the existing near-multiple zero compensation.
macro_rules! compensated {
    ($name:ident, $ty:ty) => {
        pub(super) fn $name(a: $ty, b: $ty) -> $ty {
            let remainder = a % b;
            if !a.is_finite() || !b.is_finite() || b == 0.0 {
                return remainder;
            }
            // Whole-number operands have an exact remainder in their binary
            // format. Rounded products must not replace that result.
            if a.fract() == 0.0 && b.fract() == 0.0 {
                return remainder;
            }
            let quotient = a / b;
            // Once quotient spacing reaches one, subtraction cannot establish
            // that the represented operands were a near multiple.
            if quotient.abs() < (1.0 / <$ty>::EPSILON) && a - b * quotient.trunc() == 0.0 {
                return 0.0;
            }
            remainder
        }
    };
}

compensated!(reduce_f64, f64);
compensated!(reduce_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_whole_operands_use_an_independent_integer_oracle() {
        for (a, b) in [(100_000_000_000_000_000_i128, 11_i128), (1_i128 << 60, 13)] {
            assert_eq!(reduce_f64(a as f64, b as f64), (a % b) as f64);
            assert_eq!(reduce_f64(-(a as f64), b as f64), -(a % b) as f64);
        }
        assert_eq!(
            reduce_f32(1_000_000_000.0, 7.0),
            (1_000_000_000_i128 % 7) as f32
        );
    }

    #[test]
    fn compensation_does_not_replace_large_fractional_quotient_reduction() {
        assert_eq!(reduce_f64(2.0, 0.1), 0.0);
        assert_eq!(reduce_f32(2.0, 0.1), 0.0);
        // Scale exact whole operands by a power of two: the integer oracle
        // remains exact, while the divisor is no longer a whole number.
        let a = 100_000_000_000_000_000_i128;
        assert_eq!(
            reduce_f64(a as f64 / 16.0, 11.0 / 16.0),
            (a % 11) as f64 / 16.0
        );
        assert_eq!(
            reduce_f32(1_000_000_000.0 / 16.0, 7.0 / 16.0),
            (1_000_000_000_i128 % 7) as f32 / 16.0
        );
    }
}
