//! Quotient-free reduction with the existing near-multiple zero compensation.
macro_rules! compensated {
    ($name:ident, $parts:ident, $apply:ident, $ty:ty, $bias:expr, $exponent_mask:expr) => {
        /// Evaluate the admitted real floating strategy without class conversion.
        pub fn $apply(a: $ty, b: $ty, kind: runmat_types::FloatingRemainderKind) -> $ty {
            let modulus = kind == runmat_types::FloatingRemainderKind::Modulus;
            if a.is_nan() || b.is_nan() {
                return <$ty>::NAN;
            }
            if b == 0.0 {
                return if modulus { a } else { <$ty>::NAN };
            }
            if a.is_infinite() {
                return <$ty>::NAN;
            }
            if a == 0.0 {
                return 0.0;
            }
            if b.is_infinite() {
                return if modulus && a.is_sign_negative() != b.is_sign_negative() {
                    b
                } else {
                    a
                };
            }
            let mut result = $name(a, b);
            if modulus && result != 0.0 && result.is_sign_negative() != b.is_sign_negative() {
                result += b;
            }
            if result == 0.0 {
                0.0
            } else {
                result
            }
        }
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
            let (significand, exponent) = $parts(a);
            let (mut divisor, divisor_exponent) = $parts(b);
            let gap = exponent - divisor_exponent;
            let fraction_bits = <$ty>::MANTISSA_DIGITS - 1;
            if gap < 0 || gap > fraction_bits as i32 {
                return remainder;
            }
            // Compare the rounded product at a normal binary scale. This
            // gives CPU and GPU the same policy even when the operands are
            // subnormal and the device flushes arithmetic subnormals.
            for _ in 0..gap {
                divisor *= 0.5;
            }
            let candidate = (significand / divisor).round_ties_even();
            if candidate < ((1_u64 << fraction_bits) as $ty) && candidate * divisor == significand {
                return 0.0;
            }
            remainder
        }

        fn $parts(value: $ty) -> ($ty, i32) {
            let fraction_bits = <$ty>::MANTISSA_DIGITS - 1;
            let implicit = 1_u64 << fraction_bits;
            let bits = value.to_bits() as u64;
            let encoded = ((bits >> fraction_bits) & $exponent_mask) as i32;
            let mut significand = (bits & (implicit - 1)) as $ty;
            let mut exponent = encoded - $bias - fraction_bits as i32;
            if encoded == 0 {
                exponent = 1 - $bias - fraction_bits as i32;
            } else {
                significand += implicit as $ty;
            }
            while significand != 0.0 && significand < implicit as $ty {
                significand *= 2.0;
                exponent -= 1;
            }
            (significand, exponent)
        }
    };
}

compensated!(reduce_f64, parts_f64, apply_f64, f64, 1023, 2047);
compensated!(reduce_f32, parts_f32, apply_f32, f32, 127, 255);

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
        // A rounded product can land just below its mathematical multiple;
        // normalized near-multiple compensation includes that side too.
        assert_eq!(reduce_f64(3.0 * 0.7, 0.7), 0.0);
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

    #[test]
    fn encoded_extremes_and_exceptional_values_keep_the_declared_policy() {
        use runmat_types::FloatingRemainderKind::{Modulus, Remainder};
        let tiny = f64::from_bits(1);
        assert_eq!(apply_f64(f64::MAX, 3.0 * tiny, Remainder).to_bits(), 2);
        assert_eq!(apply_f64(11.0 * tiny, 3.0 * tiny, Remainder).to_bits(), 2);
        assert_eq!(apply_f64(-11.0 * tiny, 3.0 * tiny, Modulus).to_bits(), 1);
        let tiny32 = f32::from_bits(1);
        assert_eq!(apply_f32(f32::MAX, 11.0 * tiny32, Remainder).to_bits(), 10);
        assert_eq!(
            apply_f64(-0.0, 0.0, Modulus).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(apply_f64(-0.0, 1.0, Remainder).to_bits(), 0);
        assert_eq!(apply_f64(-1.0, f64::INFINITY, Modulus), f64::INFINITY);
        assert_eq!(apply_f64(-1.0, f64::INFINITY, Remainder), -1.0);
        for kind in [Modulus, Remainder] {
            assert!(apply_f64(f64::NAN, 0.0, kind).is_nan());
            assert!(apply_f64(f64::INFINITY, 1.0, kind).is_nan());
        }
    }
}
