//! Pure numeric rounding after the owning form has validated digit controls.
//! Providers reuse these host semantics rather than copying the builtin parser.

macro_rules! floating_digits {
    ($decimal:ident, $significant:ident, $scaled:ident, $ty:ty, $maximum:expr, $subnormal:expr, $roundtrip:expr) => {
        /// Round a represented floating value to the admitted decimal count.
        pub fn $decimal(value: $ty, digits: i32) -> $ty {
            $scaled(value, i64::from(digits))
        }

        /// Round to an admitted positive number of significant digits.
        pub fn $significant(value: $ty, digits: i32) -> $ty {
            if !value.is_finite() {
                return value;
            }
            if value == 0.0 {
                return 0.0;
            }
            // A round-trip decimal representation already identifies this
            // binary value uniquely; additional digits cannot change it.
            if digits >= $roundtrip {
                return value;
            }
            let order = value.abs().log10().floor() as i64;
            $scaled(value, i64::from(digits) - 1 - order)
        }

        fn $scaled(value: $ty, digits: i64) -> $ty {
            if !value.is_finite() || value == 0.0 {
                return value;
            }
            let order = value.abs().log10().floor() as i64;
            if digits >= i64::from($roundtrip) - 1 - order {
                return value;
            }
            if digits > $subnormal {
                return value;
            }
            if digits < -$maximum {
                return (0.0 as $ty).copysign(value);
            }
            // Split only when the decimal scale itself cannot be represented.
            // This retains subnormal values without constructing 10^324.
            let first = digits.min($maximum) as i32;
            let second = (digits - i64::from(first)) as i32;
            let factor = (10.0 as $ty).powi(first.abs());
            let tail = (10.0 as $ty).powi(second);
            let scaled = if first < 0 {
                value / factor
            } else {
                (value * factor) * tail
            };
            if !scaled.is_finite() {
                // At this magnitude the requested fractional decimal quantum
                // is below the represented value's spacing; it cannot change it.
                return value;
            }
            if first < 0 {
                scaled.round() * factor
            } else {
                (scaled.round() / factor) / tail
            }
        }
    };
}

floating_digits!(decimal_f64, significant_f64, scaled_f64, f64, 308, 324, 17);
floating_digits!(decimal_f32, significant_f32, scaled_f32, f32, 38, 45, 9);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn zero_sign_distinguishes_decimal_and_significant_forms() {
        for value in [0.0_f32, -0.0_f32] {
            assert_eq!(decimal_f32(value, 1).to_bits(), value.to_bits());
            assert_eq!(significant_f32(value, 2).to_bits(), 0.0_f32.to_bits());
        }
        for value in [0.0_f64, -0.0_f64] {
            assert_eq!(decimal_f64(value, 1).to_bits(), value.to_bits());
            assert_eq!(significant_f64(value, 2).to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn extreme_digits_do_not_overflow_control_arithmetic_or_finite_values() {
        assert_eq!(decimal_f64(23.75, 20).to_bits(), 23.75_f64.to_bits());
        assert_eq!(significant_f64(23.75, 22).to_bits(), 23.75_f64.to_bits());
        assert_eq!(decimal_f32(23.75, 20).to_bits(), 23.75_f32.to_bits());
        assert_eq!(significant_f32(23.75, 22).to_bits(), 23.75_f32.to_bits());
        assert_eq!(
            decimal_f64(1.8050000000000002, 16).to_bits(),
            1.8050000000000002_f64.to_bits()
        );
        assert_eq!(decimal_f32(5.369, 8).to_bits(), 5.369_f32.to_bits());
        assert_eq!(significant_f64(1.25, i32::MAX), 1.25);
        assert_eq!(significant_f32(1.25, i32::MAX), 1.25);
        assert_eq!(decimal_f64(f64::MAX, 2), f64::MAX);
        assert_eq!(decimal_f32(f32::MAX, 2), f32::MAX);
        assert_eq!(decimal_f64(123.0, i32::MIN), 0.0);
        assert_eq!(decimal_f32(123.0, i32::MIN), 0.0);
        assert_eq!(
            decimal_f64(-123.0, i32::MIN).to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn subnormal_scaling_and_ordinary_decimal_rounding_remain_useful() {
        assert_eq!(decimal_f64(750_000_000_000.0, -11), 800_000_000_000.0);
        assert_eq!(decimal_f64(150_000.0, -5), 200_000.0);
        assert!((decimal_f64(1.234e-310, 310) / 1e-310 - 1.0).abs() < 1e-12);
        assert!((decimal_f32(1.234e-40, 40) / 1e-40 - 1.0).abs() < 1e-4);
        assert_eq!(decimal_f64(863178137.0, -2), 863178100.0);
        assert_eq!(significant_f64(1253.0, 2), 1300.0);
        assert_eq!(decimal_f64(2.05, 1), 2.1);
    }
}
