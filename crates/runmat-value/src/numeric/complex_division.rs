//! Finite complex quotients without squaring an unscaled denominator.
//!
//! The finite path represents each product as a mantissa and binary exponent.
//! This retains a tiny component when the other numerator component is huge,
//! including when the denominator selects that tiny component. Zero divisors
//! and nonfinite operands retain the previous elementwise formula.

macro_rules! finite_quotient {
    ($name:ident, $float:ty, $frexp:path, $scalbn:path, $legacy:ident, $low:expr, $high:expr) => {
        pub fn $name((a, b): ($float, $float), (c, d): ($float, $float)) -> ($float, $float) {
            if !a.is_finite()
                || !b.is_finite()
                || !c.is_finite()
                || !d.is_finite()
                || (c == 0.0 && d == 0.0)
            {
                return $legacy((a, b), (c, d));
            }

            // Here every nonzero product, two-term sum, squared denominator,
            // and quotient stays representable. This uses exactly the prior
            // arithmetic for ordinary finite operands.
            let ordinary =
                |value: $float| value == 0.0 || (value.abs() >= $low && value.abs() <= $high);
            if [a, b, c, d].into_iter().all(ordinary) {
                return $legacy((a, b), (c, d));
            }

            // Products cannot overflow: frexp mantissas have magnitude < 1.
            fn product(x: $float, y: $float) -> ($float, i32) {
                if x == 0.0 || y == 0.0 {
                    return (x * y, 0);
                }
                let (xm, xe) = $frexp(x);
                let (ym, ye) = $frexp(y);
                (xm * ym, xe + ye)
            }

            fn sum((am, ae): ($float, i32), (bm, be): ($float, i32)) -> ($float, i32) {
                if am == 0.0 && bm == 0.0 {
                    return (am + bm, 0);
                }
                if am == 0.0 {
                    return (bm, be);
                }
                if bm == 0.0 {
                    return (am, ae);
                }
                let exponent = ae.max(be);
                let combined = $scalbn(am, ae - exponent) + $scalbn(bm, be - exponent);
                if combined == 0.0 {
                    return (combined, 0);
                }
                let (fraction, shift) = $frexp(combined);
                (fraction, exponent + shift)
            }

            fn quotient((nm, ne): ($float, i32), (dm, de): ($float, i32)) -> $float {
                if nm == 0.0 {
                    return nm / dm;
                }
                $scalbn(nm / dm, ne - de)
            }

            let denominator = sum(product(c, c), product(d, d));
            let real = sum(product(a, c), product(b, d));
            let (minus_ad, ad_exponent) = product(a, d);
            let imaginary = sum(product(b, c), (-minus_ad, ad_exponent));
            (
                quotient(real, denominator),
                quotient(imaginary, denominator),
            )
        }
    };
}

finite_quotient!(
    divide_f64,
    f64,
    libm::frexp,
    libm::scalbn,
    legacy_f64,
    f64::from_bits(773_u64 << 52),
    f64::from_bits(1273_u64 << 52)
);
finite_quotient!(
    divide_f32,
    f32,
    libm::frexpf,
    libm::scalbnf,
    legacy_f32,
    f32::from_bits(97_u32 << 23),
    f32::from_bits(157_u32 << 23)
);

fn legacy_f64((a, b): (f64, f64), (c, d): (f64, f64)) -> (f64, f64) {
    let divisor = c * c + d * d;
    if divisor == 0.0 {
        (f64::NAN, f64::NAN)
    } else {
        ((a * c + b * d) / divisor, (b * c - a * d) / divisor)
    }
}

fn legacy_f32((a, b): (f32, f32), (c, d): (f32, f32)) -> (f32, f32) {
    let divisor = c * c + d * d;
    if divisor == 0.0 {
        (f32::NAN, f32::NAN)
    } else {
        ((a * c + b * d) / divisor, (b * c - a * d) / divisor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_quotients_survive_large_small_and_subnormal_scales() {
        for scale in [1.0e200, 1.0e-200, f64::from_bits(1)] {
            let (re, im) = divide_f64((scale, 0.0), (scale, scale));
            assert!((re - 0.5).abs() < 1.0e-15, "scale={scale}: {re}");
            assert!((im + 0.5).abs() < 1.0e-15, "scale={scale}: {im}");
        }
        for scale in [1.0e30_f32, 1.0e-30_f32, f32::from_bits(1)] {
            let (re, im) = divide_f32((scale, 0.0), (scale, scale));
            assert!((re - 0.5).abs() < 1.0e-6, "scale={scale}: {re}");
            assert!((im + 0.5).abs() < 1.0e-6, "scale={scale}: {im}");
        }
        let (re, im) = divide_f64((1.0e-200, 1.0e200), (1.0e-200, 1.0e200));
        assert!((re - 1.0).abs() < 1.0e-15 && im.abs() < 1.0e-15);

        // A common numerator scale would discard the minor component before
        // this axis-aligned division selects it.
        let tiny = f64::from_bits(1);
        assert_eq!(divide_f64((f64::MAX, tiny), (0.0, 1.0)), (tiny, -f64::MAX));
        let tiny_single = f32::from_bits(1);
        assert_eq!(
            divide_f32((f32::MAX, tiny_single), (0.0, 1.0)),
            (tiny_single, -f32::MAX)
        );
        // Mixed, non-axis denominator: the minor numerator component is
        // still visible in the real result when the major term vanishes.
        let (re, im) = divide_f64((f64::MAX, tiny), (tiny, 1.0));
        assert!(re.is_finite() && im.is_finite());
        assert!((re / (f64::MAX * tiny) - 1.0).abs() < 1.0e-12);
        assert_eq!(im, -f64::MAX);
        assert_eq!(
            divide_f64((f64::MAX, f64::MAX), (0.5, -0.5)),
            (0.0, f64::INFINITY)
        );
        assert_eq!(
            divide_f64((f64::MAX, -f64::MAX), (f64::MAX, f64::MAX)),
            (0.0, -1.0)
        );
    }

    #[test]
    fn zero_and_nonfinite_operands_preserve_previous_policy() {
        assert!(divide_f64((1.0, 0.0), (0.0, 0.0)).0.is_nan());
        assert!(divide_f32((1.0, 0.0), (0.0, 0.0)).0.is_nan());
        let old = legacy_f64((f64::INFINITY, 1.0), (2.0, 3.0));
        let new = divide_f64((f64::INFINITY, 1.0), (2.0, 3.0));
        assert_eq!(
            (old.0.is_nan(), old.1.is_nan()),
            (new.0.is_nan(), new.1.is_nan())
        );
        for a in [0.0, -0.0] {
            for b in [0.0, -0.0] {
                for denominator in [(1.0, 0.0), (1.0, -0.0), (0.0, 1.0), (-0.0, 1.0)] {
                    let old = legacy_f64((a, b), denominator);
                    let new = divide_f64((a, b), denominator);
                    assert_eq!(
                        (old.0.to_bits(), old.1.to_bits()),
                        (new.0.to_bits(), new.1.to_bits())
                    );
                }
            }
        }
    }
}
