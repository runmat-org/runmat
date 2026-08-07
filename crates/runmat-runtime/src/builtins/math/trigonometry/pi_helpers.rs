const MIN_ALWAYS_INTEGRAL: f64 = 4_503_599_627_370_496.0;

fn half_turn_mod4(value: f64) -> Option<i32> {
    if !value.is_finite() {
        return None;
    }
    let doubled = value * 2.0;
    if !doubled.is_finite() {
        return (value.abs() >= MIN_ALWAYS_INTEGRAL).then_some(0);
    }
    let rounded = doubled.round();
    if doubled != rounded {
        return None;
    }
    match rounded.rem_euclid(4.0) {
        0.0 => Some(0),
        1.0 => Some(1),
        2.0 => Some(2),
        3.0 => Some(3),
        _ => None,
    }
}

pub(crate) fn sinpi_real(value: f64) -> f64 {
    if !value.is_finite() {
        return f64::NAN;
    }
    match half_turn_mod4(value) {
        Some(0 | 2) => 0.0,
        Some(1) => 1.0,
        Some(3) => -1.0,
        _ => (std::f64::consts::PI * value).sin(),
    }
}

pub(crate) fn cospi_real(value: f64) -> f64 {
    if !value.is_finite() {
        return f64::NAN;
    }
    match half_turn_mod4(value) {
        Some(0) => 1.0,
        Some(1 | 3) => 0.0,
        Some(2) => -1.0,
        _ => {
            let cycle = value.rem_euclid(2.0);
            let reduced = if cycle > 1.0 { 2.0 - cycle } else { cycle };
            (std::f64::consts::PI * reduced).cos()
        }
    }
}

fn mul_preserving_exact_zero(factor: f64, magnitude: f64) -> f64 {
    if factor == 0.0 && !magnitude.is_nan() {
        factor
    } else {
        factor * magnitude
    }
}

pub(crate) fn sinpi_complex(re: f64, im: f64) -> (f64, f64) {
    let scaled_im = std::f64::consts::PI * im;
    let sin_re = sinpi_real(re);
    let cos_re = cospi_real(re);
    (
        mul_preserving_exact_zero(sin_re, scaled_im.cosh()),
        mul_preserving_exact_zero(cos_re, scaled_im.sinh()),
    )
}

pub(crate) fn cospi_complex(re: f64, im: f64) -> (f64, f64) {
    let scaled_im = std::f64::consts::PI * im;
    let cos_re = cospi_real(re);
    let neg_sin_re = -sinpi_real(re);
    (
        mul_preserving_exact_zero(cos_re, scaled_im.cosh()),
        mul_preserving_exact_zero(neg_sin_re, scaled_im.sinh()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn sinpi_exact_half_turns() {
        assert_eq!(sinpi_real(0.0), 0.0);
        assert_eq!(sinpi_real(0.5), 1.0);
        assert_eq!(sinpi_real(1.0), 0.0);
        assert_eq!(sinpi_real(1.5), -1.0);
        assert_eq!(sinpi_real(-0.5), -1.0);
        assert_eq!(sinpi_real(-1.0), 0.0);
        assert_eq!(sinpi_real(9_007_199_254_740_991.0), 0.0);
        assert_eq!(sinpi_real(9_007_199_254_740_992.0), 0.0);
        assert_eq!(sinpi_real(1.0e300), 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cospi_exact_half_turns() {
        assert_eq!(cospi_real(0.0), 1.0);
        assert_eq!(cospi_real(0.5), 0.0);
        assert_eq!(cospi_real(1.0), -1.0);
        assert_eq!(cospi_real(1.5), 0.0);
        assert_eq!(cospi_real(-0.5), 0.0);
        assert_eq!(cospi_real(-1.0), -1.0);
        assert_eq!(cospi_real(9_007_199_254_740_991.0), -1.0);
        assert_eq!(cospi_real(9_007_199_254_740_992.0), 1.0);
        assert_eq!(cospi_real(1.0e300), 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn complex_exact_zero_factors_survive_overflowing_imaginary_scale() {
        let (re, im) = sinpi_complex(0.0, f64::INFINITY);
        assert_eq!(re, 0.0);
        assert!(im.is_infinite() && im.is_sign_positive());

        let (re, im) = sinpi_complex(0.5, f64::INFINITY);
        assert!(re.is_infinite() && re.is_sign_positive());
        assert_eq!(im, 0.0);

        let (re, im) = cospi_complex(0.5, f64::INFINITY);
        assert_eq!(re, 0.0);
        assert!(im.is_infinite() && im.is_sign_negative());

        let (re, im) = cospi_complex(1.0, 1.0e300);
        assert!(re.is_infinite() && re.is_sign_negative());
        assert_eq!(im, -0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn real_nonfinite_inputs_return_nan() {
        assert!(sinpi_real(f64::NAN).is_nan());
        assert!(sinpi_real(f64::INFINITY).is_nan());
        assert!(sinpi_real(f64::NEG_INFINITY).is_nan());
        assert!(cospi_real(f64::NAN).is_nan());
        assert!(cospi_real(f64::INFINITY).is_nan());
        assert!(cospi_real(f64::NEG_INFINITY).is_nan());
    }
}
