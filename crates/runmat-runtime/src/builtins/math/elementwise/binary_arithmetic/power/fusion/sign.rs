//! Ordinary real POWER's represented finite even-integral control theorem.
//! Call only after the selected real-domain and arithmetic receipts agree
//! with the exact original scalar bits and operand identities.

use runmat_accelerate_api::PowerScalarControl;

pub(super) fn even_integral_control(control: PowerScalarControl) -> bool {
    let exponent = control.value();
    exponent.is_finite() && exponent.fract() == 0.0 && exponent % 2.0 == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn exact_single_double_bits_cover_even_negative_zero_large_and_reject_other_controls() {
        let single = |value: f32| PowerScalarControl::SingleBits(value.to_bits());
        let double = |value: f64| PowerScalarControl::DoubleBits(value.to_bits());
        for control in [
            single(2.0),
            single(-2.0),
            single(-0.0),
            single(16_777_216.0),
            double(2.0),
            double(-2.0),
            double(9_007_199_254_740_992.0),
        ] {
            assert!(even_integral_control(control), "{control:?}");
        }
        for control in [
            single(1.8),
            single(3.0),
            single(f32::NAN),
            single(f32::INFINITY),
            double(1.5),
            double(3.0),
            double(f64::NEG_INFINITY),
        ] {
            assert!(!even_integral_control(control), "{control:?}");
        }
    }
}
