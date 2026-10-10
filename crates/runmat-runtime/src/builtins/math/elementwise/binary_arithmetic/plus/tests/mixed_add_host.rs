//! Ordinary host authority for mixed-precision rounding and scalar rank.
use super::*;
#[path = "mixed_double_source_host.rs"]
mod double_source;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ordinary_host_mixed_add_retains_double_rounding_both_tie_sides_and_scalar_shapes() {
    let below = f32::from_bits(0x3f7f_ffff);
    let half = f64::from(f32::EPSILON) / 4.0;
    for scalar in [half - f64::EPSILON, half + f64::EPSILON] {
        let expected = (f64::from(below) + scalar) as f32;
        let rounded_first = (f64::from(below) + f64::from(scalar as f32)) as f32;
        assert_eq!(
            expected.to_bits(),
            if scalar < half {
                below.to_bits()
            } else {
                1.0_f32.to_bits()
            }
        );
        if scalar < half {
            assert_ne!(expected.to_bits(), rounded_first.to_bits());
        }
        for reverse in [false, true] {
            let single = Value::Tensor(Tensor::from_f32(vec![below; 2], vec![2, 1]).unwrap());
            let double = Value::Tensor(Tensor::new(vec![scalar], vec![1, 1, 1]).unwrap());
            let (left, right) = if reverse {
                (double, single)
            } else {
                (single, double)
            };
            let Value::Tensor(output) = plus_host(left, right).unwrap() else {
                panic!("Single host result is a tensor");
            };
            assert_eq!(output.shape(), &[2, 1, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            assert!(output
                .as_f32_slice()
                .unwrap()
                .iter()
                .all(|value| value.to_bits() == expected.to_bits()));
        }
    }
}
