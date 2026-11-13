//! Host authority for Double source + original Single control, without GPU credit.
use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn host_double_source_single_control_keeps_double_intermediate_single_result_rank_and_order() {
    let half = f64::from(f32::EPSILON) / 2.0;
    for source in [
        1.0 + half - f64::EPSILON,
        1.0 + half + f64::EPSILON,
        -0.0,
        f64::INFINITY,
    ] {
        for control in [
            0.0_f32,
            -0.0,
            f32::from_bits(1),
            f32::INFINITY,
            f32::from_bits(0x7fc0_1234),
        ] {
            for reverse in [false, true] {
                let double = Value::Tensor(Tensor::new(vec![source; 2], vec![2, 1]).unwrap());
                let single = Value::Tensor(Tensor::from_f32(vec![control], vec![1, 1, 1]).unwrap());
                let expected = (source + f64::from(control)) as f32;
                let (left, right) = if reverse {
                    (single, double)
                } else {
                    (double, single)
                };
                let Value::Tensor(output) = plus_host(left, right).unwrap() else {
                    panic!("mixed host Single tensor");
                };
                assert_eq!(output.shape(), &[2, 1, 1]);
                assert_eq!(output.numeric_dtype(), NumericDType::F32);
                for value in output.as_f32_slice().unwrap() {
                    if expected.is_nan() {
                        assert!(value.is_nan());
                    } else {
                        assert_eq!(value.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }
}
