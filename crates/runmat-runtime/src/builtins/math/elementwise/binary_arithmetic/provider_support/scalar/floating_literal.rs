//! Original ordinary floating scalar class/bits, not the legacy f64 hook ABI.
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_value::Value;

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn host_floating_scalar_literal(
    value: &Value,
) -> Option<ScalarLiteral> {
    match value {
        Value::Num(value) => Some(ScalarLiteral::F64Bits(value.to_bits())),
        Value::Tensor(value) if crate::builtins::common::tensor::is_scalar_tensor(value) => {
            if let Some(values) = value.as_f32_slice() {
                return values
                    .first()
                    .map(|value| ScalarLiteral::F32Bits(value.to_bits()));
            }
            value
                .as_f64_slice()?
                .first()
                .map(|value| ScalarLiteral::F64Bits(value.to_bits()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::Tensor;
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn scalar_class_and_original_ieee_bits_survive_capture() {
        for bits in [
            (-0.0_f64).to_bits(),
            1.0_f64.to_bits(),
            0x7ff8_0000_0000_1234,
        ] {
            let value = f64::from_bits(bits);
            assert_eq!(
                host_floating_scalar_literal(&Value::Num(value)),
                Some(ScalarLiteral::F64Bits(bits))
            );
            assert_eq!(
                host_floating_scalar_literal(&Value::Tensor(
                    Tensor::new(vec![value], vec![1, 1, 1]).unwrap()
                )),
                Some(ScalarLiteral::F64Bits(bits))
            );
        }
        assert_eq!(
            host_floating_scalar_literal(&Value::Tensor(
                Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap()
            )),
            Some(ScalarLiteral::F32Bits((-0.0_f32).to_bits()))
        );
        assert_eq!(host_floating_scalar_literal(&Value::Bool(true)), None);
        assert_eq!(
            host_floating_scalar_literal(&Value::Tensor(
                Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()
            )),
            None
        );
    }
}
