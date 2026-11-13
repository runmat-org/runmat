//! Original scalar class survives identity. Only a Single control's value is
//! widened here; no device handle, output carrier, or conversion effect is minted.
use super::*;

pub(super) enum InputRoute {
    SingleSource,
    DoubleSource,
}

pub(super) fn native_double_control(
    source: ProviderElementType,
    control: ScalarLiteral,
) -> Option<(InputRoute, ScalarLiteral)> {
    match (source, control) {
        (ProviderElementType::F32, ScalarLiteral::F64Bits(_)) => {
            Some((InputRoute::SingleSource, control))
        }
        (ProviderElementType::F64, ScalarLiteral::F32Bits(bits)) => Some((
            InputRoute::DoubleSource,
            ScalarLiteral::F64Bits(f64::from(f32::from_bits(bits)).to_bits()),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn exact_single_control_projection_keeps_original_class_outside_native_encoding() {
        for bits in [
            0_u32,
            0x8000_0000,
            1,
            0x8000_0001,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_1234,
        ] {
            let original = ScalarLiteral::F32Bits(bits);
            let (route, native) =
                native_double_control(ProviderElementType::F64, original).unwrap();
            assert!(matches!(route, InputRoute::DoubleSource));
            assert_eq!(original, ScalarLiteral::F32Bits(bits));
            let ScalarLiteral::F64Bits(widened) = native else {
                unreachable!()
            };
            let single = f32::from_bits(bits);
            if single.is_nan() {
                assert!(f64::from_bits(widened).is_nan());
            } else {
                assert_eq!((f64::from_bits(widened) as f32).to_bits(), bits);
            }
            assert!(native_double_control(ProviderElementType::F32, original).is_none());
        }
        let original = ScalarLiteral::F64Bits(0x7ff8_0000_0000_1234);
        let (route, native) = native_double_control(ProviderElementType::F32, original).unwrap();
        assert!(matches!(route, InputRoute::SingleSource));
        assert_eq!(native, original);
        assert!(native_double_control(ProviderElementType::F64, original).is_none());
    }
}
