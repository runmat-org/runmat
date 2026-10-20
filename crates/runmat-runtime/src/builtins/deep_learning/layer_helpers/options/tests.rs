use super::*;
use crate::builtins::deep_learning::layer_helpers::parse_layer_options;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn recurrent_gate_factor_admission_requires_a_row_or_scalar() {
    for kind in [DeepLearningLayerKind::Lstm, DeepLearningLayerKind::Bilstm] {
        let gates = kind.recurrent_gate_count().unwrap();
        for (shape, admitted) in [
            (vec![1, gates], true),
            (vec![gates], false),
            (vec![gates, 1], false),
        ] {
            let factor =
                Value::Tensor(runmat_value::Tensor::from_f32(vec![1.; gates], shape).unwrap());
            assert_eq!(
                recurrent_factor(&factor, kind, "lstmLayer", "WeightLearnRateFactor").is_ok(),
                admitted
            );
        }
        let scalar = Value::Tensor(runmat_value::Tensor::from_f32(vec![1.], Vec::new()).unwrap());
        assert!(recurrent_factor(&scalar, kind, "lstmLayer", "WeightLearnRateFactor").is_ok());
    }
}

#[test]
fn constructor_admission_uses_the_canonical_convolution_modes() {
    let kind = DeepLearningLayerKind::Convolution1D;
    for mode in DeepLearningConvolutionPaddingMode::ALL {
        let value = Value::String(mode.canonical_name().to_ascii_uppercase().into());
        assert!(parse_layer_options(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String("Padding".into()),
                value
            ]),
            kind,
            "convolution1dLayer"
        )
        .is_ok());
    }
    for mode in DeepLearningConvolutionPaddingValueMode::ALL {
        let value = Value::String(mode.canonical_name().to_ascii_uppercase().into());
        assert!(parse_layer_options(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String("PaddingValue".into()),
                value
            ]),
            kind,
            "convolution1dLayer"
        )
        .is_ok());
    }
    for name in ["Padding", "PaddingValue"] {
        assert!(parse_layer_options(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String(name.into()),
                Value::String("invented".into()),
            ]),
            kind,
            "convolution1dLayer"
        )
        .is_err());
    }
    for (name, value) in [
        ("Padding", Value::Num(2.0)),
        ("PaddingValue", Value::Num(3.5)),
    ] {
        assert!(parse_layer_options(
            crate::call::FunctionArgumentView::from_slice(&[Value::String(name.into()), value]),
            kind,
            "convolution1dLayer"
        )
        .is_ok());
    }
}

#[test]
fn constructor_admission_uses_the_canonical_recurrent_modes() {
    for kind in [DeepLearningLayerKind::Lstm, DeepLearningLayerKind::Bilstm] {
        for (option, values) in [
            (
                DeepLearningLayerOption::OutputMode,
                DeepLearningRecurrentOutputMode::NAMES,
            ),
            (
                DeepLearningLayerOption::StateActivationFunction,
                DeepLearningRecurrentStateActivation::NAMES,
            ),
            (
                DeepLearningLayerOption::GateActivationFunction,
                DeepLearningRecurrentGateActivation::NAMES,
            ),
        ] {
            for &value in values {
                assert!(parse_layer_options(
                    crate::call::FunctionArgumentView::from_slice(&[
                        Value::String(option.canonical_name().into()),
                        Value::String(value.to_ascii_uppercase().into()),
                    ]),
                    kind,
                    "lstmLayer"
                )
                .is_ok());
            }
            assert!(parse_layer_options(
                crate::call::FunctionArgumentView::from_slice(&[
                    Value::String(option.canonical_name().into()),
                    Value::String("invented".into()),
                ]),
                kind,
                "lstmLayer"
            )
            .is_err());
        }
    }
}
