use super::*;
use crate::builtins::deep_learning::layer_helpers::parse_layer_options;

#[test]
fn constructor_admission_uses_the_canonical_convolution_modes() {
    let kind = DeepLearningLayerKind::Convolution1D;
    for mode in DeepLearningConvolutionPaddingMode::ALL {
        let value = Value::String(mode.canonical_name().to_ascii_uppercase());
        assert!(parse_layer_options(
            &[Value::String("Padding".into()), value],
            kind,
            "convolution1dLayer"
        )
        .is_ok());
    }
    for mode in DeepLearningConvolutionPaddingValueMode::ALL {
        let value = Value::String(mode.canonical_name().to_ascii_uppercase());
        assert!(parse_layer_options(
            &[Value::String("PaddingValue".into()), value],
            kind,
            "convolution1dLayer"
        )
        .is_ok());
    }
    for name in ["Padding", "PaddingValue"] {
        assert!(parse_layer_options(
            &[Value::String(name.into()), Value::String("invented".into()),],
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
            &[Value::String(name.into()), value],
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
                    &[
                        Value::String(option.canonical_name().into()),
                        Value::String(value.to_ascii_uppercase()),
                    ],
                    kind,
                    "lstmLayer"
                )
                .is_ok());
            }
            assert!(parse_layer_options(
                &[
                    Value::String(option.canonical_name().into()),
                    Value::String("invented".into()),
                ],
                kind,
                "lstmLayer"
            )
            .is_err());
        }
    }
}
