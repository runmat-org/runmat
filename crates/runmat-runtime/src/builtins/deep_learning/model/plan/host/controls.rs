#[cfg(test)]
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_builtins::{
    DeepLearningConvolutionPaddingMode, DeepLearningConvolutionPaddingValueMode,
    DeepLearningLayerKind as K, DeepLearningLayerOption as O,
    DeepLearningLayerStructuralProperty as P, DeepLearningNormalizationAxes,
    DeepLearningRecurrentGateActivation, DeepLearningRecurrentOutputMode,
    DeepLearningRecurrentStateActivation,
};
use runmat_value::{NumericScalar, ObjectInstance, Value};

use crate::builtins::deep_learning::model::execution::host::{conv, normalization, recurrent};
use crate::builtins::deep_learning::{
    deep_learning_error, logical_scalar, numeric_scalar, positive_usize, scalar_text,
};
use crate::BuiltinResult;

#[derive(Clone, Copy)]
pub(in crate::builtins::deep_learning) enum Controls {
    Input { min_length: usize },
    Dense,
    Convolution(conv::ConvOptions),
    Recurrent(recurrent::RecurrentOptions),
    Normalization(normalization::NormalizationOptions),
    Pooling,
    Relu,
    Elu { alpha: f64 },
    Softmax,
    Output,
}

fn value(layer: &ObjectInstance, kind: K, option: O) -> Option<&Value> {
    layer
        .properties()
        .iter()
        .find_map(|(name, value)| (kind.option(name) == Some(option)).then_some(value))
}

fn text(
    layer: &ObjectInstance,
    kind: K,
    option: O,
    default: &'static str,
    function: &'static str,
) -> BuiltinResult<String> {
    value(layer, kind, option)
        .map(|value| scalar_text(value, function))
        .transpose()
        .map(|value| value.unwrap_or_else(|| default.to_string()))
}

fn positive(
    layer: &ObjectInstance,
    kind: K,
    option: O,
    default: usize,
    function: &'static str,
) -> BuiltinResult<usize> {
    value(layer, kind, option)
        .map(|value| positive_usize(value, function, option.canonical_name()))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn invalid(function: &'static str, option: O) -> crate::RuntimeError {
    deep_learning_error(
        function,
        format!(
            "{function}: unsupported {} execution option",
            option.canonical_name()
        ),
    )
}

pub(super) fn decode(
    layer: &ObjectInstance,
    kind: K,
    function: &'static str,
) -> BuiltinResult<Controls> {
    Ok(match kind {
        K::FeatureInput | K::SequenceInput => {
            let min_length = if kind == K::SequenceInput {
                positive(layer, kind, O::MinLength, 1, function)?
            } else {
                1
            };
            Controls::Input { min_length }
        }
        K::FullyConnected => Controls::Dense,
        K::Convolution1D => {
            let stride = positive(layer, kind, O::Stride, 1, function)?;
            let dilation = positive(layer, kind, O::DilationFactor, 1, function)?;
            let padding = match value(layer, kind, O::Padding) {
                None => conv::Padding::Valid,
                Some(value) => {
                    if let Ok(name) = scalar_text(value, function) {
                        DeepLearningConvolutionPaddingMode::from_name(&name)
                            .map(Into::into)
                            .ok_or_else(|| invalid(function, O::Padding))?
                    } else {
                        let values = padding_extents(value, function)?;
                        match values.as_slice() {
                            [0] | [0, 0] => conv::Padding::Valid,
                            [extent] => conv::Padding::Explicit {
                                left: *extent,
                                right: *extent,
                            },
                            [left, right] => conv::Padding::Explicit {
                                left: *left,
                                right: *right,
                            },
                            _ => return Err(invalid(function, O::Padding)),
                        }
                    }
                }
            };
            let padding_value = match value(layer, kind, O::PaddingValue) {
                None => conv::PaddingValue::Constant(0.0),
                Some(value) => {
                    if let Ok(name) = scalar_text(value, function) {
                        DeepLearningConvolutionPaddingValueMode::from_name(&name)
                            .map(Into::into)
                            .ok_or_else(|| invalid(function, O::PaddingValue))?
                    } else {
                        conv::PaddingValue::Constant(numeric_scalar(
                            value,
                            function,
                            O::PaddingValue.canonical_name(),
                        )?)
                    }
                }
            };
            Controls::Convolution(conv::ConvOptions {
                stride,
                dilation,
                padding,
                padding_value,
            })
        }
        K::Lstm | K::Bilstm => {
            for option in [O::HasStateInputs, O::HasStateOutputs] {
                if value(layer, kind, option)
                    .map(|v| logical_scalar(v, function, option.canonical_name()))
                    .transpose()?
                    .unwrap_or(false)
                {
                    return Err(invalid(function, option));
                }
            }
            let output_mode = DeepLearningRecurrentOutputMode::from_name(&text(
                layer,
                kind,
                O::OutputMode,
                DeepLearningRecurrentOutputMode::Sequence.canonical_name(),
                function,
            )?)
            .ok_or_else(|| invalid(function, O::OutputMode))?;
            let state_activation = DeepLearningRecurrentStateActivation::from_name(&text(
                layer,
                kind,
                O::StateActivationFunction,
                DeepLearningRecurrentStateActivation::Tanh.canonical_name(),
                function,
            )?)
            .ok_or_else(|| invalid(function, O::StateActivationFunction))?;
            let gate_activation = DeepLearningRecurrentGateActivation::from_name(&text(
                layer,
                kind,
                O::GateActivationFunction,
                DeepLearningRecurrentGateActivation::Sigmoid.canonical_name(),
                function,
            )?)
            .ok_or_else(|| invalid(function, O::GateActivationFunction))?;
            let hidden_size = layer
                .properties()
                .get(P::NumHiddenUnits.canonical_name())
                .ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: recurrent layer is missing NumHiddenUnits"),
                    )
                })
                .and_then(|value| {
                    positive_usize(value, function, P::NumHiddenUnits.canonical_name())
                })?;
            Controls::Recurrent(recurrent::RecurrentOptions {
                hidden_size,
                bidirectional: kind == K::Bilstm,
                output_mode,
                state_activation,
                gate_activation,
            })
        }
        K::LayerNormalization => {
            let epsilon = value(layer, kind, O::Epsilon)
                .map(|v| numeric_scalar(v, function, O::Epsilon.canonical_name()))
                .transpose()?
                .unwrap_or(1e-5);
            let operation = text(layer, kind, O::OperationDimension, "auto", function)?;
            let operation = DeepLearningNormalizationAxes::from_name(&operation)
                .ok_or_else(|| invalid(function, O::OperationDimension))?;
            Controls::Normalization(normalization::NormalizationOptions { epsilon, operation })
        }
        K::GlobalAveragePooling1D => Controls::Pooling,
        K::Relu => Controls::Relu,
        K::Elu => {
            let alpha = layer
                .properties()
                .get(P::Alpha.canonical_name())
                .map(|value| numeric_scalar(value, function, P::Alpha.canonical_name()))
                .transpose()?
                .unwrap_or(1.0);
            Controls::Elu { alpha }
        }
        K::Softmax => Controls::Softmax,
        K::ClassificationOutput | K::RegressionOutput => Controls::Output,
    })
}

fn padding_extents(value: &Value, function: &'static str) -> BuiltinResult<Vec<usize>> {
    let values = match value {
        Value::Num(value) => vec![NumericScalar::F64(*value)],
        Value::Int(value) => vec![value.clone().into()],
        Value::Tensor(tensor) if (1..=2).contains(&tensor.len()) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .expect("admitted element index")
            })
            .collect(),
        _ => return Err(invalid(function, O::Padding)),
    };
    values
        .into_iter()
        .map(|value| {
            if let Some(integer) = value.into_int_value() {
                return integer
                    .try_to_usize()
                    .ok_or_else(|| invalid(function, O::Padding));
            }
            let float = match value {
                NumericScalar::F64(value) => value,
                NumericScalar::F32(value) => f64::from(value),
                _ => unreachable!(),
            };
            if float.is_finite()
                && float >= 0.0
                && float.fract() == 0.0
                && float < usize::MAX as f64
            {
                Ok(float as usize)
            } else {
                Err(invalid(function, O::Padding))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{BILSTM_LAYER_CLASS, CONVOLUTION_1D_LAYER_CLASS, LSTM_LAYER_CLASS};

    fn convolution(padding: Option<Value>) -> ObjectInstance {
        let mut layer = ObjectInstance::new(CONVOLUTION_1D_LAYER_CLASS.owned());
        if let Some(padding) = padding {
            layer
                .replace_dl_property("Padding", padding)
                .expect("fixture property");
        }
        layer
    }

    fn admitted_padding(layer: &ObjectInstance) -> conv::Padding {
        let Controls::Convolution(options) = decode(layer, K::Convolution1D, "forward").unwrap()
        else {
            panic!("convolution controls expected")
        };
        options.padding
    }

    #[test]
    fn absent_and_explicit_zero_padding_use_valid_geometry() {
        assert_eq!(admitted_padding(&convolution(None)), conv::Padding::Valid);
        assert_eq!(
            admitted_padding(&convolution(Some(Value::Num(0.0)))),
            conv::Padding::Valid
        );
        assert_eq!(
            admitted_padding(&convolution(Some(Value::String("same".into())))),
            conv::Padding::Same
        );
    }

    #[test]
    fn convolution_text_modes_lower_without_changing_numeric_padding() {
        for mode in DeepLearningConvolutionPaddingMode::ALL {
            assert_eq!(
                admitted_padding(&convolution(Some(Value::String(
                    mode.canonical_name().to_ascii_uppercase().into()
                )))),
                conv::Padding::from(mode)
            );
        }
        assert_eq!(
            admitted_padding(&convolution(Some(Value::Num(2.0)))),
            conv::Padding::Explicit { left: 2, right: 2 }
        );
        for mode in DeepLearningConvolutionPaddingValueMode::ALL {
            let mut layer = convolution(None);
            layer
                .replace_dl_property(
                    O::PaddingValue.canonical_name(),
                    Value::String(mode.canonical_name().to_ascii_uppercase().into()),
                )
                .expect("fixture property");
            let Controls::Convolution(options) =
                decode(&layer, K::Convolution1D, "forward").expect("padding-value mode must lower")
            else {
                panic!("convolution controls expected")
            };
            assert_eq!(options.padding_value, conv::PaddingValue::from(mode));
        }
        let mut layer = convolution(None);
        layer
            .replace_dl_property(O::PaddingValue.canonical_name(), Value::Num(3.5))
            .expect("fixture property");
        let Controls::Convolution(options) =
            decode(&layer, K::Convolution1D, "forward").expect("numeric padding value must lower")
        else {
            panic!("convolution controls expected")
        };
        assert_eq!(options.padding_value, conv::PaddingValue::Constant(3.5));
        for option in [O::Padding, O::PaddingValue] {
            let mut layer = convolution(None);
            layer
                .replace_dl_property(option.canonical_name(), Value::String("invented".into()))
                .expect("fixture property");
            assert!(decode(&layer, K::Convolution1D, "forward").is_err());
        }
    }

    #[test]
    fn recurrent_modes_lower_for_both_layer_kinds() {
        for (kind, class) in [(K::Lstm, LSTM_LAYER_CLASS), (K::Bilstm, BILSTM_LAYER_CLASS)] {
            for mode in DeepLearningRecurrentOutputMode::ALL {
                let mut layer = ObjectInstance::new(class.owned());
                layer
                    .replace_dl_property("NumHiddenUnits", Value::Num(2.0))
                    .expect("fixture property");
                layer
                    .replace_dl_property(
                        O::OutputMode.canonical_name(),
                        Value::String(mode.canonical_name().to_ascii_uppercase().into()),
                    )
                    .expect("fixture property");
                let Controls::Recurrent(options) =
                    decode(&layer, kind, "forward").expect("output mode must lower")
                else {
                    panic!("recurrent controls expected")
                };
                assert_eq!(options.output_mode, mode);
                for activation in DeepLearningRecurrentStateActivation::ALL {
                    layer
                        .replace_dl_property(
                            O::StateActivationFunction.canonical_name(),
                            Value::String(activation.canonical_name().to_ascii_uppercase().into()),
                        )
                        .expect("fixture property");
                    let Controls::Recurrent(options) =
                        decode(&layer, kind, "forward").expect("state activation must lower")
                    else {
                        panic!("recurrent controls expected")
                    };
                    assert_eq!(options.state_activation, activation);
                }
                layer
                    .remove_dl_property(O::StateActivationFunction.canonical_name())
                    .expect("fixture property removal");
                for activation in DeepLearningRecurrentGateActivation::ALL {
                    layer
                        .replace_dl_property(
                            O::GateActivationFunction.canonical_name(),
                            Value::String(activation.canonical_name().to_ascii_uppercase().into()),
                        )
                        .expect("fixture property");
                    let Controls::Recurrent(options) =
                        decode(&layer, kind, "forward").expect("gate activation must lower")
                    else {
                        panic!("recurrent controls expected")
                    };
                    assert_eq!(options.gate_activation, activation);
                }
                layer
                    .remove_dl_property(O::GateActivationFunction.canonical_name())
                    .expect("fixture property removal");
                for option in [
                    O::OutputMode,
                    O::StateActivationFunction,
                    O::GateActivationFunction,
                ] {
                    layer
                        .replace_dl_property(
                            option.canonical_name(),
                            Value::String("invented".into()),
                        )
                        .expect("fixture property");
                    assert!(decode(&layer, kind, "forward").is_err());
                    layer
                        .remove_dl_property(option.canonical_name())
                        .expect("fixture property removal");
                }
            }
        }
    }
}
