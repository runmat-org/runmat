use runmat_builtins::{
    DeepLearningLayerKind as K, DeepLearningLayerOption as O, DeepLearningNormalizationAxes,
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
        .properties
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
                Some(value)
                    if scalar_text(value, function)
                        .is_ok_and(|name| name.eq_ignore_ascii_case("same")) =>
                {
                    conv::Padding::Same
                }
                Some(value)
                    if scalar_text(value, function)
                        .is_ok_and(|name| name.eq_ignore_ascii_case("causal")) =>
                {
                    conv::Padding::Causal
                }
                Some(value) => {
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
            };
            let padding_value = match value(layer, kind, O::PaddingValue) {
                None => conv::PaddingValue::Constant(0.0),
                Some(value)
                    if scalar_text(value, function)
                        .is_ok_and(|name| name.eq_ignore_ascii_case("replicate")) =>
                {
                    conv::PaddingValue::Replicate
                }
                Some(value)
                    if scalar_text(value, function)
                        .is_ok_and(|name| name.eq_ignore_ascii_case("symmetric-include-edge")) =>
                {
                    conv::PaddingValue::SymmetricIncludeEdge
                }
                Some(value)
                    if scalar_text(value, function)
                        .is_ok_and(|name| name.eq_ignore_ascii_case("symmetric-exclude-edge")) =>
                {
                    conv::PaddingValue::SymmetricExcludeEdge
                }
                Some(value) => {
                    conv::PaddingValue::Constant(numeric_scalar(value, function, "PaddingValue")?)
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
            let output_mode = match text(layer, kind, O::OutputMode, "sequence", function)?
                .to_ascii_lowercase()
                .as_str()
            {
                "sequence" => recurrent::OutputMode::Sequence,
                "last" => recurrent::OutputMode::Last,
                _ => return Err(invalid(function, O::OutputMode)),
            };
            let state_activation =
                match text(layer, kind, O::StateActivationFunction, "tanh", function)?
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "tanh" => recurrent::Activation::Tanh,
                    "softsign" => recurrent::Activation::Softsign,
                    "relu" => recurrent::Activation::Relu,
                    _ => return Err(invalid(function, O::StateActivationFunction)),
                };
            let gate_activation =
                match text(layer, kind, O::GateActivationFunction, "sigmoid", function)?
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "sigmoid" => recurrent::GateActivation::Sigmoid,
                    "hard-sigmoid" => recurrent::GateActivation::HardSigmoid,
                    _ => return Err(invalid(function, O::GateActivationFunction)),
                };
            let hidden_size = layer
                .properties
                .get("NumHiddenUnits")
                .ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: recurrent layer is missing NumHiddenUnits"),
                    )
                })
                .and_then(|value| positive_usize(value, function, "NumHiddenUnits"))?;
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
                .map(|v| numeric_scalar(v, function, "Epsilon"))
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
                .properties
                .get("Alpha")
                .map(|value| numeric_scalar(value, function, "Alpha"))
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
    use runmat_builtins::CONVOLUTION_1D_LAYER_CLASS;

    fn convolution(padding: Option<Value>) -> ObjectInstance {
        let mut layer = ObjectInstance::new(CONVOLUTION_1D_LAYER_CLASS.owned());
        if let Some(padding) = padding {
            layer.properties.insert("Padding".into(), padding);
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
}
