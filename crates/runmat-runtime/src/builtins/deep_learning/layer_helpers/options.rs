use super::*;
use runmat_builtins::{
    DeepLearningConvolutionPaddingMode, DeepLearningConvolutionPaddingValueMode,
    DeepLearningDenseOperationDimension, DeepLearningNormalizationAxes,
    DeepLearningRecurrentGateActivation, DeepLearningRecurrentOutputMode,
    DeepLearningRecurrentStateActivation,
};

/// Materialize only structural controls consumed on the host. Data, state,
/// learnables and callable captures retain their existing storage and owner.
pub(in crate::builtins::deep_learning) fn requires_host_control(
    option: DeepLearningLayerOption,
) -> bool {
    use DeepLearningLayerOption as O;
    match option {
        O::Stride
        | O::DilationFactor
        | O::MinLength
        | O::InputSize
        | O::NumChannels
        | O::OperationDimension
        | O::Epsilon
        | O::Padding
        | O::PaddingValue
        | O::WeightLearnRateFactor
        | O::WeightL2Factor
        | O::BiasLearnRateFactor
        | O::BiasL2Factor
        | O::InputWeightsLearnRateFactor
        | O::RecurrentWeightsLearnRateFactor
        | O::InputWeightsL2Factor
        | O::RecurrentWeightsL2Factor
        | O::ScaleLearnRateFactor
        | O::OffsetLearnRateFactor
        | O::ScaleL2Factor
        | O::OffsetL2Factor
        | O::HasStateInputs
        | O::HasStateOutputs
        | O::SplitComplexInputs => true,
        O::Name
        | O::Description
        | O::OutputMode
        | O::NormalizationDimension
        | O::Normalization
        | O::StateActivationFunction
        | O::GateActivationFunction
        | O::WeightsInitializer
        | O::BiasInitializer
        | O::InputWeightsInitializer
        | O::RecurrentWeightsInitializer
        | O::ScaleInitializer
        | O::OffsetInitializer
        | O::Mean
        | O::StandardDeviation
        | O::Min
        | O::Max
        | O::Weights
        | O::Bias
        | O::InputLearnables
        | O::OutputLearnables
        | O::CellState
        | O::HiddenState
        | O::InputWeights
        | O::RecurrentWeights
        | O::Scale
        | O::Offset
        | O::Classes
        | O::ClassWeights
        | O::ResponseNames => false,
    }
}

pub(in crate::builtins::deep_learning) fn validate(
    kind: DeepLearningLayerKind,
    option: DeepLearningLayerOption,
    value: &Value,
    function: &'static str,
) -> BuiltinResult<()> {
    use DeepLearningLayerOption as O;
    let label = option.canonical_name();
    match option {
        O::Name | O::Description => {
            scalar_text(value, function)?;
        }
        O::OutputMode => {
            let text = scalar_text(value, function)?;
            if DeepLearningRecurrentOutputMode::from_name(&text).is_none() {
                return Err(invalid_value(function, label));
            }
        }
        O::NormalizationDimension => {
            text_choice(
                value,
                function,
                label,
                option.text_values_for(kind).unwrap(),
            )?;
        }
        O::Normalization => {
            if !is_callable(value) {
                text_choice(
                    value,
                    function,
                    label,
                    option.text_values_for(kind).unwrap(),
                )?;
            }
        }
        O::StateActivationFunction => {
            let text = scalar_text(value, function)?;
            if DeepLearningRecurrentStateActivation::from_name(&text).is_none() {
                return Err(invalid_value(function, label));
            }
        }
        O::GateActivationFunction => {
            let text = scalar_text(value, function)?;
            if DeepLearningRecurrentGateActivation::from_name(&text).is_none() {
                return Err(invalid_value(function, label));
            }
        }
        O::Padding => validate_padding(value, function)?,
        O::Stride | O::DilationFactor | O::MinLength => {
            positive_scalar_or_deferred(value, function, label)?;
        }
        O::InputSize | O::NumChannels => {
            if !scalar_text(value, function).is_ok_and(|text| text.eq_ignore_ascii_case("auto")) {
                positive_scalar_or_deferred(value, function, label)?;
            }
        }
        O::OperationDimension => {
            if let Ok(text) = scalar_text(value, function) {
                let valid = match kind {
                    DeepLearningLayerKind::LayerNormalization => {
                        DeepLearningNormalizationAxes::from_name(&text).is_some()
                    }
                    DeepLearningLayerKind::FullyConnected => {
                        DeepLearningDenseOperationDimension::from_name(&text).is_some()
                    }
                    _ => false,
                };
                if !valid {
                    return Err(invalid_value(function, label));
                }
            } else {
                positive_scalar_or_deferred(value, function, label)?;
            }
        }
        O::Epsilon => finite_scalar_or_deferred(value, function, label, Some(0.0), true)?,
        O::WeightLearnRateFactor
        | O::WeightL2Factor
        | O::BiasLearnRateFactor
        | O::BiasL2Factor
        | O::InputWeightsLearnRateFactor
        | O::RecurrentWeightsLearnRateFactor
        | O::InputWeightsL2Factor
        | O::RecurrentWeightsL2Factor
        | O::ScaleLearnRateFactor
        | O::OffsetLearnRateFactor
        | O::ScaleL2Factor
        | O::OffsetL2Factor => {
            if kind.recurrent_gate_count().is_some() && option.is_recurrent_gate_factor() {
                recurrent_factor(value, kind, function, label)?;
            } else {
                finite_scalar_or_deferred(value, function, label, Some(0.0), false)?;
            }
        }
        O::HasStateInputs | O::HasStateOutputs | O::SplitComplexInputs => {
            if let Value::GpuTensor(handle) = value {
                if !handle.shape().iter().all(|dimension| *dimension == 1) {
                    return Err(invalid_value(function, label));
                }
            } else {
                logical_scalar(value, function, label)?;
            }
        }
        O::WeightsInitializer
        | O::BiasInitializer
        | O::InputWeightsInitializer
        | O::RecurrentWeightsInitializer
        | O::ScaleInitializer
        | O::OffsetInitializer => {
            if !is_callable(value) && scalar_text(value, function).is_err() {
                return Err(invalid_value(function, label));
            }
        }
        O::PaddingValue => {
            if let Ok(text) = scalar_text(value, function) {
                if DeepLearningConvolutionPaddingValueMode::from_name(&text).is_none() {
                    return Err(invalid_value(function, label));
                }
            } else {
                finite_scalar_or_deferred(value, function, label, None, false)?;
            }
        }
        // Values whose class, shape, cross-property constraints, and callbacks
        // belong to the typed network plan are preserved, not interpreted as
        // evidence that a numerical execution route already exists.
        O::Mean
        | O::StandardDeviation
        | O::Min
        | O::Max
        | O::Weights
        | O::Bias
        | O::InputLearnables
        | O::OutputLearnables
        | O::CellState
        | O::HiddenState
        | O::InputWeights
        | O::RecurrentWeights
        | O::Scale
        | O::Offset
        | O::Classes
        | O::ClassWeights
        | O::ResponseNames => {}
    }
    Ok(())
}

fn invalid_value(function: &'static str, label: &str) -> crate::RuntimeError {
    deep_learning_error(
        function,
        format!("{function}: invalid {label} option value"),
    )
}

fn text_choice(
    value: &Value,
    function: &'static str,
    label: &str,
    choices: &[&str],
) -> BuiltinResult<()> {
    let text = scalar_text(value, function)?;
    if choices
        .iter()
        .any(|choice| text.eq_ignore_ascii_case(choice))
    {
        Ok(())
    } else {
        Err(invalid_value(function, label))
    }
}

pub(in crate::builtins::deep_learning) fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

fn positive_scalar_or_deferred(
    value: &Value,
    function: &'static str,
    label: &str,
) -> BuiltinResult<()> {
    if let Value::GpuTensor(handle) = value {
        return if handle.shape().iter().all(|dimension| *dimension == 1) {
            Ok(())
        } else {
            Err(invalid_value(function, label))
        };
    }
    positive_usize(value, function, label).map(|_| ())
}

fn finite_scalar_or_deferred(
    value: &Value,
    function: &'static str,
    label: &str,
    minimum: Option<f64>,
    exclusive: bool,
) -> BuiltinResult<()> {
    if let Value::GpuTensor(handle) = value {
        return if handle.shape().iter().all(|dimension| *dimension == 1) {
            Ok(())
        } else {
            Err(invalid_value(function, label))
        };
    }
    let number = numeric_scalar(value, function, label)?;
    if minimum.is_some_and(|bound| (exclusive && number <= bound) || (!exclusive && number < bound))
    {
        Err(invalid_value(function, label))
    } else {
        Ok(())
    }
}

fn recurrent_factor(
    value: &Value,
    kind: DeepLearningLayerKind,
    function: &'static str,
    label: &str,
) -> BuiltinResult<()> {
    let count = kind
        .recurrent_gate_count()
        .ok_or_else(|| invalid_value(function, label))?;
    if let Value::Tensor(tensor) = value {
        if tensor.shape() == [1, count] {
            let values = numeric_values(value, function, label)?;
            return if values
                .iter()
                .all(|factor| factor.is_finite() && *factor >= 0.0)
            {
                Ok(())
            } else {
                Err(invalid_value(function, label))
            };
        }
    }
    finite_scalar_or_deferred(value, function, label, Some(0.0), false)
}

fn validate_padding(value: &Value, function: &'static str) -> BuiltinResult<()> {
    if let Ok(text) = scalar_text(value, function) {
        return if DeepLearningConvolutionPaddingMode::from_name(&text).is_some() {
            Ok(())
        } else {
            Err(invalid_value(function, "Padding"))
        };
    }
    if let Value::GpuTensor(handle) = value {
        return if handle
            .shape()
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .is_some_and(|count| (1..=2).contains(&count))
        {
            Ok(())
        } else {
            Err(invalid_value(function, "Padding"))
        };
    }
    let valid = match value {
        Value::Num(number) => valid_padding_number(*number),
        Value::Int(integer) => integer.try_to_usize().is_some(),
        Value::Tensor(tensor) if (1..=2).contains(&tensor.host_buffer().len()) => {
            if let Some(storage) = tensor.integer_storage() {
                (0..tensor.host_buffer().len()).all(|index| {
                    storage
                        .value_at(index)
                        .is_some_and(|value| value.try_to_usize().is_some())
                })
            } else {
                tensor
                    .materialize_f64()
                    .into_iter()
                    .all(valid_padding_number)
            }
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid_value(function, "Padding"))
    }
}

fn valid_padding_number(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && value.fract() == 0.0
}

#[cfg(test)]
mod tests;
