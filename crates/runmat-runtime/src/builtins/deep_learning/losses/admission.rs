//! Host-known loss-input validation before provider effects.

use super::*;

fn host_payload(value: &Value, label: &'static str) -> BuiltinResult<Option<LossPayload>> {
    if matches!(value, Value::GpuTensor(_))
        || crate::builtins::common::validation::value_contains_explicit_gpu(value)
    {
        Ok(None)
    } else {
        LossPayload::parse(value, label).map(Some)
    }
}

fn known_shape(value: &Value, payload: Option<&LossPayload>) -> Option<Vec<usize>> {
    payload
        .map(|payload| payload.shape.clone())
        .or_else(|| match value {
            Value::GpuTensor(handle) => Some(handle.shape().to_vec()),
            _ => None,
        })
}

pub(super) fn preflight_host_decidable(
    predictions: &Value,
    targets: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let prediction_payload = host_payload(predictions, "predictions")?;
    let target_payload = host_payload(targets, "targets")?;
    if target_payload.as_ref().is_some_and(|payload| {
        payload
            .data
            .iter()
            .any(|target| !(0.0..=1.0).contains(target))
    }) {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: targets must be probabilities in the range [0, 1]",
        ));
    }
    let prediction_shape = known_shape(predictions, prediction_payload.as_ref());
    let target_shape = known_shape(targets, target_payload.as_ref());
    if let (Some(prediction_shape), Some(target_shape)) = (&prediction_shape, &target_shape) {
        if prediction_shape != target_shape {
            return Err(deep_learning_error(
                "crossentropy",
                format!(
                    "crossentropy: targets must match prediction shape {prediction_shape:?}, got {target_shape:?}"
                ),
            ));
        }
        if prediction_shape.contains(&0) {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: predictions and targets must not be empty",
            ));
        }
    }
    if let (Some(predictions), Some(targets)) = (&prediction_payload, &target_payload) {
        require_loss_payload_compatible(predictions, targets, None)?;
    }

    let positional_weights = rest
        .first()
        .is_some_and(|value| !is_crossentropy_option_name(value));
    let mut options = ParsedLossOptions::from_view(
        rest.range(usize::from(positional_weights)..)
            .expect("crossentropy option suffix"),
    )?;
    let weights = if positional_weights {
        Some(&rest[0])
    } else {
        options.get(OptionKey::Weights)
    };
    let mut host_weights = None;
    if let Some(value) = weights.filter(|value| !matches!(value, Value::GpuTensor(_))) {
        host_weights = Some(LossWeights::parse(value, "weights")?);
    }
    if let Some(key) = options.unknown_name() {
        return Err(deep_learning_error(
            "crossentropy",
            format!("crossentropy: unsupported option '{key}'"),
        ));
    }
    for (name, value) in options.iter() {
        if matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        match name {
            OptionKey::ClassificationMode | OptionKey::TargetCategories => {
                parse_classification_mode(value, "ClassificationMode")?;
            }
            OptionKey::Reduction => {
                parse_reduction(value)?;
            }
            OptionKey::NormalizationFactor => {
                NormalizationFactor::parse(value)?;
            }
            OptionKey::DataFormat | OptionKey::WeightsFormat => {
                scalar_text(value, "crossentropy")?;
            }
            OptionKey::Mask => {
                if let Some(shape) = prediction_shape.as_deref() {
                    let len = shape.iter().copied().product();
                    mask_values_for_layout(value, shape, len)?;
                }
            }
            OptionKey::Weights => {}
        }
    }
    if let Some(shape) = prediction_shape.as_deref() {
        let data_format = options
            .remove(OptionKey::DataFormat)
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?;
        let data_format = parse_loss_format(data_format.as_deref(), shape, "DataFormat")?;
        let effective_format = data_format.as_ref().or_else(|| {
            prediction_payload
                .as_ref()
                .and_then(|payload| payload.format.as_ref())
        });
        if let Some(weights) = host_weights {
            let weights_format = options
                .remove(OptionKey::WeightsFormat)
                .filter(|value| !matches!(value, Value::GpuTensor(_)))
                .map(|value| scalar_text(&value, "crossentropy"))
                .transpose()?;
            let weights_format =
                parse_loss_format(weights_format.as_deref(), &weights.shape, "WeightsFormat")?;
            weights.materialize_for_layout(
                shape,
                shape.iter().copied().product(),
                effective_format,
                weights_format.as_ref(),
            )?;
        }
    }
    Ok(())
}
