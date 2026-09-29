//! Host-known loss-input validation before provider effects.

use super::*;

fn host_payload(value: &Value, label: &'static str) -> BuiltinResult<Option<LossPayload>> {
    if matches!(value, Value::GpuTensor(_)) {
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
    rest: &[Value],
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
    let mut options = parse_name_values(
        rest[usize::from(positional_weights)..].to_vec(),
        "crossentropy",
    )?;
    let weights = if positional_weights {
        Some(&rest[0])
    } else {
        options.get("weights")
    };
    let mut host_weights = None;
    if let Some(value) = weights.filter(|value| !matches!(value, Value::GpuTensor(_))) {
        host_weights = Some(LossWeights::parse(value, "weights")?);
    }
    for key in options.keys() {
        if !matches!(
            key.as_str(),
            "classificationmode"
                | "targetcategories"
                | "reduction"
                | "normalizationfactor"
                | "dataformat"
                | "weightsformat"
                | "weights"
                | "mask"
        ) {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported option '{key}'"),
            ));
        }
    }
    for (name, value) in &options {
        if matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        match name.as_str() {
            "classificationmode" | "targetcategories" => {
                ClassificationMode::parse(value, "ClassificationMode")?;
            }
            "reduction" => {
                CrossentropyReduction::parse(value)?;
            }
            "normalizationfactor" => {
                NormalizationFactor::parse(value)?;
            }
            "dataformat" | "weightsformat" => {
                scalar_text(value, "crossentropy")?;
            }
            "mask" => {
                if let Some(shape) = prediction_shape.as_deref() {
                    let len = shape.iter().copied().product();
                    mask_values_for_layout(value, shape, len)?;
                }
            }
            "weights" => {}
            _ => unreachable!("validated option names"),
        }
    }
    if let Some(shape) = prediction_shape.as_deref() {
        let data_format = options
            .remove("dataformat")
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        let effective_format = data_format.as_deref().or_else(|| {
            prediction_payload
                .as_ref()
                .and_then(|payload| payload.format.as_deref())
        });
        validate_format_for_shape(effective_format, shape, "DataFormat")?;
        if let Some(weights) = host_weights {
            let weights_format = options
                .remove("weightsformat")
                .filter(|value| !matches!(value, Value::GpuTensor(_)))
                .map(|value| scalar_text(&value, "crossentropy"))
                .transpose()?
                .filter(|format| !format.is_empty());
            weights.materialize_for_layout(
                shape,
                shape.iter().copied().product(),
                effective_format,
                weights_format.as_deref(),
            )?;
        }
    }
    Ok(())
}
