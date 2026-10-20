use super::*;
use std::collections::BTreeMap;

pub(super) type ClassificationMode = runmat_builtins::CrossentropyClassificationMode;
pub(super) type CrossentropyReduction = runmat_builtins::CrossentropyReductionMode;
pub(super) type NormalizationMode = runmat_builtins::CrossentropyNormalizationMode;

pub(super) fn parse_classification_mode(
    value: &Value,
    option: &str,
) -> BuiltinResult<ClassificationMode> {
    let text = scalar_text(value, "crossentropy")?
        .to_ascii_lowercase()
        .replace(['_', ' '], "-");
    ClassificationMode::from_name(&text).ok_or_else(|| {
        deep_learning_error(
            "crossentropy",
            format!("crossentropy: unsupported {option} '{text}'"),
        )
    })
}

pub(super) fn parse_reduction(value: &Value) -> BuiltinResult<CrossentropyReduction> {
    let text = scalar_text(value, "crossentropy")?.to_ascii_lowercase();
    CrossentropyReduction::from_name(&text).ok_or_else(|| {
        deep_learning_error(
            "crossentropy",
            format!("crossentropy: unsupported Reduction '{text}'"),
        )
    })
}

pub(super) fn provider_mode(mode: ClassificationMode) -> ProviderCrossentropyMode {
    match mode {
        ClassificationMode::SingleLabel => ProviderCrossentropyMode::SingleLabel,
        ClassificationMode::MultiLabel => ProviderCrossentropyMode::MultiLabel,
    }
}

pub(super) struct ParsedLossOptions {
    known: BTreeMap<OptionKey, Value>,
    unknown: BTreeMap<String, Value>,
}

impl ParsedLossOptions {
    pub(super) fn from_view(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: name-value options must be paired",
            ));
        }
        let mut known = BTreeMap::new();
        let mut unknown = BTreeMap::new();
        for pair in args.chunks_exact(2) {
            let name = scalar_text(&pair[0], "crossentropy")?.to_ascii_lowercase();
            let value = pair[1].clone();
            if let Some(key) = OptionKey::from_name(&name) {
                known.insert(key, value);
            } else {
                unknown.insert(name, value);
            }
        }
        Ok(Self { known, unknown })
    }

    pub(super) fn remove(&mut self, key: OptionKey) -> Option<Value> {
        self.known.remove(&key)
    }

    pub(super) fn get(&self, key: OptionKey) -> Option<&Value> {
        self.known.get(&key)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (OptionKey, &Value)> {
        self.known.iter().map(|(key, value)| (*key, value))
    }

    pub(super) fn unknown_name(&self) -> Option<&str> {
        self.unknown.keys().next().map(String::as_str)
    }

    pub(super) fn unsupported_name(&self) -> Option<String> {
        let known = self
            .known
            .keys()
            .map(|key| key.canonical_name().to_ascii_lowercase())
            .min();
        match (known, self.unknown.keys().next()) {
            (Some(known), Some(unknown)) if unknown < &known => Some(unknown.clone()),
            (Some(known), _) => Some(known),
            (None, Some(unknown)) => Some(unknown.clone()),
            (None, None) => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum NormalizationFactor {
    Named(NormalizationMode),
    Scalar(f64),
}

impl NormalizationFactor {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
                let text = scalar_text(value, "crossentropy")?
                    .to_ascii_lowercase()
                    .replace(['_', ' '], "-");
                match NormalizationMode::from_name(&text) {
                    Some(mode) => Ok(Self::Named(mode)),
                    None => Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: unsupported NormalizationFactor '{text}'"),
                    )),
                }
            }
            _ => {
                let scalar = numeric_scalar(value, "crossentropy", "NormalizationFactor")?;
                if scalar <= 0.0 {
                    return Err(deep_learning_error(
                        "crossentropy",
                        "crossentropy: NormalizationFactor must be positive",
                    ));
                }
                Ok(Self::Scalar(scalar))
            }
        }
    }
}

pub(super) struct CrossentropyOptions {
    pub(super) mode: ClassificationMode,
    pub(super) reduction: CrossentropyReduction,
    pub(super) normalization: NormalizationFactor,
    pub(super) data_format: Option<Layout>,
    pub(super) weights_format: Option<Layout>,
    pub(super) weights: Option<LossWeights>,
    pub(super) mask: Option<LossMask>,
}

impl CrossentropyOptions {
    pub(super) fn from_args(
        args: crate::call::FunctionArgumentView<'_>,
        predictions: &LossPayload,
        positional_weights: Option<LossWeights>,
    ) -> BuiltinResult<Self> {
        let mut parsed = ParsedLossOptions::from_view(args)?;
        let mode = parsed
            .remove(OptionKey::ClassificationMode)
            .or_else(|| parsed.remove(OptionKey::TargetCategories))
            .map(|value| parse_classification_mode(&value, "ClassificationMode"))
            .transpose()?
            .unwrap_or(ClassificationMode::SingleLabel);
        let reduction = parsed
            .remove(OptionKey::Reduction)
            .map(|value| parse_reduction(&value))
            .transpose()?
            .unwrap_or(CrossentropyReduction::Sum);
        let data_format = parsed
            .remove(OptionKey::DataFormat)
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?;
        let data_format =
            parse_loss_format(data_format.as_deref(), &predictions.shape, "DataFormat")?;
        let normalization = parsed
            .remove(OptionKey::NormalizationFactor)
            .map(|value| NormalizationFactor::parse(&value))
            .transpose()?
            .unwrap_or_else(|| {
                if predictions.format.is_some() || data_format.is_some() {
                    NormalizationFactor::Named(NormalizationMode::BatchSize)
                } else {
                    NormalizationFactor::Named(NormalizationMode::AllElements)
                }
            });
        let weights_format = parsed
            .remove(OptionKey::WeightsFormat)
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?;
        let named_weights = parsed
            .remove(OptionKey::Weights)
            .map(|value| LossWeights::parse(&value, "weights"))
            .transpose()?;
        let weights = positional_weights.or(named_weights);
        let mask = parsed
            .remove(OptionKey::Mask)
            .map(|value| LossMask::parse(&value, predictions))
            .transpose()?;
        if let Some(first) = parsed.unsupported_name() {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported option '{first}'"),
            ));
        }
        let weights_format = if let Some(weights) = &weights {
            parse_loss_format(weights_format.as_deref(), &weights.shape, "WeightsFormat")?
        } else {
            // With no weights the format is inert, but still admit its external text.
            None
        };
        Ok(Self {
            mode,
            reduction,
            normalization,
            data_format,
            weights_format,
            weights,
            mask,
        })
    }

    pub(super) fn mask_values(&self, predictions: &LossPayload) -> BuiltinResult<Vec<f64>> {
        match &self.mask {
            Some(mask) => Ok(mask.data.clone()),
            None => Ok(vec![1.0; predictions.data.len()]),
        }
    }

    pub(super) fn reduce_sum(
        &self,
        predictions: &LossPayload,
        weighted_losses: &[f64],
        mask: &[f64],
    ) -> BuiltinResult<f64> {
        match self.normalization {
            NormalizationFactor::Named(NormalizationMode::AllElements) => {
                Ok(weighted_losses.iter().sum::<f64>() / predictions.data.len() as f64)
            }
            NormalizationFactor::Named(NormalizationMode::BatchSize) => {
                Ok(weighted_losses.iter().sum::<f64>()
                    / batch_size(predictions, self.data_format.as_ref())?)
            }
            NormalizationFactor::Named(NormalizationMode::MaskIncluded) => reduce_mask_included(
                predictions,
                weighted_losses,
                mask,
                self.data_format.as_ref(),
            ),
            NormalizationFactor::Named(NormalizationMode::None) => {
                Ok(weighted_losses.iter().sum::<f64>())
            }
            NormalizationFactor::Scalar(value) => Ok(weighted_losses.iter().sum::<f64>() / value),
        }
    }
}
