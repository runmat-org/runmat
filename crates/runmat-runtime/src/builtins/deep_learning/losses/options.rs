use super::*;

#[derive(Clone, Copy)]
pub(super) enum ClassificationMode {
    SingleLabel,
    MultiLabel,
}

impl ClassificationMode {
    pub(super) fn parse(value: &Value, option: &str) -> BuiltinResult<Self> {
        let text = scalar_text(value, "crossentropy")?
            .to_ascii_lowercase()
            .replace(['_', ' '], "-");
        match text.as_str() {
            "single-label" | "singlelabel" | "exclusive" => Ok(Self::SingleLabel),
            "multi-label" | "multilabel" | "independent" => Ok(Self::MultiLabel),
            other => Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported {option} '{other}'"),
            )),
        }
    }

    pub(super) fn into_provider(self) -> ProviderCrossentropyMode {
        match self {
            Self::SingleLabel => ProviderCrossentropyMode::SingleLabel,
            Self::MultiLabel => ProviderCrossentropyMode::MultiLabel,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum CrossentropyReduction {
    Sum,
    None,
}

impl CrossentropyReduction {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        match scalar_text(value, "crossentropy")?
            .to_ascii_lowercase()
            .as_str()
        {
            "sum" => Ok(Self::Sum),
            "none" => Ok(Self::None),
            other => Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported Reduction '{other}'"),
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum NormalizationFactor {
    AllElements,
    BatchSize,
    MaskIncluded,
    None,
    Scalar(f64),
}

impl NormalizationFactor {
    pub(super) fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
                match scalar_text(value, "crossentropy")?
                    .to_ascii_lowercase()
                    .replace(['_', ' '], "-")
                    .as_str()
                {
                    "all-elements" | "all" => Ok(Self::AllElements),
                    "batch-size" | "batch" => Ok(Self::BatchSize),
                    "mask-included" | "mask" => Ok(Self::MaskIncluded),
                    "none" => Ok(Self::None),
                    other => Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: unsupported NormalizationFactor '{other}'"),
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
    pub(super) data_format: Option<String>,
    pub(super) weights_format: Option<String>,
    pub(super) weights: Option<LossWeights>,
    pub(super) mask: Option<LossMask>,
}

impl CrossentropyOptions {
    pub(super) fn from_args(args: Vec<Value>, predictions: &LossPayload) -> BuiltinResult<Self> {
        let mut parsed = parse_name_values(args, "crossentropy")?;
        let mode = parsed
            .remove("classificationmode")
            .or_else(|| parsed.remove("targetcategories"))
            .map(|value| ClassificationMode::parse(&value, "ClassificationMode"))
            .transpose()?
            .unwrap_or(ClassificationMode::SingleLabel);
        let reduction = parsed
            .remove("reduction")
            .map(|value| CrossentropyReduction::parse(&value))
            .transpose()?
            .unwrap_or(CrossentropyReduction::Sum);
        let data_format = parsed
            .remove("dataformat")
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        let normalization = parsed
            .remove("normalizationfactor")
            .map(|value| NormalizationFactor::parse(&value))
            .transpose()?
            .unwrap_or_else(|| {
                if predictions.format.is_some() || data_format.is_some() {
                    NormalizationFactor::BatchSize
                } else {
                    NormalizationFactor::AllElements
                }
            });
        let weights_format = parsed
            .remove("weightsformat")
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        let weights = parsed
            .remove("weights")
            .map(|value| LossWeights::parse(&value, "weights"))
            .transpose()?;
        let mask = parsed
            .remove("mask")
            .map(|value| LossMask::parse(&value, predictions))
            .transpose()?;
        if !parsed.is_empty() {
            let first = parsed.keys().next().cloned().unwrap_or_default();
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported option '{first}'"),
            ));
        }
        validate_format_for_shape(
            predictions
                .format_or_default(data_format.as_deref())
                .as_deref(),
            &predictions.shape,
            "DataFormat",
        )?;
        if let Some(weights_format) = weights_format.as_deref() {
            if let Some(weights) = &weights {
                validate_format_for_shape(Some(weights_format), &weights.shape, "WeightsFormat")?;
            }
        }
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
            NormalizationFactor::AllElements => {
                Ok(weighted_losses.iter().sum::<f64>() / predictions.data.len() as f64)
            }
            NormalizationFactor::BatchSize => Ok(weighted_losses.iter().sum::<f64>()
                / batch_size(predictions, self.data_format.as_deref())?),
            NormalizationFactor::MaskIncluded => reduce_mask_included(
                predictions,
                weighted_losses,
                mask,
                self.data_format.as_deref(),
            ),
            NormalizationFactor::None => Ok(weighted_losses.iter().sum::<f64>()),
            NormalizationFactor::Scalar(value) => Ok(weighted_losses.iter().sum::<f64>() / value),
        }
    }
}
