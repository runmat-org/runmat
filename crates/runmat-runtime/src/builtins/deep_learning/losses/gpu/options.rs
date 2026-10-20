use super::factor::{GpuLossFactor, GpuLossFactorParse};
use super::*;

pub(super) struct GpuCrossentropyOptions {
    pub(super) mode: ClassificationMode,
    pub(super) reduction: CrossentropyReduction,
    normalization: NormalizationFactor,
    data_format: Option<Layout>,
    pub(super) weights: Option<GpuLossFactor>,
    pub(super) mask: Option<GpuLossFactor>,
}

impl GpuCrossentropyOptions {
    pub(super) fn from_args(
        args: crate::call::FunctionArgumentView<'_>,
        shape: &[usize],
    ) -> BuiltinResult<Option<Self>> {
        let len = checked_len(shape)?;
        let positional_weights = if args
            .first()
            .is_some_and(|value| !is_crossentropy_option_name(value))
        {
            Some(args.first().expect("positional loss weight").clone())
        } else {
            None
        };
        let option_start = usize::from(positional_weights.is_some());
        let mut parsed = ParsedLossOptions::from_view(
            args.range(option_start..)
                .expect("crossentropy option suffix"),
        )?;
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
        let data_format = parse_loss_format(data_format.as_deref(), shape, "DataFormat")?;
        let weights_format = parsed
            .remove(OptionKey::WeightsFormat)
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        let weights_value = positional_weights.or_else(|| parsed.remove(OptionKey::Weights));
        let weights = match weights_value {
            Some(value) => match GpuLossFactor::weights(
                value,
                shape,
                len,
                data_format.as_ref(),
                weights_format.as_deref(),
            )? {
                GpuLossFactorParse::Supported(factor) => factor,
                GpuLossFactorParse::Unsupported => return Ok(None),
            },
            None => None,
        };
        let mask = match parsed.remove(OptionKey::Mask) {
            Some(value) => match GpuLossFactor::mask(value, shape, len)? {
                GpuLossFactorParse::Supported(factor) => factor,
                GpuLossFactorParse::Unsupported => return Ok(None),
            },
            None => None,
        };
        let normalization = parsed
            .remove(OptionKey::NormalizationFactor)
            .map(|value| NormalizationFactor::parse(&value))
            .transpose()?
            .unwrap_or_else(|| {
                if data_format.is_some() {
                    NormalizationFactor::Named(NormalizationMode::BatchSize)
                } else {
                    NormalizationFactor::Named(NormalizationMode::AllElements)
                }
            });
        if matches!(
            normalization,
            NormalizationFactor::Named(NormalizationMode::MaskIncluded)
        ) {
            return Ok(None);
        }
        if let Some(first) = parsed.unsupported_name() {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported option '{first}'"),
            ));
        }
        Ok(Some(Self {
            mode,
            reduction,
            normalization,
            data_format,
            weights,
            mask,
        }))
    }

    pub(super) fn normalization_denominator(
        &self,
        len: usize,
        shape: &[usize],
    ) -> BuiltinResult<f64> {
        match self.normalization {
            NormalizationFactor::Named(NormalizationMode::AllElements) => Ok(len as f64),
            NormalizationFactor::Named(NormalizationMode::BatchSize) => {
                Ok(batch_size_for_shape(shape, self.data_format.as_ref()))
            }
            NormalizationFactor::Named(NormalizationMode::MaskIncluded) => {
                Err(deep_learning_error(
                    "crossentropy",
                    "crossentropy: gpuArray MaskIncluded normalization requires host fallback",
                ))
            }
            NormalizationFactor::Named(NormalizationMode::None) => Ok(1.0),
            NormalizationFactor::Scalar(value) => Ok(value),
        }
    }
}
