use super::factor::{GpuLossFactor, GpuLossFactorParse};
use super::*;

pub(super) struct GpuCrossentropyOptions {
    pub(super) mode: ClassificationMode,
    pub(super) reduction: CrossentropyReduction,
    normalization: NormalizationFactor,
    data_format: Option<String>,
    pub(super) weights: Option<GpuLossFactor>,
    pub(super) mask: Option<GpuLossFactor>,
}

impl GpuCrossentropyOptions {
    pub(super) fn from_args(args: Vec<Value>, shape: &[usize]) -> BuiltinResult<Option<Self>> {
        let len = checked_len(shape)?;
        let mut args = args;
        let positional_weights = if args
            .first()
            .is_some_and(|value| !is_crossentropy_option_name(value))
        {
            Some(args.remove(0))
        } else {
            None
        };
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
        let weights_format = parsed
            .remove("weightsformat")
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        validate_format_for_shape(data_format.as_deref(), shape, "DataFormat")?;
        let weights_value = positional_weights.or_else(|| parsed.remove("weights"));
        let weights = match weights_value {
            Some(value) => match GpuLossFactor::weights(
                value,
                shape,
                len,
                data_format.as_deref(),
                weights_format.as_deref(),
            )? {
                GpuLossFactorParse::Supported(factor) => factor,
                GpuLossFactorParse::Unsupported => return Ok(None),
            },
            None => None,
        };
        let mask = match parsed.remove("mask") {
            Some(value) => match GpuLossFactor::mask(value, shape, len)? {
                GpuLossFactorParse::Supported(factor) => factor,
                GpuLossFactorParse::Unsupported => return Ok(None),
            },
            None => None,
        };
        let normalization = parsed
            .remove("normalizationfactor")
            .map(|value| NormalizationFactor::parse(&value))
            .transpose()?
            .unwrap_or_else(|| {
                if data_format.is_some() {
                    NormalizationFactor::BatchSize
                } else {
                    NormalizationFactor::AllElements
                }
            });
        if matches!(normalization, NormalizationFactor::MaskIncluded) {
            return Ok(None);
        }
        if !parsed.is_empty() {
            let first = parsed.keys().next().cloned().unwrap_or_default();
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
            NormalizationFactor::AllElements => Ok(len as f64),
            NormalizationFactor::BatchSize => {
                Ok(batch_size_for_shape(shape, self.data_format.as_deref()))
            }
            NormalizationFactor::MaskIncluded => Err(deep_learning_error(
                "crossentropy",
                "crossentropy: gpuArray MaskIncluded normalization requires host fallback",
            )),
            NormalizationFactor::None => Ok(1.0),
            NormalizationFactor::Scalar(value) => Ok(value),
        }
    }
}
