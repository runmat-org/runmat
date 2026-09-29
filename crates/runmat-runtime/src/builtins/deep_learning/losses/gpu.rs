use super::*;

pub(super) async fn try_crossentropy_gpu(
    predictions: &Value,
    targets: &Value,
    rest: &[Value],
) -> BuiltinResult<Option<Value>> {
    let (Value::GpuTensor(predictions), Value::GpuTensor(targets)) = (predictions, targets) else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(predictions) == GpuTensorStorage::ComplexInterleaved
        || runmat_accelerate_api::handle_storage(targets) == GpuTensorStorage::ComplexInterleaved
    {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: complex gpuArray inputs are not supported",
        ));
    }
    if predictions.shape() != targets.shape() {
        return Err(deep_learning_error(
            "crossentropy",
            format!(
                "crossentropy: targets must match prediction shape {:?}, got {:?}",
                predictions.shape(),
                targets.shape()
            ),
        ));
    }
    let len = predictions.shape().iter().copied().product::<usize>();
    if len == 0 {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must not be empty",
        ));
    }

    let Some(options) = GpuCrossentropyOptions::from_args(rest.to_vec(), predictions.shape())?
    else {
        return Ok(None);
    };
    let Some(provider) = runmat_accelerate_api::provider_for_handle(predictions) else {
        return Ok(None);
    };
    let Some(target_provider) = runmat_accelerate_api::provider_for_handle(targets) else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(provider, target_provider) {
        return Ok(None);
    }

    let mut temporary_inputs = Vec::new();
    let weights = match materialize_gpu_loss_factor(
        provider,
        options.weights.as_ref(),
        predictions.shape(),
        "weights",
        &mut temporary_inputs,
    ) {
        Ok(handle) => handle,
        Err(err) if provider_is_unsupported(&err) => return Ok(None),
        Err(err) => {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {err}"),
            ))
        }
    };
    let mask = match materialize_gpu_loss_factor(
        provider,
        options.mask.as_ref(),
        predictions.shape(),
        "mask",
        &mut temporary_inputs,
    ) {
        Ok(handle) => handle,
        Err(err) if provider_is_unsupported(&err) => {
            drop(temporary_inputs);
            return Ok(None);
        }
        Err(err) => {
            drop(temporary_inputs);
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {err}"),
            ));
        }
    };

    let result = match provider.crossentropy_terms(&ProviderCrossentropyRequest {
        predictions,
        targets,
        weights: weights.as_ref(),
        mask: mask.as_ref(),
        mode: options.mode.into_provider(),
    }) {
        Ok(result) => result,
        Err(err) if provider_is_unsupported(&err) => {
            drop(temporary_inputs);
            return Ok(None);
        }
        Err(err) => {
            drop(temporary_inputs);
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {err}"),
            ));
        }
    };

    match options.reduction {
        CrossentropyReduction::None => {
            drop(temporary_inputs);
            Ok(Some(
                crate::builtins::common::gpu_helpers::resident_gpu_value(result.losses)?,
            ))
        }
        CrossentropyReduction::Sum => {
            let sum = match provider.reduce_sum(&result.losses).await {
                Ok(sum) => sum,
                Err(err) if provider_is_unsupported(&err) => {
                    drop(temporary_inputs);
                    return Ok(None);
                }
                Err(err) => {
                    drop(temporary_inputs);
                    return Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: {err}"),
                    ));
                }
            };
            let scalar = match provider.read_scalar(&sum, 0) {
                Ok(value) => value / options.normalization_denominator(len, predictions.shape())?,
                Err(err) if provider_is_unsupported(&err) => {
                    drop(temporary_inputs);
                    return Ok(None);
                }
                Err(err) => {
                    drop(temporary_inputs);
                    return Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: {err}"),
                    ));
                }
            };
            drop(temporary_inputs);
            if !scalar.is_finite() {
                return Err(deep_learning_error(
                    "crossentropy",
                    "crossentropy: reduction produced a non-finite loss",
                ));
            }
            Ok(Some(Value::Num(scalar)))
        }
    }
}

struct GpuCrossentropyOptions {
    mode: ClassificationMode,
    reduction: CrossentropyReduction,
    normalization: NormalizationFactor,
    data_format: Option<String>,
    weights: Option<GpuLossFactor>,
    mask: Option<GpuLossFactor>,
}

impl GpuCrossentropyOptions {
    fn from_args(args: Vec<Value>, shape: &[usize]) -> BuiltinResult<Option<Self>> {
        let len = shape.iter().copied().product::<usize>();
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

    fn normalization_denominator(&self, len: usize, shape: &[usize]) -> BuiltinResult<f64> {
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

enum GpuLossFactor {
    Host(Vec<f64>),
    Resident(GpuTensorHandle),
}

enum GpuLossFactorParse {
    Supported(Option<GpuLossFactor>),
    Unsupported,
}

impl GpuLossFactor {
    fn weights(
        value: Value,
        shape: &[usize],
        len: usize,
        data_format: Option<&str>,
        weights_format: Option<&str>,
    ) -> BuiltinResult<GpuLossFactorParse> {
        match value {
            Value::GpuTensor(handle) => {
                if weights_format.is_some()
                    || runmat_accelerate_api::handle_storage(&handle)
                        == GpuTensorStorage::ComplexInterleaved
                    || handle.shape() != shape
                {
                    return Ok(GpuLossFactorParse::Unsupported);
                }
                Ok(GpuLossFactorParse::Supported(Some(Self::Resident(handle))))
            }
            other => {
                let weights = LossWeights::parse(&other, "weights")?;
                let materialized =
                    weights.materialize_for_layout(shape, len, data_format, weights_format)?;
                Ok(GpuLossFactorParse::Supported(Some(Self::Host(
                    materialized,
                ))))
            }
        }
    }

    fn mask(value: Value, shape: &[usize], len: usize) -> BuiltinResult<GpuLossFactorParse> {
        match value {
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_storage(&handle)
                    == GpuTensorStorage::ComplexInterleaved
                    || handle.shape() != shape
                {
                    return Ok(GpuLossFactorParse::Unsupported);
                }
                Ok(GpuLossFactorParse::Supported(Some(Self::Resident(handle))))
            }
            other => Ok(GpuLossFactorParse::Supported(Some(Self::Host(
                mask_values_for_layout(&other, shape, len)?,
            )))),
        }
    }
}

fn materialize_gpu_loss_factor(
    provider: &dyn AccelProvider,
    factor: Option<&GpuLossFactor>,
    shape: &[usize],
    label: &'static str,
    temporary_inputs: &mut Vec<GpuTensorHandle>,
) -> anyhow::Result<Option<GpuTensorHandle>> {
    match factor {
        Some(GpuLossFactor::Resident(handle)) => Ok(Some(handle.clone())),
        Some(GpuLossFactor::Host(data)) => {
            let expected = shape.iter().copied().product::<usize>();
            anyhow::ensure!(
                data.len() == expected,
                "{label} length does not match prediction shape"
            );
            let handle = provider.upload(&HostTensorView { data, shape })?;
            temporary_inputs.push(handle.clone());
            Ok(Some(handle))
        }
        None => Ok(None),
    }
}

fn provider_is_unsupported(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("not supported") || message.contains("unsupported")
}
