use super::*;
use runmat_accelerate_api::{ProviderExecutionError, ProviderPrecision};

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
    let len = checked_len(predictions.shape())?;
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
    let Some(provider) =
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(predictions)
    else {
        return Ok(None);
    };
    let Some(target_provider) =
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(targets)
    else {
        return Ok(None);
    };
    if !runmat_accelerate_api::same_provider_instance(provider, target_provider) {
        return Ok(None);
    }
    let precision = runmat_accelerate_api::handle_precision(predictions);
    if precision.is_none()
        || runmat_accelerate_api::handle_is_logical(predictions)
        || runmat_accelerate_api::handle_is_logical(targets)
        || runmat_accelerate_api::handle_precision(targets) != precision
    {
        return Ok(None);
    }
    validate_live(predictions)?;
    validate_live(targets)?;

    let mut temporary_inputs = Vec::new();
    let weights = match materialize_gpu_loss_factor(
        provider,
        options.weights.as_ref(),
        predictions.shape(),
        "weights",
        &mut temporary_inputs,
    ) {
        Ok(handle) => handle,
        Err(err) if runmat_accelerate_api::is_unsupported_provider_operation(&err) => {
            return Ok(None)
        }
        Err(err) => return Err(provider_error(err)),
    };
    let mask = match materialize_gpu_loss_factor(
        provider,
        options.mask.as_ref(),
        predictions.shape(),
        "mask",
        &mut temporary_inputs,
    ) {
        Ok(handle) => handle,
        Err(err) if runmat_accelerate_api::is_unsupported_provider_operation(&err) => {
            return Ok(None)
        }
        Err(err) => return Err(provider_error(err)),
    };

    for factor in [weights.as_ref(), mask.as_ref()].into_iter().flatten() {
        if runmat_accelerate_api::handle_storage(factor) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(factor)
            || runmat_accelerate_api::handle_precision(factor) != precision
            || crate::builtins::common::gpu_helpers::exact_provider_for_handle(factor)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            return Ok(None);
        }
        validate_live(factor)?;
    }
    for temporary in &temporary_inputs {
        validate_result_handle(
            temporary,
            predictions.shape(),
            precision,
            provider,
            &[predictions, targets],
        )?;
    }

    let result = match provider.crossentropy_terms(&ProviderCrossentropyRequest {
        predictions,
        targets,
        weights: weights.as_ref(),
        mask: mask.as_ref(),
        mode: options.mode.into_provider(),
    }) {
        Ok(result) => result,
        Err(err) if runmat_accelerate_api::is_unsupported_provider_operation(&err) => {
            return Ok(None)
        }
        Err(err) => return Err(provider_error(err)),
    };
    let factor_handles = [weights.as_ref(), mask.as_ref()];
    let borrowed = [
        Some(predictions),
        Some(targets),
        factor_handles[0],
        factor_handles[1],
    ];
    validate_result_handle(
        &result.losses,
        predictions.shape(),
        precision,
        provider,
        &borrowed.into_iter().flatten().collect::<Vec<_>>(),
    )?;

    match options.reduction {
        CrossentropyReduction::None => Ok(Some(
            crate::builtins::common::gpu_helpers::resident_gpu_value(result.losses)?,
        )),
        CrossentropyReduction::Sum => {
            let sum = match provider.reduce_sum(&result.losses).await {
                Ok(sum) => sum,
                Err(err) if runmat_accelerate_api::is_unsupported_provider_operation(&err) => {
                    return Ok(None)
                }
                Err(err) => return Err(provider_error(err)),
            };
            validate_result_handle(
                &sum,
                &[1, 1],
                precision,
                provider,
                &[predictions, targets, &result.losses],
            )?;
            let scalar = match provider.read_scalar(&sum, 0) {
                Ok(value) => value / options.normalization_denominator(len, predictions.shape())?,
                Err(err) if runmat_accelerate_api::is_unsupported_provider_operation(&err) => {
                    return Ok(None)
                }
                Err(err) => return Err(provider_error(err)),
            };
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
            let expected = shape
                .iter()
                .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
                .ok_or_else(|| {
                    anyhow::anyhow!("{label} element count exceeds addressable storage")
                })?;
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

fn provider_error(error: anyhow::Error) -> crate::RuntimeError {
    let source = ProviderExecutionError::from(error);
    crate::runtime_error::with_provider_execution_source(
        deep_learning_error("crossentropy", format!("crossentropy: {source}")),
        source,
    )
}

fn checked_len(shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| {
            deep_learning_error(
                "crossentropy",
                "crossentropy: tensor element count exceeds addressable storage",
            )
        })
}

fn validate_live(handle: &GpuTensorHandle) -> BuiltinResult<()> {
    handle.allocation_lease().validate_live().map_err(|source| {
        crate::build_runtime_error("crossentropy: provider input allocation is retired")
            .with_builtin("crossentropy")
            .with_identifier("RunMat:deepLearning:InvalidInput")
            .with_source(source)
            .build()
    })
}

fn validate_result_handle(
    result: &GpuTensorHandle,
    shape: &[usize],
    precision: Option<ProviderPrecision>,
    provider: &dyn AccelProvider,
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<()> {
    if result.shape() != shape
        || runmat_accelerate_api::handle_storage(result) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(result)
        || runmat_accelerate_api::handle_precision(result) != precision
        || crate::builtins::common::gpu_helpers::exact_provider_for_handle(result)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || borrowed
            .iter()
            .any(|input| result.allocation_identity() == input.allocation_identity())
    {
        return Err(deep_learning_error("crossentropy", "crossentropy: provider returned a loss tensor with the wrong owner, shape, class, or allocation"));
    }
    validate_live(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operational_error_with_unsupported_words_does_not_become_fallback() {
        let error = anyhow::anyhow!("unsupported operation encountered a device failure");
        assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
            &error
        ));
        let error = provider_error(error);
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(error.source.is_some());
    }

    #[test]
    fn result_validation_rejects_alias_and_shape_before_publication() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.8, 0.2],
                    shape: &[1, 2],
                })
                .unwrap();
            let result = provider
                .upload(&HostTensorView {
                    data: &[0.1, 0.2],
                    shape: &[1, 2],
                })
                .unwrap();
            let precision = runmat_accelerate_api::handle_precision(&input);
            assert!(
                validate_result_handle(&input, &[1, 2], precision, provider, &[&input]).is_err()
            );
            assert!(
                validate_result_handle(&result, &[2, 1], precision, provider, &[&input]).is_err()
            );
            validate_result_handle(&result, &[1, 2], precision, provider, &[&input]).unwrap();
        });
    }
}
