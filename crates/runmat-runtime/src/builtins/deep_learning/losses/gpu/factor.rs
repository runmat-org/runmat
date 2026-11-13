use super::*;
use runmat_accelerate_api::{NumericUploadError, ProviderPrecision};

pub(super) enum GpuLossFactor {
    Host(Vec<f64>),
    Resident(GpuTensorHandle),
}

pub(super) enum GpuLossFactorParse {
    Supported(Option<GpuLossFactor>),
    Unsupported,
}

impl GpuLossFactor {
    pub(super) fn weights(
        value: Value,
        shape: &[usize],
        len: usize,
        data_format: Option<&Layout>,
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
                let weights_format =
                    parse_loss_format(weights_format, &weights.shape, "WeightsFormat")?;
                let materialized = weights.materialize_for_layout(
                    shape,
                    len,
                    data_format,
                    weights_format.as_ref(),
                )?;
                Ok(GpuLossFactorParse::Supported(Some(Self::Host(
                    materialized,
                ))))
            }
        }
    }

    pub(super) fn mask(
        value: Value,
        shape: &[usize],
        len: usize,
    ) -> BuiltinResult<GpuLossFactorParse> {
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

pub(super) enum MaterializedFactor {
    Ready(Option<GpuTensorHandle>),
    Unsupported,
}

pub(super) fn materialize_gpu_loss_factor(
    provider: &dyn AccelProvider,
    factor: Option<GpuLossFactor>,
    shape: &[usize],
    precision: ProviderPrecision,
    label: &'static str,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<MaterializedFactor> {
    match factor {
        Some(GpuLossFactor::Resident(handle)) => Ok(MaterializedFactor::Ready(Some(handle))),
        Some(GpuLossFactor::Host(data)) => {
            let dtype = match precision {
                ProviderPrecision::F32 => NumericDType::F32,
                ProviderPrecision::F64 => NumericDType::F64,
            };
            let tensor = Tensor::new_with_dtype(data, shape.to_vec(), dtype).map_err(|error| {
                deep_learning_error(
                    "crossentropy",
                    format!("crossentropy: {label} cannot be represented with prediction shape: {error}"),
                )
            })?;
            if !(0..tensor.len()).all(|index| {
                tensor
                    .numeric_value_at(index)
                    .is_some_and(runmat_value::NumericScalar::is_finite)
            }) {
                return Err(deep_learning_error(
                    "crossentropy",
                    format!("crossentropy: {label} is not finite in the prediction precision"),
                ));
            }
            let handle = match crate::builtins::common::gpu_helpers::upload_tensor_protecting_typed(
                provider, &tensor, protected,
            ) {
                Ok(handle) => handle,
                Err(NumericUploadError::Provider(error)) if error.is_unsupported() => {
                    return Ok(MaterializedFactor::Unsupported);
                }
                Err(NumericUploadError::Provider(error)) => return Err(provider_error(error)),
                Err(error) => {
                    return Err(crate::build_runtime_error(format!(
                        "crossentropy: invalid {label} upload: {error}"
                    ))
                    .with_builtin("crossentropy")
                    .with_identifier("RunMat:deepLearning:InvalidInput")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build());
                }
            };
            validate_result_handle(&handle, shape, Some(precision), provider, protected)?;
            Ok(MaterializedFactor::Ready(Some(handle)))
        }
        None => Ok(MaterializedFactor::Ready(None)),
    }
}
