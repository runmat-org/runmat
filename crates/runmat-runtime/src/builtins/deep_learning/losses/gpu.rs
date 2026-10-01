use super::*;

mod factor;
mod options;
mod validation;

use factor::{materialize_gpu_loss_factor, GpuLossFactor, MaterializedFactor};
use options::GpuCrossentropyOptions;
use validation::{checked_len, provider_error, validate_live, validate_result_handle};

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

    let Some(mut options) = GpuCrossentropyOptions::from_args(rest.to_vec(), predictions.shape())?
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

    for factor in [options.weights.as_ref(), options.mask.as_ref()]
        .into_iter()
        .flatten()
    {
        let GpuLossFactor::Resident(handle) = factor else {
            continue;
        };
        if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle) != precision
            || crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            return Ok(None);
        }
        validate_live(handle)?;
    }

    let mut protected = vec![predictions, targets];
    if let Some(GpuLossFactor::Resident(handle)) = options.mask.as_ref() {
        protected.push(handle);
    }
    let weights = match materialize_gpu_loss_factor(
        provider,
        options.weights.take(),
        predictions.shape(),
        precision.expect("real prediction precision was admitted"),
        "weights",
        &protected,
    )? {
        MaterializedFactor::Ready(handle) => handle,
        MaterializedFactor::Unsupported => return Ok(None),
    };
    protected = vec![predictions, targets];
    if let Some(handle) = weights.as_ref() {
        protected.push(handle);
    }
    let mask = match materialize_gpu_loss_factor(
        provider,
        options.mask.take(),
        predictions.shape(),
        precision.expect("real prediction precision was admitted"),
        "mask",
        &protected,
    )? {
        MaterializedFactor::Ready(handle) => handle,
        MaterializedFactor::Unsupported => return Ok(None),
    };

    let mut result = match provider.crossentropy_terms(&ProviderCrossentropyRequest {
        predictions,
        targets,
        weights: weights.as_ref(),
        mask: mask.as_ref(),
        mode: provider_mode(options.mode),
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
    crate::builtins::common::gpu_helpers::propagate_output_provenance(
        &mut result.losses,
        borrowed.into_iter().flatten(),
    )?;

    match options.reduction {
        CrossentropyReduction::None => Ok(Some(
            crate::builtins::common::gpu_helpers::resident_gpu_value(result.losses)?,
        )),
        CrossentropyReduction::Sum => {
            let sum = match provider.reduce_sum(&result.losses).await {
                Ok(sum) => sum,
                Err(err) if err.is_unsupported() => return Ok(None),
                Err(err) => return Err(provider_error(err)),
            };
            let mut sum_protected = vec![predictions, targets, &result.losses];
            sum_protected.extend(factor_handles.into_iter().flatten());
            validate_result_handle(&sum, &[1, 1], precision, provider, &sum_protected)?;
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

#[cfg(test)]
mod tests;
