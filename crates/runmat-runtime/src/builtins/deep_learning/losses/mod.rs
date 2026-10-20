#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderCrossentropyMode,
    ProviderCrossentropyRequest,
};
use runmat_builtins::{DeepLearningAxisLabel as Axis, DeepLearningAxisLayout as Layout};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::{builtins::common::tensor, BuiltinResult};

use super::{deep_learning_error, gather_args, numeric_scalar, object, scalar_text};

mod admission;
mod format;
mod gpu;
mod host;
mod mask;
mod options;
mod weights;

use format::{
    batch_size, batch_size_for_shape, column_major_strides, parse_loss_format, reduce_mask_included,
};
use gpu::try_crossentropy_gpu;
use host::{
    evaluate_crossentropy_terms, is_crossentropy_option_name, require_loss_payload_compatible,
    LossPayload,
};
use mask::{mask_values_for_layout, LossMask};
use options::{
    parse_classification_mode, parse_reduction, provider_mode, ClassificationMode,
    CrossentropyOptions, CrossentropyReduction, NormalizationFactor, NormalizationMode,
    ParsedLossOptions,
};
use runmat_builtins::CrossentropyOptionKey as OptionKey;
use weights::LossWeights;

#[cfg(test)]
mod tests;

#[runtime_builtin(
    name = "crossentropy",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::deep_learning::losses"
)]
pub(super) async fn crossentropy_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    predictions: Value,
    targets: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "crossentropy",
        &runmat_builtins::CROSSENTROPY_INVALID_INPUT,
    )?;
    let positional_weights = rest
        .first()
        .is_some_and(|value| !is_crossentropy_option_name(value));
    let options = rest
        .view()
        .range(usize::from(positional_weights)..)
        .expect("crossentropy option suffix");
    let integer_data =
        crate::builtins::common::validation::value_has_native_integer_class(&predictions)
            || crate::builtins::common::validation::value_has_native_integer_class(&targets)
            || (positional_weights
                && crate::builtins::common::validation::value_has_native_integer_class(&rest[0]))
            || options.chunks_exact(2).any(|pair| {
                scalar_text(&pair[0], "crossentropy")
                    .is_ok_and(|name| OptionKey::from_name(&name) == Some(OptionKey::Weights))
                    && crate::builtins::common::validation::value_has_native_integer_class(&pair[1])
            });
    if integer_data {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CROSSENTROPY_INTEGER_EXTENSION,
            "crossentropy",
        )?;
    }
    if matches!(
        &predictions,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) && matches!(
        &targets,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::CROSSENTROPY_PLAIN_NUMERIC_EXTENSION,
            "crossentropy",
        )?;
    }
    admission::preflight_host_decidable(&predictions, &targets, rest.view())?;
    let explicit_gpu_input = std::iter::once(&predictions)
        .chain(std::iter::once(&targets))
        .chain(rest.iter())
        .any(crate::builtins::common::validation::value_contains_explicit_gpu);
    let has_gpu = matches!(&predictions, Value::GpuTensor(_))
        || matches!(&targets, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)));
    if explicit_gpu_input && !has_gpu {
        return Err(explicit_gpu_fallback_error());
    }
    let (predictions, targets) = if has_gpu {
        if let Some(value) = try_crossentropy_gpu(&predictions, &targets, rest.view()).await? {
            return Ok(value);
        }
        if explicit_gpu_input {
            return Err(explicit_gpu_fallback_error());
        }
        let pair = [predictions, targets];
        let pair = crate::call::BuiltinArguments::from_borrowed(&pair, runtime.as_ref())?;
        let mut gathered = gather_args(pair, runtime.as_ref()).await?;
        (
            gathered.pop_front().expect("prediction slot"),
            gathered.pop_front().expect("target slot"),
        )
    } else {
        (predictions, targets)
    };

    let predictions = LossPayload::parse(&predictions, "predictions")?;
    let targets = LossPayload::parse(&targets, "targets")?;
    if predictions.data.len() != targets.data.len() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must have the same number of elements",
        ));
    }
    if predictions.data.is_empty() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must not be empty",
        ));
    }
    let mut args = gather_args(rest, runtime.as_ref()).await?;
    let weights = if args
        .first()
        .is_some_and(|value| !is_crossentropy_option_name(value))
    {
        Some(LossWeights::parse(
            &args.pop_front().expect("positional weights"),
            "weights",
        )?)
    } else {
        None
    };
    let options = CrossentropyOptions::from_args(args.view(), &predictions, weights)?;
    require_loss_payload_compatible(&predictions, &targets, options.data_format.as_ref())?;

    let mask = options.mask_values(&predictions)?;
    let weights = match options.weights {
        Some(ref weights) => weights.materialize(
            &predictions,
            options.data_format.as_ref(),
            options.weights_format.as_ref(),
        )?,
        None => vec![1.0; predictions.data.len()],
    };

    let losses = evaluate_crossentropy_terms(&predictions.data, &targets.data, &options.mode)?;
    let mut weighted_losses = Vec::with_capacity(losses.len());
    for ((loss, weight), mask) in losses.iter().zip(weights.iter()).zip(mask.iter()) {
        weighted_losses.push(loss * weight * mask);
    }

    match options.reduction {
        CrossentropyReduction::Sum => {
            let normalized = options.reduce_sum(&predictions, &weighted_losses, &mask)?;
            if !normalized.is_finite() {
                return Err(deep_learning_error(
                    "crossentropy",
                    "crossentropy: reduction produced a non-finite loss",
                ));
            }
            predictions.materialize_scalar(normalized)
        }
        CrossentropyReduction::None => predictions.materialize_array(weighted_losses, false),
    }
}

fn explicit_gpu_fallback_error() -> crate::RuntimeError {
    super::errors::descriptor_error(
        "crossentropy",
        "crossentropy: explicit gpuArray inputs require a supported resident cross-entropy provider path",
        &runmat_builtins::CROSSENTROPY_UNSUPPORTED,
    )
}
