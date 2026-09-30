//! Provider-resident forward execution and result ownership validation.
use super::execution::NetworkOutputLayout;
use super::*;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use runmat_builtins::DeepLearningAxisLabel;

mod parameters;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GpuForwardLayout {
    BatchChannels,
    ChannelsBatch,
}

impl GpuForwardLayout {
    pub(super) fn from_axes(axes: &AxisPlan) -> Option<Self> {
        use DeepLearningAxisLabel::{Batch, Channel};
        match axes.layout().axes() {
            [Batch, Channel] => Some(Self::BatchChannels),
            [Channel, Batch] => Some(Self::ChannelsBatch),
            _ => None,
        }
    }

    fn channels(self, shape: &[usize]) -> usize {
        match self {
            Self::BatchChannels => shape[1],
            Self::ChannelsBatch => shape[0],
        }
    }
}

/// Execute the subset of sequential feed-forward networks for which every
/// operation is available through the acceleration provider. The complete
/// layer sequence is validated before the first upload or dispatch so a
/// rejected layer cannot leave behind a partially executed GPU forward pass.
pub(super) async fn evaluate_network_gpu(
    object: &ObjectInstance,
    input: &GpuTensorHandle,
    layout: GpuForwardLayout,
    output_layout: NetworkOutputLayout,
    function: &'static str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| {
            unsupported_error(
                function,
                format!("{function}: no acceleration provider owns the GPU dlarray input"),
            )
        })?;
    let layers = object
        .properties
        .get("Layers")
        .map(|value| layer_slice(value, function))
        .transpose()?
        .unwrap_or(&[]);
    validate_gpu_forward_layers(layers, input, layout, function)?;

    // Provider numerical kernels use [B,C]. CB needs an actual resident
    // transpose; labels alone cannot change its physical interpretation.
    let mut current = if layout == GpuForwardLayout::ChannelsBatch {
        match provider.transpose(input) {
            Ok(output) => validate_layout_transpose(output, input, provider, function)?,
            Err(error)
                if runmat_accelerate_api::is_unsupported_provider_operation(&error)
                    && !runmat_accelerate_api::handle_is_explicit(input) =>
            {
                return Ok(None)
            }
            Err(error) => {
                return Err(provider_failure(
                    function,
                    "input layout transpose",
                    error.into(),
                ))
            }
        }
    } else {
        input.clone()
    };
    for layer in layers {
        let Value::Object(layer) = layer else {
            unreachable!("validate_gpu_forward_layers accepts layer objects only");
        };
        if layer.class_name.is(super::FULLY_CONNECTED_LAYER_CLASS) {
            current = fully_connected_forward_gpu(provider, current, layer, function).await?;
        } else if layer.class_name.is(super::RELU_LAYER_CLASS) {
            let output = provider
                .scalar_max(&current, 0.0)
                .map_err(|err| provider_failure(function, "provider-resident ReLU", err.into()))?;
            validate_unary_forward_result(&output, &current, provider, function)?;
            current = output;
        } else if layer.class_name.is(super::ELU_LAYER_CLASS) {
            let alpha = elu_alpha(layer, function)?;
            let output = provider
                .activation_elu(&current, alpha)
                .await
                .map_err(|err| provider_failure(function, "provider-resident ELU", err.into()))?;
            validate_unary_forward_result(&output, &current, provider, function)?;
            current = output;
        } else if layer.class_name.is(super::SOFTMAX_LAYER_CLASS) {
            let output = provider
                .activation_softmax_rows(&current)
                .await
                .map_err(|err| {
                    provider_failure(function, "provider-resident Softmax", err.into())
                })?;
            validate_unary_forward_result(&output, &current, provider, function)?;
            current = output;
        }
    }
    let consumed_parameters = layers.iter().filter_map(|layer| {
        let Value::Object(layer) = layer else {
            return None;
        };
        layer
            .class_name
            .is(super::FULLY_CONNECTED_LAYER_CLASS)
            .then_some(layer)
    });
    let resident_parameters = consumed_parameters.flat_map(|layer| {
        ["Weights", "Bias"]
            .into_iter()
            .filter_map(move |name| resident_parameter_handle(layer, name))
    });
    crate::builtins::common::gpu_helpers::propagate_output_provenance(
        &mut current,
        std::iter::once(input).chain(resident_parameters),
    )?;
    if layout == GpuForwardLayout::ChannelsBatch && output_layout == NetworkOutputLayout::Preserve {
        let restored = provider
            .transpose(&current)
            .map_err(|error| provider_failure(function, "output layout transpose", error.into()))?;
        let mut restored = validate_layout_transpose(restored, &current, provider, function)?;
        crate::builtins::common::gpu_helpers::propagate_output_provenance(
            &mut restored,
            [input, &current],
        )?;
        Ok(Some(restored))
    } else {
        Ok(Some(current))
    }
}

fn validate_layout_transpose(
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        unary_gpu_output_matches_shape, GpuOutputAliasPolicy, UnaryGpuOutputContract,
    };
    let shape = [input.shape()[1], input.shape()[0]];
    // A singleton axis does not move any element during a 2-D transpose, so
    // a provider may legally return a reshaped view of the same allocation.
    let alias = if input.shape().iter().all(|extent| *extent > 1) {
        GpuOutputAliasPolicy::RequireDistinct
    } else {
        GpuOutputAliasPolicy::AllowInput
    };
    if unary_gpu_output_matches_shape(
        &output,
        input,
        provider,
        &shape,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    ) {
        Ok(output)
    } else {
        Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            output,
            crate::runtime_error::build_runtime_error(format!(
                "{function}: provider returned an invalid layout transpose descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

fn validate_unary_forward_result(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<()> {
    use crate::builtins::common::gpu_helpers::{
        unary_gpu_output_matches, GpuOutputAliasPolicy, UnaryGpuOutputContract,
    };
    let precision = runmat_accelerate_api::handle_precision(input);
    if unary_gpu_output_matches(
        output,
        input,
        provider,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision,
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::AllowInput,
        },
    ) {
        Ok(())
    } else {
        Err(crate::runtime_error::build_runtime_error(format!(
            "{function}: provider returned an invalid forward result descriptor"
        ))
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build())
    }
}

pub(super) fn validate_gpu_forward_layers(
    layers: &[Value],
    input: &GpuTensorHandle,
    layout: GpuForwardLayout,
    function: &'static str,
) -> BuiltinResult<()> {
    if runmat_accelerate_api::handle_storage(input) != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Err(unsupported_error(
            function,
            format!(
                "{function}: provider-resident forward currently requires a real dlarray input"
            ),
        ));
    }
    if input.shape().len() != 2 {
        return Err(unsupported_error(
            function,
            format!("{function}: provider-resident forward currently requires a 2-D dlarray input"),
        ));
    }
    if runmat_accelerate_api::handle_precision(input).is_none()
        || runmat_accelerate_api::handle_is_logical(input)
    {
        return Err(unsupported_error(
            function,
            format!("{function}: provider-resident forward requires single or double dlarray Data"),
        ));
    }
    let mut features = layout.channels(input.shape());
    let mut saw_input = false;
    let mut saw_transform = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        match ForwardLayerKind::from_identity(&layer.class_name) {
            Some(ForwardLayerKind::FeatureInput) if !saw_input => {
                let expected = feature_input_width(layer, function)?;
                if features != expected {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: input has {features} features, but featureInputLayer expects {expected}"
                        ),
                    ));
                }
                saw_input = true;
            }
            Some(ForwardLayerKind::FullyConnected) if saw_input => {
                features = fully_connected_output_width(layer, features, function)?;
                let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(
                    input,
                )
                .ok_or_else(|| {
                    unsupported_error(
                        function,
                        format!("{function}: no acceleration provider owns the GPU dlarray input"),
                    )
                })?;
                for name in ["Weights", "Bias"] {
                    if let Some(handle) = resident_parameter_handle(layer, name) {
                        let owner =
                            crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                                .ok_or_else(|| {
                                    unsupported_error(
                                        function,
                                        format!("{function}: {name} has no owning provider"),
                                    )
                                })?;
                        if !runmat_accelerate_api::same_provider_instance(provider, owner) {
                            return Err(unsupported_error(
                                function,
                                format!("{function}: {name} belongs to a different provider"),
                            ));
                        }
                        if runmat_accelerate_api::handle_precision(handle)
                            != runmat_accelerate_api::handle_precision(input)
                            && runmat_accelerate_api::handle_is_explicit(handle)
                        {
                            return Err(unsupported_error(
                                function,
                                format!("{function}: explicit resident {name} precision differs from input"),
                            ));
                        }
                    }
                }
                saw_transform = true;
            }
            Some(ForwardLayerKind::Relu) if saw_input => saw_transform = true,
            Some(ForwardLayerKind::Elu) if saw_input => {
                let _ = elu_alpha(layer, function)?;
                saw_transform = true;
            }
            Some(ForwardLayerKind::Softmax) if saw_input => saw_transform = true,
            Some(kind) if kind.is_output() && saw_input && saw_transform => {}
            _ => {
                return Err(unsupported_error(
                    function,
                    format!(
                        "{function}: provider-resident forward does not yet support layer type '{}'",
                        layer.class_name
                    ),
                ));
            }
        }
    }
    if !saw_input {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must start with a supported input layer"),
        ));
    }
    if !saw_transform {
        return Err(unsupported_error(
            function,
            format!(
                "{function}: provider-resident forward requires at least one supported transform layer"
            ),
        ));
    }
    Ok(())
}

fn resident_parameter_handle<'a>(
    layer: &'a ObjectInstance,
    name: &str,
) -> Option<&'a GpuTensorHandle> {
    let value = layer.properties.get(name)?;
    match value {
        Value::GpuTensor(handle) => Some(handle),
        Value::Object(object) if object.class_name.is(super::DLARRAY_CLASS) => {
            match object.properties.get("Data") {
                Some(Value::GpuTensor(handle)) => Some(handle),
                _ => None,
            }
        }
        _ => None,
    }
}

async fn fully_connected_forward_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    current: GpuTensorHandle,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    let precision = runmat_accelerate_api::handle_precision(&current).ok_or_else(|| {
        unsupported_error(
            function,
            format!("{function}: input has no floating precision"),
        )
    })?;
    let width = fully_connected_output_width(layer, current.shape()[1], function)?;
    let weights_handle = parameters::prepare_weights(provider, layer, &current, function).await?;
    let bias_handle =
        parameters::prepare_bias(provider, layer, &current, &weights_handle, width, function)
            .await?;
    let product = provider
        .matmul(&current, &weights_handle)
        .await
        .map_err(|err| provider_failure(function, "fully connected matmul", err.into()))?;
    validate_binary_forward_result(
        &product,
        &current,
        &weights_handle,
        provider,
        &[current.shape()[0], width],
        precision,
        function,
    )?;
    let output = provider
        .elem_add(&product, &bias_handle)
        .await
        .map_err(|err| provider_failure(function, "fully connected bias add", err.into()))?;
    validate_binary_forward_result(
        &output,
        &product,
        &bias_handle,
        provider,
        &[current.shape()[0], width],
        precision,
        function,
    )?;
    Ok(output)
}

fn provider_failure(
    function: &'static str,
    operation: &str,
    source: ProviderExecutionError,
) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        deep_learning_error(
            function,
            format!("{function}: {operation} failed: {source}"),
        ),
        source,
    )
}

fn validate_binary_forward_result(
    output: &GpuTensorHandle,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    function: &'static str,
) -> BuiltinResult<()> {
    use crate::builtins::common::gpu_helpers::{
        binary_gpu_output_matches, BinaryGpuOutputContract, GpuOutputAliasPolicy,
    };
    if binary_gpu_output_matches(
        output,
        left,
        right,
        provider,
        &BinaryGpuOutputContract {
            shape: shape.to_vec(),
            storage: GpuTensorStorage::Real,
            precision: Some(precision),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) {
        Ok(())
    } else {
        Err(crate::runtime_error::build_runtime_error(format!(
            "{function}: provider returned an invalid forward result descriptor"
        ))
        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build())
    }
}
