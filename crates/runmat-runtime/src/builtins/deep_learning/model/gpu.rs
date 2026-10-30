//! Provider-resident forward execution and result ownership validation.
use super::execution::NetworkOutputLayout;
use super::parameters::ParameterRole;
use super::*;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use runmat_builtins::DeepLearningAxisLabel;

mod parameters;
pub(in crate::builtins::deep_learning) mod stage_failure;
mod convolution;
mod normalization;
mod admission;
mod pooling;
pub(in crate::builtins::deep_learning) mod mixed;

pub(in crate::builtins::deep_learning) use admission::GpuCapabilityReason;
pub(super) use admission::{
    admit_gpu_forward, preflight_network_compute_owners, GpuForwardAdmission,
};
pub(in crate::builtins::deep_learning) use mixed::{
    admit_host_route, execute_host_route_selected, AdmittedHostRoute,
};
pub(in crate::builtins::deep_learning) use normalization::forward as normalize_network_gpu;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GpuForwardLayout {
    BatchChannels,
    ChannelsBatch,
    GlobalPool { axis: usize },
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
            Self::GlobalPool { .. } => unreachable!("global pooling is not a dense layout"),
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
    input_axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    runtime: Option<&crate::context::RuntimeContext>,
    function: &'static str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| {
            crate::builtins::common::gpu_helpers::ResidentComputeDecline::InvalidOwner
                .with_domain_error(unsupported_error(
                    function,
                    format!("{function}: no acceleration provider owns the resident input"),
                ))
        })?;
    let layers = object
        .properties()
        .get("Layers")
        .map(|value| layer_slice(value, function))
        .transpose()?
        .unwrap_or(&[]);
    admission::require_gpu_forward_layers(
        layers,
        input,
        layout,
        input_axes,
        source_axes,
        target_axes,
        function,
    )?;

    if let GpuForwardLayout::GlobalPool { axis } = layout {
        let check = || mixed::admit_cancellation(runtime, function);
        return pooling::forward(
            provider,
            input,
            axis,
            input_axes,
            source_axes,
            target_axes,
            &check,
            function,
        )
        .await
        .map_err(stage_failure::StageExecutionFailure::into_error);
    }

    // Provider numerical kernels use [B,C]. CB needs an actual resident
    // transpose; labels alone cannot change its physical interpretation.
    let mut current = if layout == GpuForwardLayout::ChannelsBatch {
        match provider.transpose(input) {
            Ok(output) => validate_layout_transpose(output, input, provider, function)?,
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
            unreachable!("GPU admission accepts layer objects only");
        };
        if layer.class_name().is(super::FULLY_CONNECTED_LAYER_CLASS) {
            current = fully_connected_forward_gpu(provider, current, layer, function).await?;
        } else if layer.class_name().is(super::RELU_LAYER_CLASS) {
            let output = provider
                .activation_relu(&current)
                .await
                .map_err(|err| provider_failure(function, "provider-resident ReLU", err.into()))?;
            current = validate_unary_forward_result(output, &current, provider, function)?;
        } else if layer.class_name().is(super::ELU_LAYER_CLASS) {
            let alpha = elu_alpha(layer, function)?;
            let output = provider
                .activation_elu(&current, alpha)
                .await
                .map_err(|err| provider_failure(function, "provider-resident ELU", err.into()))?;
            current = validate_unary_forward_result(output, &current, provider, function)?;
        } else if layer.class_name().is(super::SOFTMAX_LAYER_CLASS) {
            let output = provider
                .activation_softmax_rows(&current)
                .await
                .map_err(|err| {
                    provider_failure(function, "provider-resident Softmax", err.into())
                })?;
            current = validate_unary_forward_result(output, &current, provider, function)?;
        }
    }
    let consumed_parameters = layers.iter().filter_map(|layer| {
        let Value::Object(layer) = layer else {
            return None;
        };
        layer
            .class_name()
            .is(super::FULLY_CONNECTED_LAYER_CLASS)
            .then_some(layer)
    });
    let resident_parameters = consumed_parameters.flat_map(|layer| {
        [ParameterRole::Weights, ParameterRole::Bias]
            .into_iter()
            .filter_map(move |role| resident_parameter_handle(layer, role))
    });
    let current = crate::builtins::common::gpu_helpers::propagate_owned_output_provenance(
        current,
        std::iter::once(input).chain(resident_parameters),
    )?;
    if layout == GpuForwardLayout::ChannelsBatch && output_layout == NetworkOutputLayout::Preserve {
        let restored = provider
            .transpose(&current)
            .map_err(|error| provider_failure(function, "output layout transpose", error.into()))?;
        let restored = validate_layout_transpose(restored, &current, provider, function)?;
        let restored = crate::builtins::common::gpu_helpers::propagate_owned_output_provenance(
            restored,
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
    let alias = if input.shares_allocation_with(&output)
        && input.shape().iter().all(|extent| *extent > 1)
    {
        use runmat_accelerate_api::{GpuTensorLayout, TransposeInfo};
        let expected = match input.view().layout() {
            GpuTensorLayout::Contiguous => GpuTensorLayout::Transpose(TransposeInfo {
                base_rows: input.shape()[0],
                base_cols: input.shape()[1],
            }),
            GpuTensorLayout::Transpose(info)
                if info.base_rows == output.shape()[0] && info.base_cols == output.shape()[1] =>
            {
                GpuTensorLayout::Contiguous
            }
            _ => GpuTensorLayout::Contiguous,
        };
        if output.view().layout() == expected && input.view().layout() != expected {
            GpuOutputAliasPolicy::AllowInput
        } else {
            GpuOutputAliasPolicy::RequireDistinctAllocation
        }
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
    output: GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        unary_gpu_output_matches, GpuOutputAliasPolicy, UnaryGpuOutputContract,
    };
    let precision = runmat_accelerate_api::handle_precision(input);
    if unary_gpu_output_matches(
        &output,
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
        Ok(output)
    } else {
        Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            output,
            crate::runtime_error::build_runtime_error(format!(
                "{function}: provider returned an invalid forward result descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}

fn resident_parameter_handle(
    layer: &ObjectInstance,
    role: ParameterRole,
) -> Option<&GpuTensorHandle> {
    let value = layer.properties().get(role.name())?;
    match value {
        Value::GpuTensor(handle) => Some(handle),
        Value::Object(object) if object.class_name().is(super::DLARRAY_CLASS) => {
            match object.properties().get("Data") {
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
    Ok(
        fully_connected_forward_with_weights_gpu(provider, current, layer, function)
            .await?
            .0,
    )
}

/// Return the very same prepared weight view consumed by the forward matmul.
/// A tape must never reconstruct that view from a subsequently mutated layer.
async fn fully_connected_forward_with_weights_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    current: GpuTensorHandle,
    layer: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<(GpuTensorHandle, GpuTensorHandle)> {
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
    let product = validate_binary_forward_result(
        product,
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
        .map_err(|err| provider_failure(function, "fully connected bias add", err))?;
    let output = validate_binary_forward_result(
        output,
        &product,
        &bias_handle,
        provider,
        &[current.shape()[0], width],
        precision,
        function,
    )?;
    Ok((output, weights_handle))
}

pub(in crate::builtins::deep_learning) fn provider_failure(
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

pub(in crate::builtins::deep_learning) fn validate_binary_forward_result(
    output: GpuTensorHandle,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    function: &'static str,
) -> BuiltinResult<GpuTensorHandle> {
    use crate::builtins::common::gpu_helpers::{
        binary_gpu_output_matches, BinaryGpuOutputContract, GpuOutputAliasPolicy,
    };
    if binary_gpu_output_matches(
        &output,
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
        Ok(output)
    } else {
        Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            output,
            crate::runtime_error::build_runtime_error(format!(
                "{function}: provider returned an invalid forward result descriptor"
            ))
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ))
    }
}
