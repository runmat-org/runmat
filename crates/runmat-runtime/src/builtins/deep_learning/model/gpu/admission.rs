//! Pure GPU-route admission. An unsupported kernel/layout permits automatic
//! host execution; malformed geometry or an invalid resident owner does not.

use super::*;
use crate::builtins::deep_learning::model::parameters::ParameterRole;
use runmat_accelerate_api::{
    DeepLearningFeedForwardCapability, DeepLearningFeedForwardStage, DeepLearningPool1DCapability,
};
use runmat_types::ClassIdentity;
use std::fmt;

/// Validated public network ingress, before tape publication, uploads or any
/// layer execution. Inspect all known resident operands before allowing a
/// policy decline: an invalid owner/assignment must never become a retry.
pub(in crate::builtins::deep_learning::model) fn preflight_network_compute_owners(
    host: &crate::builtins::deep_learning::model::plan::host::HostNetworkPlan<'_>,
    input: Option<&GpuTensorHandle>,
    function: &'static str,
) -> BuiltinResult<()> {
    use crate::builtins::common::gpu_helpers::{self, ResidentComputeDecline};
    let operands = || {
        input
            .map(|handle| (handle, GpuOperand::Input))
            .into_iter()
            .chain(host.parameter_views().iter().filter_map(|row| {
                match row
                    .value
                    .map(crate::builtins::deep_learning::model::admission::forward_data)
                {
                    Some(Value::GpuTensor(handle)) => {
                        Some((handle, GpuOperand::Parameter(row.key.role)))
                    }
                    _ => None,
                }
            }))
    };
    let mut device = None;
    for (handle, operand) in operands() {
        match gpu_helpers::resolve_resident_compute_owner(handle) {
            Ok(_) | Err(ResidentComputeDecline::PolicyDenied) => {}
            Err(decline) => return Err(decline.with_domain_error(invalid_owner(function, operand))),
        }
        if device.is_some_and(|device| device != handle.device_id()) {
            return Err(ResidentComputeDecline::InvalidOwner
                .with_domain_error(invalid_owner(function, operand)));
        }
        device = Some(handle.device_id());
    }
    for (handle, operand) in operands() {
        gpu_helpers::require_resident_compute_owner(handle, || invalid_owner(function, operand))?;
    }
    Ok(())
}

fn has_global_pool(layers: &[Value]) -> bool {
    layers.iter().any(|layer| {
        matches!(layer, Value::Object(object) if ForwardLayerKind::from_identity(object.class_name())
            == Some(ForwardLayerKind::GlobalAveragePooling1D))
    })
}

fn classify_global_pool(
    layers: &[Value],
    input: &GpuTensorHandle,
    axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Result<usize, GpuCapabilityReason>> {
    use DeepLearningAxisLabel::{Spatial, Time};
    if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
        return Ok(Err(GpuCapabilityReason::InputStorage));
    }
    if runmat_accelerate_api::handle_precision(input).is_none()
        || runmat_accelerate_api::handle_is_logical(input)
        || runmat_accelerate_api::handle_integer_type(input).is_some()
    {
        return Ok(Err(GpuCapabilityReason::InputPrecision));
    }
    if input.shape() != axes.shape() {
        return Err(deep_learning_error(
            function,
            "resident input shape differs from admitted axes",
        ));
    }
    let mut kinds = Vec::with_capacity(layers.len());
    for layer in layers {
        let Value::Object(object) = layer else {
            return Err(deep_learning_error(
                function,
                "network layers must be layer objects",
            ));
        };
        let Some(kind) = ForwardLayerKind::from_identity(object.class_name()) else {
            return Ok(Err(GpuCapabilityReason::Layer(object.class_name().clone())));
        };
        kinds.push(kind);
    }
    let valid = matches!(
        kinds.as_slice(),
        [
            ForwardLayerKind::FeatureInput | ForwardLayerKind::SequenceInput,
            ForwardLayerKind::GlobalAveragePooling1D
        ]
    ) || matches!(
        kinds.as_slice(),
        [ForwardLayerKind::FeatureInput | ForwardLayerKind::SequenceInput,
         ForwardLayerKind::GlobalAveragePooling1D, output] if output.is_output()
    );
    if !valid {
        let unsupported = layers.iter().find_map(|layer| match layer {
            Value::Object(object)
                if !matches!(
                    ForwardLayerKind::from_identity(object.class_name()),
                    Some(
                        ForwardLayerKind::FeatureInput
                            | ForwardLayerKind::SequenceInput
                            | ForwardLayerKind::GlobalAveragePooling1D
                    )
                ) =>
            {
                Some(object.class_name().clone())
            }
            _ => None,
        });
        return Ok(Err(unsupported.map_or(
            GpuCapabilityReason::NoSupportedTransform,
            GpuCapabilityReason::Layer,
        )));
    }
    let Some(axis) = axes
        .layout()
        .index(Spatial)
        .or_else(|| axes.layout().index(Time))
    else {
        return Ok(Err(GpuCapabilityReason::AxisLayout));
    };
    if axes.shape()[axis] == 0 {
        return Err(deep_learning_error(
            function,
            "cannot average an empty pool axis",
        ));
    }
    Ok(Ok(axis))
}

fn admit_global_pool_route(
    layers: &[Value],
    input: &GpuTensorHandle,
    input_axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<Result<usize, GpuCapabilityReason>> {
    let axis = match classify_global_pool(layers, input, input_axes, function)? {
        Ok(axis) => axis,
        Err(reason) => return Ok(Err(reason)),
    };
    let route = super::pooling::route(input_axes, source_axes, target_axes, axis, function)?;
    let capability = provider
        .deep_learning_global_average_pool_1d_capability(input, &route)
        .map_err(|error| provider_failure(function, "resident pooling admission", error.into()))?;
    match capability {
        DeepLearningPool1DCapability::Ready => Ok(Ok(axis)),
        DeepLearningPool1DCapability::UnsupportedOperation => {
            Ok(Err(GpuCapabilityReason::ProviderOperation))
        }
        DeepLearningPool1DCapability::ResourceLimit => Ok(Err(GpuCapabilityReason::ResourceLimit)),
    }
}

#[derive(Debug)]
pub(in crate::builtins::deep_learning::model) enum GpuForwardAdmission {
    Ready(GpuForwardLayout),
    UnsupportedCapability(GpuCapabilityReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::builtins::deep_learning) enum GpuCapabilityReason {
    AxisLayout,
    InputStorage,
    InputRank,
    InputPrecision,
    ParameterPrecision(ParameterRole),
    Layer(ClassIdentity),
    NoSupportedTransform,
    ProviderOperation,
    ResourceLimit,
}

impl fmt::Display for GpuCapabilityReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AxisLayout => write!(
                formatter,
                "provider-resident forward does not support this axis layout"
            ),
            Self::InputStorage => write!(
                formatter,
                "provider-resident forward requires real dlarray Data"
            ),
            Self::InputRank => write!(
                formatter,
                "provider-resident forward requires 2-D dlarray Data"
            ),
            Self::InputPrecision => write!(
                formatter,
                "provider-resident forward requires single or double dlarray Data"
            ),
            Self::ParameterPrecision(role) => write!(
                formatter,
                "provider-resident forward cannot use this {} precision",
                role.name()
            ),
            Self::Layer(identity) => write!(
                formatter,
                "provider-resident forward does not yet support layer type '{}'",
                identity.display_name()
            ),
            Self::NoSupportedTransform => write!(
                formatter,
                "provider-resident forward requires a supported transform layer"
            ),
            Self::ProviderOperation => write!(
                formatter,
                "provider does not implement a required resident operation"
            ),
            Self::ResourceLimit => write!(
                formatter,
                "provider cannot execute this resident geometry within its device limits"
            ),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum GpuOperand {
    Input,
    Parameter(ParameterRole),
}

impl fmt::Display for GpuOperand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => formatter.write_str("input"),
            Self::Parameter(role) => formatter.write_str(role.name()),
        }
    }
}

pub(in crate::builtins::deep_learning::model) fn admit_gpu_forward(
    layers: &[Value],
    input: &GpuTensorHandle,
    axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<GpuForwardAdmission> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| invalid_owner(function, GpuOperand::Input))?;
    if has_global_pool(layers) {
        return match admit_global_pool_route(
            layers,
            input,
            axes,
            source_axes,
            target_axes,
            provider,
            function,
        )? {
            Ok(axis) => Ok(GpuForwardAdmission::Ready(GpuForwardLayout::GlobalPool {
                axis,
            })),
            Err(reason) => Ok(GpuForwardAdmission::UnsupportedCapability(reason)),
        };
    }
    let Some(layout) = GpuForwardLayout::from_axes(axes) else {
        return Ok(GpuForwardAdmission::UnsupportedCapability(
            GpuCapabilityReason::AxisLayout,
        ));
    };
    match classify_layers(layers, input, layout, provider, function)? {
        Some(reason) => Ok(GpuForwardAdmission::UnsupportedCapability(reason)),
        None => Ok(GpuForwardAdmission::Ready(layout)),
    }
}

/// Recheck immediately before provider effects, including direct callers.
pub(super) fn require_gpu_forward_layers(
    layers: &[Value],
    input: &GpuTensorHandle,
    layout: GpuForwardLayout,
    axes: &AxisPlan,
    source_axes: &AxisPlan,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<()> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| invalid_owner(function, GpuOperand::Input))?;
    if let GpuForwardLayout::GlobalPool { axis } = layout {
        return match admit_global_pool_route(
            layers,
            input,
            axes,
            source_axes,
            target_axes,
            provider,
            function,
        )? {
            Ok(admitted) if admitted == axis => Ok(()),
            Ok(_) => Err(deep_learning_error(
                function,
                "GPU pooling axis changed after admission",
            )),
            Err(reason) => Err(unsupported_error(function, format!("{function}: {reason}"))),
        };
    }
    if let Some(reason) = classify_layers(layers, input, layout, provider, function)? {
        return Err(unsupported_error(function, format!("{function}: {reason}")));
    }
    Ok(())
}

/// None means ready; Some is strictly a missing GPU capability. All invalid
/// input, topology and provider-owner states return an error.
fn classify_layers(
    layers: &[Value],
    input: &GpuTensorHandle,
    layout: GpuForwardLayout,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    function: &'static str,
) -> BuiltinResult<Option<GpuCapabilityReason>> {
    if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
        return Ok(Some(GpuCapabilityReason::InputStorage));
    }
    if input.shape().len() != 2 {
        return Ok(Some(GpuCapabilityReason::InputRank));
    }
    if runmat_accelerate_api::handle_precision(input).is_none()
        || runmat_accelerate_api::handle_is_logical(input)
    {
        return Ok(Some(GpuCapabilityReason::InputPrecision));
    }
    let mut features = layout.channels(input.shape());
    let batch = match layout {
        GpuForwardLayout::BatchChannels => input.shape()[0],
        GpuForwardLayout::ChannelsBatch => input.shape()[1],
        GpuForwardLayout::GlobalPool { .. } => unreachable!("pool route is admitted separately"),
    };
    if layout == GpuForwardLayout::ChannelsBatch {
        let stage = DeepLearningFeedForwardStage::Transpose2D {
            rows: input.shape()[0],
            columns: input.shape()[1],
        };
        if let Some(reason) = query_stage(provider, input, stage, function)? {
            return Ok(Some(reason));
        }
    }
    let mut saw_input = false;
    let mut saw_transform = false;
    for layer in layers {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        match ForwardLayerKind::from_identity(layer.class_name()) {
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
                let output_channels = fully_connected_output_width(layer, features, function)?;
                let stage = DeepLearningFeedForwardStage::Dense {
                    batch,
                    input_channels: features,
                    output_channels,
                };
                if let Some(reason) = query_stage(provider, input, stage, function)? {
                    return Ok(Some(reason));
                }
                features = output_channels;
                for role in [ParameterRole::Weights, ParameterRole::Bias] {
                    if let Some(handle) = resident_parameter_handle(layer, role) {
                        let owner =
                            crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                                .ok_or_else(|| {
                                    invalid_owner(function, GpuOperand::Parameter(role))
                                })?;
                        if !runmat_accelerate_api::same_provider_instance(provider, owner) {
                            return Err(invalid_owner(function, GpuOperand::Parameter(role)));
                        }
                        if runmat_accelerate_api::handle_precision(handle)
                            != runmat_accelerate_api::handle_precision(input)
                            && runmat_accelerate_api::handle_is_explicit(handle)
                        {
                            return Err(unsupported_error(
                                function,
                                format!(
                                    "{function}: explicit resident {} precision differs from input",
                                    role.name()
                                ),
                            ));
                        }
                    }
                }
                saw_transform = true;
            }
            Some(ForwardLayerKind::Relu) if saw_input => {
                let Some(elements) = batch.checked_mul(features) else {
                    return Ok(Some(GpuCapabilityReason::ResourceLimit));
                };
                if let Some(reason) = query_stage(
                    provider,
                    input,
                    DeepLearningFeedForwardStage::Relu { elements },
                    function,
                )? {
                    return Ok(Some(reason));
                }
                saw_transform = true;
            }
            Some(ForwardLayerKind::Elu) if saw_input => {
                let _ = elu_alpha(layer, function)?;
                let Some(elements) = batch.checked_mul(features) else {
                    return Ok(Some(GpuCapabilityReason::ResourceLimit));
                };
                if let Some(reason) = query_stage(
                    provider,
                    input,
                    DeepLearningFeedForwardStage::Elu { elements },
                    function,
                )? {
                    return Ok(Some(reason));
                }
                saw_transform = true;
            }
            Some(ForwardLayerKind::Softmax) if saw_input => {
                if let Some(reason) = query_stage(
                    provider,
                    input,
                    DeepLearningFeedForwardStage::SoftmaxRows {
                        rows: batch,
                        columns: features,
                    },
                    function,
                )? {
                    return Ok(Some(reason));
                }
                saw_transform = true;
            }
            Some(kind) if kind.is_output() && saw_input && saw_transform => {}
            _ => {
                // The host plan has already admitted the complete network.
                // Its later geometry is not the channel count of this GPU
                // subset once an unimplemented layer changes that geometry.
                return Ok(Some(GpuCapabilityReason::Layer(layer.class_name().clone())));
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
        return Ok(Some(GpuCapabilityReason::NoSupportedTransform));
    }
    if layout == GpuForwardLayout::ChannelsBatch {
        let stage = DeepLearningFeedForwardStage::Transpose2D {
            rows: batch,
            columns: features,
        };
        if let Some(reason) = query_stage(provider, input, stage, function)? {
            return Ok(Some(reason));
        }
    }
    Ok(None)
}

pub(super) fn query_stage(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    stage: DeepLearningFeedForwardStage,
    function: &'static str,
) -> BuiltinResult<Option<GpuCapabilityReason>> {
    let capability = provider
        .deep_learning_feed_forward_stage_capability(input, stage)
        .map_err(|error| provider_failure(function, "resident stage admission", error.into()))?;
    Ok(match capability {
        DeepLearningFeedForwardCapability::Ready => None,
        DeepLearningFeedForwardCapability::UnsupportedOperation => {
            Some(GpuCapabilityReason::ProviderOperation)
        }
        DeepLearningFeedForwardCapability::ResourceLimit => {
            Some(GpuCapabilityReason::ResourceLimit)
        }
    })
}

pub(super) fn invalid_owner(function: &'static str, operand: GpuOperand) -> crate::RuntimeError {
    crate::runtime_error::build_runtime_error(format!(
        "{function}: resident {operand} has no compatible live provider owner"
    ))
    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    use runmat_builtins::DeepLearningLayerKind as K;
    use runmat_value::Tensor;
    use std::sync::Mutex;

    struct StageProbe {
        observed: Mutex<Vec<DeepLearningFeedForwardStage>>,
        fail_with_error: bool,
    }

    impl AccelProvider for StageProbe {
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            panic!("pure stage admission must not upload")
        }

        fn download<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            panic!("pure stage admission must not download")
        }

        fn device_info(&self) -> String {
            "stage admission probe".into()
        }

        fn deep_learning_feed_forward_stage_capability(
            &self,
            _anchor: &GpuTensorHandle,
            stage: DeepLearningFeedForwardStage,
        ) -> anyhow::Result<DeepLearningFeedForwardCapability> {
            self.observed.lock().unwrap().push(stage);
            if matches!(stage, DeepLearningFeedForwardStage::Dense { .. }) {
                if self.fail_with_error {
                    return Err(anyhow::anyhow!("invalid descriptor"));
                }
                return Ok(DeepLearningFeedForwardCapability::UnsupportedOperation);
            }
            Ok(DeepLearningFeedForwardCapability::Ready)
        }
    }

    fn layers() -> Vec<Value> {
        use crate::builtins::deep_learning::object_with_identity;
        vec![
            object_with_identity(
                K::FeatureInput.class(),
                vec![("InputSize", Value::Num(2.0))],
            )
            .expect("fixture object"),
            object_with_identity(K::Relu.class(), Vec::<(&str, Value)>::new())
                .expect("fixture object"),
            object_with_identity(
                K::FullyConnected.class(),
                vec![
                    ("OutputSize", Value::Num(1.0)),
                    (
                        "Weights",
                        Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
                    ),
                    (
                        "Bias",
                        Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
                    ),
                ],
            )
            .expect("fixture object"),
            object_with_identity(K::Softmax.class(), Vec::<(&str, Value)>::new())
                .expect("fixture object"),
        ]
    }

    #[test]
    fn later_unsupported_stage_is_found_before_any_execution() {
        crate::builtins::common::test_support::with_test_provider(|owner| {
            let input = owner
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    shape: &[2, 3],
                })
                .unwrap();
            let probe = Box::leak(Box::new(StageProbe {
                observed: Mutex::new(Vec::new()),
                fail_with_error: false,
            }));
            let reason = classify_layers(
                &layers(),
                &input,
                GpuForwardLayout::ChannelsBatch,
                probe,
                "forward",
            )
            .unwrap();
            assert_eq!(reason, Some(GpuCapabilityReason::ProviderOperation));
            assert_eq!(
                *probe.observed.lock().unwrap(),
                vec![
                    DeepLearningFeedForwardStage::Transpose2D {
                        rows: 2,
                        columns: 3,
                    },
                    DeepLearningFeedForwardStage::Relu { elements: 6 },
                    DeepLearningFeedForwardStage::Dense {
                        batch: 3,
                        input_channels: 2,
                        output_channels: 1,
                    },
                ]
            );
        });
    }

    #[test]
    fn invalid_later_stage_query_is_an_error_not_a_capability_fallback() {
        crate::builtins::common::test_support::with_test_provider(|owner| {
            let input = owner
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    shape: &[2, 3],
                })
                .unwrap();
            let probe = Box::leak(Box::new(StageProbe {
                observed: Mutex::new(Vec::new()),
                fail_with_error: true,
            }));
            let error = classify_layers(
                &layers(),
                &input,
                GpuForwardLayout::ChannelsBatch,
                probe,
                "forward",
            )
            .unwrap_err();
            assert!(error.to_string().contains("resident stage admission"));
        });
    }
}
