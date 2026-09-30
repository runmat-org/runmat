//! Pure GPU-route admission. An unsupported kernel/layout permits automatic
//! host execution; malformed geometry or an invalid resident owner does not.

use super::*;
use crate::builtins::deep_learning::model::parameters::ParameterRole;
use runmat_accelerate_api::DeepLearningPool1DCapability;
use runmat_types::ClassIdentity;
use std::fmt;

fn has_global_pool(layers: &[Value]) -> bool {
    layers.iter().any(|layer| {
        matches!(layer, Value::Object(object) if ForwardLayerKind::from_identity(&object.class_name)
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
        let Some(kind) = ForwardLayerKind::from_identity(&object.class_name) else {
            return Ok(Err(GpuCapabilityReason::Layer(object.class_name.clone())));
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
                    ForwardLayerKind::from_identity(&object.class_name),
                    Some(
                        ForwardLayerKind::FeatureInput
                            | ForwardLayerKind::SequenceInput
                            | ForwardLayerKind::GlobalAveragePooling1D
                    )
                ) =>
            {
                Some(object.class_name.clone())
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
pub(in crate::builtins::deep_learning::model) enum GpuCapabilityReason {
    AxisLayout,
    InputStorage,
    InputRank,
    InputPrecision,
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
                "provider cannot execute this resident pooling geometry within its device limits"
            ),
        }
    }
}

#[derive(Clone, Copy)]
enum GpuOperand {
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
            Some(ForwardLayerKind::Relu) if saw_input => saw_transform = true,
            Some(ForwardLayerKind::Elu) if saw_input => {
                let _ = elu_alpha(layer, function)?;
                saw_transform = true;
            }
            Some(ForwardLayerKind::Softmax) if saw_input => saw_transform = true,
            Some(kind) if kind.is_output() && saw_input && saw_transform => {}
            _ => {
                // The host plan has already admitted the complete network.
                // Its later geometry is not the channel count of this GPU
                // subset once an unimplemented layer changes that geometry.
                return Ok(Some(GpuCapabilityReason::Layer(layer.class_name.clone())));
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
    Ok(None)
}

fn invalid_owner(function: &'static str, operand: GpuOperand) -> crate::RuntimeError {
    crate::runtime_error::build_runtime_error(format!(
        "{function}: resident {operand} has no compatible live provider owner"
    ))
    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}
