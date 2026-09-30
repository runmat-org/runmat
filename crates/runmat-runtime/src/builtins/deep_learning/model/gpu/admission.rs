//! Pure GPU-route admission. An unsupported kernel/layout permits automatic
//! host execution; malformed geometry or an invalid resident owner does not.

use super::*;
use crate::builtins::deep_learning::model::parameters::ParameterRole;
use runmat_types::ClassIdentity;
use std::fmt;

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
    function: &'static str,
) -> BuiltinResult<GpuForwardAdmission> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| invalid_owner(function, GpuOperand::Input))?;
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
    function: &'static str,
) -> BuiltinResult<()> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| invalid_owner(function, GpuOperand::Input))?;
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
                    if let Some(handle) = resident_parameter_handle(layer, role.name()) {
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
