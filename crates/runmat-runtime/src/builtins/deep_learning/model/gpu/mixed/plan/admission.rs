//! Live resident-owner and provider admission over the one lowered stage
//! sequence. Queries finish before the executor may upload or dispatch.

use super::*;
use crate::builtins::deep_learning::model::admission::resident_input_handle;
use crate::builtins::deep_learning::model::plan::host::HostNetworkPlan;
use runmat_accelerate_api::{
    AccelProvider, DeepLearningConv1DCapability, DeepLearningConv1DOperands,
    DeepLearningFeedForwardStage as PhysicalStage, DeepLearningForwardCapability,
    DeepLearningLayerNormCapability, DeepLearningPool1DCapability, GpuTensorStorage, GpuTensorView,
    ProviderViewCapability, ProviderViewChain, ProviderViewSource, ProviderViewStep,
};
use runmat_builtins::DeepLearningAxisLabel as Label;

fn admit_view(
    provider: &'static dyn AccelProvider,
    anchor: &GpuTensorHandle,
    current: &GpuTensorView,
    operation: ProviderViewStep,
    function: &'static str,
) -> BuiltinResult<Result<GpuTensorView, GpuCapabilityReason>> {
    let query = ProviderViewChain {
        source: ProviderViewSource::Planned(current.clone()),
        steps: vec![operation],
    };
    let expected = query
        .analyze(current, anchor.physical().element_type())
        .map_err(|error| {
            deep_learning_error(function, format!("invalid lowered view: {error:?}"))
        })?;
    let capability = provider
        .provider_view_chain_capability(anchor, &query)
        .map_err(|error| provider_failure(function, "physical view admission", error.into()))?;
    Ok(match capability {
        ProviderViewCapability::Ready(actual) if actual == expected => Ok(actual.view),
        ProviderViewCapability::Ready(_) => {
            return Err(provider_failure(
                function,
                "physical view admission",
                anyhow::anyhow!("provider view disagrees with lowered physical view").into(),
            ))
        }
        ProviderViewCapability::UnsupportedOperation => Err(GpuCapabilityReason::ProviderOperation),
        ProviderViewCapability::ResourceLimit => Err(GpuCapabilityReason::ResourceLimit),
    })
}

fn admit_stages(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    input: &GpuTensorHandle,
    plan: &Plan,
    provider: &'static dyn AccelProvider,
    purpose: RoutePurpose,
    function: &'static str,
) -> BuiltinResult<Result<(), GpuCapabilityReason>> {
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        unreachable!("host plan admitted Layers")
    };
    let mut view = input.view().clone();
    for stage in &plan.stages {
        match stage {
            Stage::Materialize => {
                view = match admit_view(
                    provider,
                    input,
                    &view,
                    ProviderViewStep::Reshape(view.shape().to_vec()),
                    function,
                )? {
                    Ok(view) => view,
                    Err(reason) => return Ok(Err(reason)),
                };
            }
            Stage::Transpose {
                output_axes,
                output_layout,
                materialize,
            } => {
                view = match admit_view(
                    provider,
                    input,
                    &view,
                    if *materialize {
                        ProviderViewStep::Permute2D
                    } else {
                        ProviderViewStep::Transpose2D
                    },
                    function,
                )? {
                    Ok(view) => view,
                    Err(reason) => return Ok(Err(reason)),
                };
                if view.shape() != output_axes.shape() || view.layout() != *output_layout {
                    return Err(deep_learning_error(
                        function,
                        "provider view differs from lowered axes",
                    ));
                }
            }
            Stage::Dense { layer_index } => {
                let [batch, input_channels] = view.shape() else {
                    return Err(deep_learning_error(
                        function,
                        "lowered dense input is not a matrix",
                    ));
                };
                let batch = *batch;
                let input_channels = *input_channels;
                let output_channels = host
                    .steps()
                    .iter()
                    .find(|step| step.index == *layer_index)
                    .expect("lowered dense stage belongs to admitted layer")
                    .output_axes
                    .axis_size(Label::Channel)
                    .expect("admitted dense output has channels");
                if let Err(reason) = super::super::parameters::admit_dense(
                    host,
                    input,
                    *layer_index,
                    input_channels,
                    output_channels,
                    function,
                )? {
                    return Ok(Err(reason));
                }
                if let Some(reason) = super::super::super::admission::query_stage(
                    provider,
                    input,
                    PhysicalStage::Dense {
                        batch,
                        input_channels,
                        output_channels,
                    },
                    function,
                )? {
                    return Ok(Err(reason));
                }
                view = view
                    .with_shape(vec![batch, output_channels])
                    .with_layout(GpuTensorLayout::Contiguous);
            }
            Stage::Conv {
                layer_index,
                request,
            } => {
                if view.layout() != GpuTensorLayout::Contiguous
                    || view.shape() != request.input_shape.as_slice()
                {
                    return Ok(Err(GpuCapabilityReason::AxisLayout));
                }
                let Some(Value::Object(layer)) = layers.data().get(*layer_index) else {
                    unreachable!("host plan admitted convolution layer")
                };
                let weights = match super::super::super::parameters::admit_resident_parameter(
                    layer,
                    crate::builtins::deep_learning::model::parameters::ParameterRole::Weights,
                    provider,
                    input,
                    function,
                )? {
                    Ok(handle) => handle,
                    Err(reason) => return Ok(Err(reason)),
                };
                let bias = match super::super::super::parameters::admit_resident_parameter(
                    layer,
                    crate::builtins::deep_learning::model::parameters::ParameterRole::Bias,
                    provider,
                    input,
                    function,
                )? {
                    Ok(handle) => handle,
                    Err(reason) => return Ok(Err(reason)),
                };
                let current_input = (view == *input.view()).then_some(input);
                let capability = provider
                    .deep_learning_conv1d_capability(
                        DeepLearningConv1DOperands {
                            anchor: input,
                            input: current_input,
                            weights,
                            bias,
                            upstream: None,
                        },
                        request,
                    )
                    .map_err(|error| {
                        provider_failure(function, "resident convolution admission", error.into())
                    })?;
                match capability {
                    DeepLearningConv1DCapability::Ready => {}
                    DeepLearningConv1DCapability::UnsupportedOperation => {
                        return Ok(Err(GpuCapabilityReason::ProviderOperation));
                    }
                    DeepLearningConv1DCapability::ResourceLimit => {
                        return Ok(Err(GpuCapabilityReason::ResourceLimit));
                    }
                }
                view = view
                    .with_shape(request.output_shape.clone())
                    .with_layout(GpuTensorLayout::Contiguous);
            }
            Stage::Relu | Stage::Elu { .. } => {
                let Some(elements) = view
                    .shape()
                    .iter()
                    .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
                else {
                    return Ok(Err(GpuCapabilityReason::ResourceLimit));
                };
                let physical = match stage {
                    Stage::Relu => PhysicalStage::Relu { elements },
                    Stage::Elu { .. } => PhysicalStage::Elu { elements },
                    _ => unreachable!(),
                };
                if let Some(reason) = super::super::super::admission::query_stage(
                    provider, input, physical, function,
                )? {
                    return Ok(Err(reason));
                }
            }
            Stage::Softmax => {
                let [rows, columns] = view.shape() else {
                    return Err(deep_learning_error(
                        function,
                        "lowered softmax input is not a matrix",
                    ));
                };
                if let Some(reason) = super::super::super::admission::query_stage(
                    provider,
                    input,
                    PhysicalStage::SoftmaxRows {
                        rows: *rows,
                        columns: *columns,
                    },
                    function,
                )? {
                    return Ok(Err(reason));
                }
            }
            Stage::Normalize { stage } => {
                let Some(Value::Object(layer)) = layers.data().get(stage.layer_index) else {
                    unreachable!("host plan admitted normalization layer")
                };
                let affine =
                    match normalization::admit_parameters(layer, provider, input, function)? {
                        Ok(parameters) => parameters,
                        Err(reason) => return Ok(Err(reason)),
                    };
                let request = stage.geometry.resident_request();
                let capability = provider
                    .deep_learning_layer_norm_planned_capability(
                        input,
                        affine.scale,
                        affine.offset,
                        &request,
                    )
                    .map_err(|error| {
                        provider_failure(function, "resident normalization admission", error.into())
                    })?;
                match capability {
                    DeepLearningLayerNormCapability::Ready => {}
                    DeepLearningLayerNormCapability::UnsupportedOperation => {
                        return Ok(Err(GpuCapabilityReason::ProviderOperation))
                    }
                    DeepLearningLayerNormCapability::ResourceLimit => {
                        return Ok(Err(GpuCapabilityReason::ResourceLimit))
                    }
                }
                if purpose == RoutePurpose::ResidentTape {
                    let reverse = provider
                        .deep_learning_layer_norm_adjoint_capability(
                            input,
                            affine.scale,
                            affine.offset,
                            None,
                            None,
                            &request,
                        )
                        .map_err(|error| {
                            provider_failure(
                                function,
                                "planned normalization adjoint admission",
                                error.into(),
                            )
                        })?;
                    match reverse {
                        DeepLearningLayerNormCapability::Ready => {}
                        DeepLearningLayerNormCapability::UnsupportedOperation => {
                            return Ok(Err(GpuCapabilityReason::ProviderOperation))
                        }
                        DeepLearningLayerNormCapability::ResourceLimit => {
                            return Ok(Err(GpuCapabilityReason::ResourceLimit))
                        }
                    }
                }
            }
            Stage::Pool {
                axis,
                input_axes,
                output_axes,
            } => {
                let route = pooling::route(input_axes, output_axes, output_axes, *axis, function)?;
                let capability = provider
                    .deep_learning_global_average_pool_1d_planned_capability(
                        input,
                        input_axes.shape(),
                        &route,
                    )
                    .map_err(|error| {
                        provider_failure(function, "resident pooling admission", error.into())
                    })?;
                match capability {
                    DeepLearningPool1DCapability::Ready => {}
                    DeepLearningPool1DCapability::UnsupportedOperation => {
                        return Ok(Err(GpuCapabilityReason::ProviderOperation))
                    }
                    DeepLearningPool1DCapability::ResourceLimit => {
                        return Ok(Err(GpuCapabilityReason::ResourceLimit))
                    }
                }
                if purpose == RoutePurpose::ResidentTape {
                    let reverse = provider
                        .deep_learning_global_average_pool_1d_adjoint_capability(
                            input,
                            input_axes.shape(),
                            None,
                            route.reduction,
                        )
                        .map_err(|error| {
                            provider_failure(
                                function,
                                "planned pooling adjoint admission",
                                error.into(),
                            )
                        })?;
                    match reverse {
                        DeepLearningPool1DCapability::Ready => {}
                        DeepLearningPool1DCapability::UnsupportedOperation => {
                            return Ok(Err(GpuCapabilityReason::ProviderOperation))
                        }
                        DeepLearningPool1DCapability::ResourceLimit => {
                            return Ok(Err(GpuCapabilityReason::ResourceLimit))
                        }
                    }
                }
                view = view
                    .with_shape(output_axes.shape().to_vec())
                    .with_layout(GpuTensorLayout::Contiguous);
            }
        }
    }
    if view.shape() != plan.output_axes.shape() {
        return Err(deep_learning_error(
            function,
            "provider stage output differs from lowered axes",
        ));
    }
    Ok(Ok(()))
}

/// Preserve the existing resident admission and third-party provider route.
/// The same lowered plan may also describe a proposed host upload, but this
/// function never invents an anchor handle for it.
pub(in crate::builtins::deep_learning::model::gpu::mixed) fn admit(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    admit_for_purpose(
        network,
        host,
        target_axes,
        RoutePurpose::ExistingMixedForward,
        function,
    )
}

pub(in crate::builtins::deep_learning::model::gpu::mixed) fn admit_for_tape(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    admit_for_purpose(
        network,
        host,
        target_axes,
        RoutePurpose::ResidentTape,
        function,
    )
}

fn admit_for_purpose(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    purpose: RoutePurpose,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    if !super::lowering::applicable(host, purpose) {
        return Ok(None);
    }
    let Some(input) = resident_input_handle(host.admitted_input()) else {
        return Ok(Some(Err(GpuCapabilityReason::InputStorage)));
    };
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| {
            super::super::super::admission::invalid_owner(
                function,
                super::super::super::admission::GpuOperand::Input,
            )
        })?;
    if input.shape() != host.input_axes().shape() {
        return Err(deep_learning_error(
            function,
            "resident input differs from admitted axes",
        ));
    }
    if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
        return Ok(Some(Err(GpuCapabilityReason::InputStorage)));
    }
    if runmat_accelerate_api::handle_precision(input).is_none()
        || runmat_accelerate_api::handle_is_logical(input)
        || runmat_accelerate_api::handle_integer_type(input).is_some()
    {
        return Ok(Some(Err(GpuCapabilityReason::InputPrecision)));
    }
    let lowered = match purpose {
        RoutePurpose::ExistingMixedForward => super::lower(network, host, target_axes, function)?,
        RoutePurpose::ResidentTape => {
            super::lowering::lower_for_tape(network, host, target_axes, function)?
        }
        RoutePurpose::HostPlacement => {
            super::lowering::lower_for_host_placement(network, host, target_axes, function)?
        }
    };
    let Some(plan) = lowered else {
        return Ok(None);
    };
    let plan = match plan {
        Ok(plan) => plan,
        Err(reason) => return Ok(Some(Err(reason))),
    };
    if let Err(reason) = admit_stages(network, host, input, &plan, provider, purpose, function)? {
        return Ok(Some(Err(reason)));
    }
    let Some(route) = super::super::physical::from_plan(host, &plan, function)? else {
        return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
    };
    match provider
        .deep_learning_forward_route_capability(&route)
        .map_err(|error| provider_failure(function, "physical network admission", error.into()))?
    {
        None | Some(DeepLearningForwardCapability::Ready(_)) => {}
        Some(DeepLearningForwardCapability::UnsupportedOperation) => {
            return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)))
        }
        Some(DeepLearningForwardCapability::ResourceLimit) => {
            return Ok(Some(Err(GpuCapabilityReason::ResourceLimit)))
        }
    }
    Ok(Some(Ok(plan)))
}
