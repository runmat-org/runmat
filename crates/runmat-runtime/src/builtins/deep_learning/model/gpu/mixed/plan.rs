//! Lower one admitted host network to physical resident stages. Every
//! capability query completes before the executor receives this plan.

use super::super::*;
use super::parameters;
use crate::builtins::deep_learning::model::plan::{
    host::{Controls, HostNetworkPlan},
    normalization::{Geometry, NormalizationStage},
};
use runmat_accelerate_api::{
    DeepLearningFeedForwardStage as PhysicalStage, DeepLearningLayerNormCapability,
    DeepLearningPool1DCapability, GpuTensorLayout, GpuTensorView, ProviderViewCapability,
    ProviderViewChain, ProviderViewSource, ProviderViewStep,
};
use runmat_builtins::{DeepLearningAxisLabel as Label, DeepLearningAxisLayout};

pub(super) enum Stage {
    Materialize,
    Transpose {
        output_axes: AxisPlan,
        output_layout: GpuTensorLayout,
        materialize: bool,
    },
    Dense {
        layer_index: usize,
    },
    Relu,
    Elu {
        alpha: f64,
    },
    Softmax,
    Normalize {
        stage: NormalizationStage,
    },
    Pool {
        axis: usize,
        input_axes: AxisPlan,
        output_axes: AxisPlan,
    },
}

pub(super) struct Plan {
    pub stages: Vec<Stage>,
    pub output_axes: AxisPlan,
    pub consumed_resident_parameters: Vec<GpuTensorHandle>,
}

fn query(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    anchor: &GpuTensorHandle,
    stage: PhysicalStage,
    function: &'static str,
) -> BuiltinResult<Result<(), GpuCapabilityReason>> {
    Ok(
        match admission::query_stage(provider, anchor, stage, function)? {
            Some(reason) => Err(reason),
            None => Ok(()),
        },
    )
}

fn query_view(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    anchor: &GpuTensorHandle,
    source: &GpuTensorView,
    step: ProviderViewStep,
    function: &'static str,
) -> BuiltinResult<Result<GpuTensorView, GpuCapabilityReason>> {
    let request = ProviderViewChain {
        source: ProviderViewSource::Planned(source.clone()),
        steps: vec![step],
    };
    let capability = provider
        .provider_view_chain_capability(anchor, &request)
        .map_err(|error| provider_failure(function, "physical view admission", error.into()))?;
    Ok(match capability {
        ProviderViewCapability::Ready(outcome) => Ok(outcome.view),
        ProviderViewCapability::UnsupportedOperation => Err(GpuCapabilityReason::ProviderOperation),
        ProviderViewCapability::ResourceLimit => Err(GpuCapabilityReason::ResourceLimit),
    })
}

fn ensure_contiguous(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    anchor: &GpuTensorHandle,
    view: &mut GpuTensorView,
    stages: &mut Vec<Stage>,
    function: &'static str,
) -> BuiltinResult<Result<(), GpuCapabilityReason>> {
    if matches!(view.layout(), GpuTensorLayout::Contiguous) {
        return Ok(Ok(()));
    }
    *view = match query_view(
        provider,
        anchor,
        view,
        ProviderViewStep::Reshape(view.shape().to_vec()),
        function,
    )? {
        Ok(view) => view,
        Err(reason) => return Ok(Err(reason)),
    };
    stages.push(Stage::Materialize);
    Ok(Ok(()))
}

fn batch_channels() -> DeepLearningAxisLayout {
    DeepLearningAxisLayout::from_axes(&[Label::Batch, Label::Channel])
        .expect("two distinct canonical axes")
}

fn transpose_to_bc(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    anchor: &GpuTensorHandle,
    current: &AxisPlan,
    view: &mut GpuTensorView,
    stages: &mut Vec<Stage>,
    materialize: bool,
    function: &'static str,
) -> BuiltinResult<Result<AxisPlan, GpuCapabilityReason>> {
    if current.layout().axes() == [Label::Batch, Label::Channel] {
        return Ok(Ok(current.clone()));
    }
    if current.layout().axes() != [Label::Channel, Label::Batch] {
        return Ok(Err(GpuCapabilityReason::AxisLayout));
    }
    let target = batch_channels();
    let lowering = current.lower_to(&target).map_err(|error| {
        deep_learning_error(
            function,
            format!("{function}: invalid resident axes: {error:?}"),
        )
    })?;
    let output_axes = AxisPlan::new(target, lowering.target_shape()).map_err(|error| {
        deep_learning_error(
            function,
            format!("{function}: invalid resident axes: {error:?}"),
        )
    })?;
    *view = match query_view(
        provider,
        anchor,
        view,
        if materialize {
            ProviderViewStep::Permute2D
        } else {
            ProviderViewStep::Transpose2D
        },
        function,
    )? {
        Ok(view) => view,
        Err(reason) => return Ok(Err(reason)),
    };
    if view.shape() != output_axes.shape() {
        return Err(deep_learning_error(
            function,
            "physical view transition differs from admitted axes",
        ));
    }
    stages.push(Stage::Transpose {
        output_axes: output_axes.clone(),
        output_layout: view.layout(),
        materialize,
    });
    Ok(Ok(output_axes))
}

fn semantic_shape_matches(semantic: &AxisPlan, physical: &AxisPlan) -> bool {
    semantic
        .lower_to(physical.layout())
        .is_ok_and(|lowering| lowering.target_shape() == physical.shape())
}

/// `None` leaves the existing single-operation routes untouched. Every
/// resident parameter view used by a dense step is admitted from its live
/// owner before any network stage can execute.
pub(super) fn admit(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    input: &GpuTensorHandle,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    let steps = host.steps();
    let transformations = steps
        .iter()
        .filter(|step| !matches!(step.controls, Controls::Input { .. } | Controls::Output))
        .count();
    let combines_pool_or_norm = steps.iter().any(|step| {
        matches!(
            step.controls,
            Controls::Normalization(_) | Controls::Pooling
        )
    });
    if transformations < 2
        || !combines_pool_or_norm
        || parameters::has_other_resident_parameters(host)
    {
        return Ok(None);
    }
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| admission::invalid_owner(function, admission::GpuOperand::Input))?;
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
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        unreachable!("host plan admitted Layers")
    };
    let mut physical = host.input_axes().clone();
    let mut view = input.view().clone();
    let mut stages = Vec::with_capacity(steps.len() + 2);
    for step in steps {
        if !semantic_shape_matches(&step.input_axes, &physical) {
            return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
        }
        match step.controls {
            Controls::Input { .. } | Controls::Output => {}
            Controls::Dense => {
                physical = match transpose_to_bc(
                    provider,
                    input,
                    &physical,
                    &mut view,
                    &mut stages,
                    false,
                    function,
                )? {
                    Ok(axes) => axes,
                    Err(reason) => return Ok(Some(Err(reason))),
                };
                let batch = physical.axis_size(Label::Batch).expect("BC has batch");
                let input_channels = physical.axis_size(Label::Channel).expect("BC has channels");
                let output_channels = step
                    .output_axes
                    .axis_size(Label::Channel)
                    .expect("host-admitted dense output has channels");
                if let Err(reason) = parameters::admit_dense(
                    host,
                    input,
                    step.index,
                    input_channels,
                    output_channels,
                    function,
                )? {
                    return Ok(Some(Err(reason)));
                }
                if let Err(reason) = query(
                    provider,
                    input,
                    PhysicalStage::Dense {
                        batch,
                        input_channels,
                        output_channels,
                    },
                    function,
                )? {
                    return Ok(Some(Err(reason)));
                }
                physical = physical
                    .with_channel_extent(output_channels)
                    .map_err(|error| {
                        deep_learning_error(
                            function,
                            format!("{function}: invalid dense axes: {error:?}"),
                        )
                    })?;
                view = view
                    .with_shape(physical.shape().to_vec())
                    .with_layout(GpuTensorLayout::Contiguous);
                stages.push(Stage::Dense {
                    layer_index: step.index,
                });
            }
            Controls::Relu | Controls::Elu { .. } => {
                if let Err(reason) =
                    ensure_contiguous(provider, input, &mut view, &mut stages, function)?
                {
                    return Ok(Some(Err(reason)));
                }
                let stage = match step.controls {
                    Controls::Relu => PhysicalStage::Relu {
                        elements: physical.element_count(),
                    },
                    Controls::Elu { .. } => PhysicalStage::Elu {
                        elements: physical.element_count(),
                    },
                    _ => unreachable!(),
                };
                if let Err(reason) = query(provider, input, stage, function)? {
                    return Ok(Some(Err(reason)));
                }
                stages.push(match step.controls {
                    Controls::Relu => Stage::Relu,
                    Controls::Elu { alpha } => Stage::Elu { alpha },
                    _ => unreachable!(),
                });
            }
            Controls::Softmax => {
                physical = match transpose_to_bc(
                    provider,
                    input,
                    &physical,
                    &mut view,
                    &mut stages,
                    true,
                    function,
                )? {
                    Ok(axes) => axes,
                    Err(reason) => return Ok(Some(Err(reason))),
                };
                if let Err(reason) =
                    ensure_contiguous(provider, input, &mut view, &mut stages, function)?
                {
                    return Ok(Some(Err(reason)));
                }
                if let Err(reason) = query(
                    provider,
                    input,
                    PhysicalStage::SoftmaxRows {
                        rows: physical.shape()[0],
                        columns: physical.shape()[1],
                    },
                    function,
                )? {
                    return Ok(Some(Err(reason)));
                }
                stages.push(Stage::Softmax);
            }
            Controls::Normalization(options) => {
                if let Err(reason) =
                    ensure_contiguous(provider, input, &mut view, &mut stages, function)?
                {
                    return Ok(Some(Err(reason)));
                }
                let geometry = match Geometry::new(&physical, options) {
                    Ok(geometry) => geometry,
                    Err(_) => return Ok(Some(Err(GpuCapabilityReason::AxisLayout))),
                };
                let request = geometry.resident_request();
                let capability = provider
                    .deep_learning_layer_norm_planned_capability(input, None, None, &request)
                    .map_err(|error| {
                        provider_failure(function, "resident normalization admission", error.into())
                    })?;
                match capability {
                    DeepLearningLayerNormCapability::Ready => {}
                    DeepLearningLayerNormCapability::UnsupportedOperation => {
                        return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
                    }
                    DeepLearningLayerNormCapability::ResourceLimit => {
                        return Ok(Some(Err(GpuCapabilityReason::ResourceLimit)));
                    }
                }
                stages.push(Stage::Normalize {
                    stage: NormalizationStage {
                        layer_index: step.index,
                        geometry,
                    },
                });
            }
            Controls::Pooling => {
                if let Err(reason) =
                    ensure_contiguous(provider, input, &mut view, &mut stages, function)?
                {
                    return Ok(Some(Err(reason)));
                }
                if physical != step.input_axes {
                    return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
                }
                let Some(axis) = physical
                    .layout()
                    .index(Label::Spatial)
                    .or_else(|| physical.layout().index(Label::Time))
                else {
                    return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
                };
                let route = pooling::route(
                    &physical,
                    &step.output_axes,
                    &step.output_axes,
                    axis,
                    function,
                )?;
                let capability = provider
                    .deep_learning_global_average_pool_1d_planned_capability(
                        input,
                        physical.shape(),
                        &route,
                    )
                    .map_err(|error| {
                        provider_failure(function, "resident pooling admission", error.into())
                    })?;
                match capability {
                    DeepLearningPool1DCapability::Ready => {}
                    DeepLearningPool1DCapability::UnsupportedOperation => {
                        return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
                    }
                    DeepLearningPool1DCapability::ResourceLimit => {
                        return Ok(Some(Err(GpuCapabilityReason::ResourceLimit)));
                    }
                }
                stages.push(Stage::Pool {
                    axis,
                    input_axes: physical,
                    output_axes: step.output_axes.clone(),
                });
                physical = step.output_axes.clone();
                view = view
                    .with_shape(physical.shape().to_vec())
                    .with_layout(GpuTensorLayout::Contiguous);
            }
            Controls::Convolution(_) | Controls::Recurrent(_) => {
                let Value::Object(layer) = &layers.data[step.index] else {
                    unreachable!("host plan admitted layer objects")
                };
                return Ok(Some(Err(GpuCapabilityReason::Layer(
                    layer.class_name.clone(),
                ))));
            }
        }
    }
    if physical != *target_axes {
        let lowering = match physical.lower_to(target_axes.layout()) {
            Ok(lowering) if lowering.target_shape() == target_axes.shape() => lowering,
            _ => return Ok(Some(Err(GpuCapabilityReason::AxisLayout))),
        };
        if physical.shape().len() != 2 || lowering.target_to_source() != [1, 0] {
            return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
        }
        view = match query_view(
            provider,
            input,
            &view,
            ProviderViewStep::Transpose2D,
            function,
        )? {
            Ok(view) => view,
            Err(reason) => return Ok(Some(Err(reason))),
        };
        if view.shape() != target_axes.shape() {
            return Err(deep_learning_error(
                function,
                "physical output view differs from admitted axes",
            ));
        }
        stages.push(Stage::Transpose {
            output_axes: target_axes.clone(),
            output_layout: view.layout(),
            materialize: false,
        });
        physical = target_axes.clone();
    }
    let consumed_resident_parameters =
        parameters::consumed_resident_handles(host.parameter_views(), &stages);
    Ok(Some(Ok(Plan {
        stages,
        output_axes: physical,
        consumed_resident_parameters,
    })))
}
