//! Pure Controls-to-physical-stage lowering. Both input origins use this one
//! sequence; no provider handle, upload, query, or timing is needed here.

use super::*;
use crate::builtins::deep_learning::model::execution::host::conv::{resident_request, ConvError};
use crate::builtins::deep_learning::model::plan::{
    host::{Controls, HostNetworkPlan},
    normalization::{Geometry, NormalizationStage},
};
use runmat_accelerate_api::{
    GpuTensorView, NumericElementType, ProviderViewChain, ProviderViewPlanError,
    ProviderViewSource, ProviderViewStep,
};
use runmat_builtins::{DeepLearningAxisLabel as Label, DeepLearningAxisLayout};

fn transition(
    current: &GpuTensorView,
    operation: ProviderViewStep,
    element: NumericElementType,
) -> Result<GpuTensorView, GpuCapabilityReason> {
    ProviderViewChain {
        source: ProviderViewSource::Current,
        steps: vec![operation],
    }
    .analyze(current, element)
    .map(|outcome| outcome.view)
    .map_err(|error| match error {
        ProviderViewPlanError::SizeOverflow => GpuCapabilityReason::ResourceLimit,
        ProviderViewPlanError::InvalidSource | ProviderViewPlanError::InvalidTransition => {
            GpuCapabilityReason::AxisLayout
        }
    })
}

fn ensure_contiguous(
    view: &mut GpuTensorView,
    stages: &mut Vec<Stage>,
    element: NumericElementType,
) -> Result<(), GpuCapabilityReason> {
    if view.layout() != GpuTensorLayout::Contiguous {
        *view = transition(
            view,
            ProviderViewStep::Reshape(view.shape().to_vec()),
            element,
        )?;
        stages.push(Stage::Materialize);
    }
    Ok(())
}

fn batch_channels() -> DeepLearningAxisLayout {
    DeepLearningAxisLayout::from_axes(&[Label::Batch, Label::Channel])
        .expect("two distinct canonical axes")
}

fn transpose_to_bc(
    current: &AxisPlan,
    view: &mut GpuTensorView,
    stages: &mut Vec<Stage>,
    element: NumericElementType,
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
    *view = match transition(
        view,
        if materialize {
            ProviderViewStep::Permute2D
        } else {
            ProviderViewStep::Transpose2D
        },
        element,
    ) {
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

pub(super) fn applicable(host: &HostNetworkPlan<'_>, purpose: RoutePurpose) -> bool {
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
    let selected = match purpose {
        RoutePurpose::ExistingMixedForward => transformations >= 2 && combines_pool_or_norm,
        RoutePurpose::ResidentTape => true,
        RoutePurpose::HostPlacement => true,
    };
    selected && !super::super::parameters::has_unsupported_resident_parameters(host)
}

/// Return `None` for networks owned by the existing standalone routes. A
/// returned plan is geometry only: the provider must still admit each live
/// operand and the complete route before execution.
pub(in crate::builtins::deep_learning::model::gpu::mixed) fn lower(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    lower_for_purpose(
        network,
        host,
        target_axes,
        RoutePurpose::ExistingMixedForward,
        function,
    )
}

pub(in crate::builtins::deep_learning::model::gpu::mixed) fn lower_for_tape(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    lower_for_purpose(
        network,
        host,
        target_axes,
        RoutePurpose::ResidentTape,
        function,
    )
}

pub(in crate::builtins::deep_learning::model::gpu::mixed) fn lower_for_host_placement(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    lower_for_purpose(
        network,
        host,
        target_axes,
        RoutePurpose::HostPlacement,
        function,
    )
}

fn lower_for_purpose(
    network: &ObjectInstance,
    host: &HostNetworkPlan<'_>,
    target_axes: &AxisPlan,
    purpose: RoutePurpose,
    function: &'static str,
) -> BuiltinResult<Option<Result<Plan, GpuCapabilityReason>>> {
    if !applicable(host, purpose) {
        return Ok(None);
    }
    let steps = host.steps();
    let Some(source) = super::super::physical::admitted_source(host) else {
        return Ok(Some(Err(GpuCapabilityReason::InputPrecision)));
    };
    let source_view = source.view();
    let element = source.element_type();
    if source_view.shape() != host.input_axes().shape() {
        return Err(deep_learning_error(
            function,
            "physical input differs from admitted axes",
        ));
    }
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        unreachable!("host plan admitted Layers")
    };
    let mut physical = host.input_axes().clone();
    let mut view = source_view;
    let mut stages = Vec::with_capacity(steps.len() + 2);
    for step in steps {
        if !semantic_shape_matches(&step.input_axes, &physical) {
            return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
        }
        match step.controls {
            Controls::Input { .. } | Controls::Output => {}
            Controls::Dense => {
                physical = match transpose_to_bc(
                    &physical,
                    &mut view,
                    &mut stages,
                    element,
                    false,
                    function,
                )? {
                    Ok(axes) => axes,
                    Err(reason) => return Ok(Some(Err(reason))),
                };
                let output_channels = step
                    .output_axes
                    .axis_size(Label::Channel)
                    .expect("host-admitted dense output has channels");
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
                if let Err(reason) = ensure_contiguous(&mut view, &mut stages, element) {
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
                    &physical,
                    &mut view,
                    &mut stages,
                    element,
                    true,
                    function,
                )? {
                    Ok(axes) => axes,
                    Err(reason) => return Ok(Some(Err(reason))),
                };
                if let Err(reason) = ensure_contiguous(&mut view, &mut stages, element) {
                    return Ok(Some(Err(reason)));
                }
                stages.push(Stage::Softmax);
            }
            Controls::Normalization(options) => {
                if let Err(reason) = ensure_contiguous(&mut view, &mut stages, element) {
                    return Ok(Some(Err(reason)));
                }
                let geometry = match Geometry::new(&physical, options) {
                    Ok(geometry) => geometry,
                    Err(_) => return Ok(Some(Err(GpuCapabilityReason::AxisLayout))),
                };
                stages.push(Stage::Normalize {
                    stage: NormalizationStage {
                        layer_index: step.index,
                        geometry,
                    },
                });
            }
            Controls::Pooling => {
                if let Err(reason) = ensure_contiguous(&mut view, &mut stages, element) {
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
            Controls::Convolution(options) => {
                if let Err(reason) = ensure_contiguous(&mut view, &mut stages, element) {
                    return Ok(Some(Err(reason)));
                }
                if physical != step.input_axes {
                    return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
                }
                let parameter_shape = |role| {
                    host.parameter_views()
                        .iter()
                        .find(|row| row.key.layer_index == step.index && row.key.role == role)
                        .and_then(|row| row.value)
                        .and_then(|value| super::super::physical::source_shape(value))
                };
                let Some(weights_shape) = parameter_shape(
                    crate::builtins::deep_learning::model::parameters::ParameterRole::Weights,
                ) else {
                    return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
                };
                let Some(bias_shape) = parameter_shape(
                    crate::builtins::deep_learning::model::parameters::ParameterRole::Bias,
                ) else {
                    return Ok(Some(Err(GpuCapabilityReason::ProviderOperation)));
                };
                let request = match resident_request(
                    &physical,
                    weights_shape,
                    bias_shape,
                    host.execution_precision(),
                    options,
                ) {
                    Ok(request) => request,
                    Err(ConvError::ShapeOverflow) => {
                        return Ok(Some(Err(GpuCapabilityReason::ResourceLimit)));
                    }
                    Err(ConvError::UnsupportedPrecision) => {
                        return Ok(Some(Err(GpuCapabilityReason::InputPrecision)));
                    }
                    Err(_) => return Ok(Some(Err(GpuCapabilityReason::AxisLayout))),
                };
                if request.output_shape.as_slice() != step.output_axes.shape() {
                    return Ok(Some(Err(GpuCapabilityReason::AxisLayout)));
                }
                stages.push(Stage::Conv {
                    layer_index: step.index,
                    request,
                });
                physical = step.output_axes.clone();
                view = view
                    .with_shape(physical.shape().to_vec())
                    .with_layout(GpuTensorLayout::Contiguous);
            }
            Controls::Recurrent(_) => {
                let Value::Object(layer) = &layers.data()[step.index] else {
                    unreachable!("host plan admitted layer objects")
                };
                return Ok(Some(Err(GpuCapabilityReason::Layer(
                    layer.class_name().clone(),
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
        view = match transition(&view, ProviderViewStep::Transpose2D, element) {
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
        super::super::parameters::consumed_resident_handles(host.parameter_views(), &stages);
    Ok(Some(Ok(Plan {
        stages,
        output_axes: physical,
        consumed_resident_parameters,
    })))
}
