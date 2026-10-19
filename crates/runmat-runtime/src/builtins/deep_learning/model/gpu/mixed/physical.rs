//! Project the selected executable mixed plan into the provider's physical
//! whole-route query. The executable `Stage` sequence remains the sole runtime
//! lowering; this adapter does not re-read layer class names or controls.

use super::super::*;
use super::plan::{Plan, Stage};
use crate::builtins::deep_learning::model::{
    admission::forward_data, parameters::ParameterRole, plan::host::HostNetworkPlan,
};
use runmat_accelerate_api::{
    DeepLearningFeedForwardStage as PhysicalStage, DeepLearningForwardOperation as Operation,
    DeepLearningForwardParameter, DeepLearningForwardParameters as Bindings,
    DeepLearningForwardRoute, DeepLearningForwardSource, DeepLearningForwardStep, GpuTensorLayout,
    NumericElementType, ProviderViewStep,
};
use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::NumericDType;

pub(super) fn admitted_source<'a>(
    host: &HostNetworkPlan<'a>,
) -> Option<DeepLearningForwardSource<'a>> {
    let element_type = match host.execution_precision() {
        NumericDType::F32 => NumericElementType::F32,
        NumericDType::F64 => NumericElementType::F64,
        _ => return None,
    };
    source(host.admitted_input(), element_type)
}

pub(super) fn source_shape(value: &Value) -> Option<&[usize]> {
    match forward_data(value) {
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(&[1, 1]),
        _ => None,
    }
}

fn source<'a>(
    value: &'a Value,
    element_type: NumericElementType,
) -> Option<DeepLearningForwardSource<'a>> {
    match forward_data(value) {
        Value::GpuTensor(handle) => Some(DeepLearningForwardSource::Resident(handle)),
        Value::Tensor(tensor) => Some(DeepLearningForwardSource::Host {
            element_type,
            shape: tensor.shape().to_vec(),
        }),
        Value::LogicalArray(array) => Some(DeepLearningForwardSource::Host {
            element_type,
            shape: array.shape.to_vec(),
        }),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(DeepLearningForwardSource::Host {
            element_type,
            shape: vec![1, 1],
        }),
        _ => None,
    }
}

fn parameter<'a>(
    host: &HostNetworkPlan<'a>,
    layer_index: usize,
    role: ParameterRole,
    element_type: NumericElementType,
    host_upload_shape: Option<Vec<usize>>,
    view_steps: Vec<ProviderViewStep>,
    parameters: &mut Vec<DeepLearningForwardParameter<'a>>,
) -> Option<usize> {
    let row = host
        .parameter_views()
        .iter()
        .find(|row| row.key.layer_index == layer_index && row.key.role == role)?;
    let mut source = source(row.value?, element_type)?;
    let view_steps = if let DeepLearningForwardSource::Host { shape, .. } = &mut source {
        if let Some(prepared) = host_upload_shape {
            *shape = prepared;
        }
        Vec::new()
    } else {
        view_steps
    };
    let index = parameters.len();
    parameters.push(DeepLearningForwardParameter { source, view_steps });
    Some(index)
}

fn step(operation: Operation, parameters: Bindings) -> DeepLearningForwardStep {
    DeepLearningForwardStep {
        operation,
        parameters,
    }
}

/// `None` means an admitted parameter cannot be prepared as a real numeric
/// physical operand. The caller rejects that route before any provider effect.
/// Providers without whole-route queries continue using the existing fully
/// admitted per-stage sequence for representable operands.
pub(super) fn from_plan<'a, 'h: 'a>(
    host: &HostNetworkPlan<'h>,
    plan: &Plan,
    function: &'static str,
) -> BuiltinResult<Option<DeepLearningForwardRoute<'a>>> {
    let Some(input) = admitted_source(host) else {
        return Ok(None);
    };
    let element_type = input.element_type();
    let mut view = input.view();
    let mut parameters = Vec::new();
    let mut steps = Vec::with_capacity(plan.stages.len());
    for selected in &plan.stages {
        match selected {
            Stage::Materialize => {
                steps.push(step(
                    Operation::View(ProviderViewStep::Reshape(view.shape().to_vec())),
                    Bindings::None,
                ));
                view = view.with_layout(GpuTensorLayout::Contiguous);
            }
            Stage::Transpose {
                output_axes,
                output_layout,
                materialize,
            } => {
                steps.push(step(
                    Operation::View(if *materialize {
                        ProviderViewStep::Permute2D
                    } else {
                        ProviderViewStep::Transpose2D
                    }),
                    Bindings::None,
                ));
                view = view
                    .with_shape(output_axes.shape().to_vec())
                    .with_layout(*output_layout);
            }
            Stage::Dense { layer_index } => {
                let [batch, input_channels] = view.shape() else {
                    return Err(deep_learning_error(
                        function,
                        "selected dense route is not a matrix",
                    ));
                };
                let (batch, input_channels) = (*batch, *input_channels);
                let network_step = host
                    .steps()
                    .iter()
                    .find(|step| step.index == *layer_index)
                    .expect("selected dense stage belongs to admitted network");
                let output_channels = network_step
                    .output_axes
                    .axis_size(DeepLearningAxisLabel::Channel)
                    .expect("admitted dense output has channel axis");
                let weights_row = host
                    .parameter_views()
                    .iter()
                    .find(|row| {
                        row.key.layer_index == *layer_index
                            && row.key.role == ParameterRole::Weights
                    })
                    .expect("admitted dense weights have a canonical parameter row");
                let Some(weights_value) = weights_row.value else {
                    return Ok(None);
                };
                let weights_shape = match forward_data(weights_value) {
                    Value::GpuTensor(handle) => handle.shape(),
                    Value::Tensor(tensor) => tensor.shape(),
                    Value::LogicalArray(array) => &array.shape,
                    Value::Num(_) | Value::Int(_) | Value::Bool(_) => &[1, 1],
                    _ => return Ok(None),
                };
                let mut weight_views = Vec::with_capacity(2);
                if weights_shape.len() == 1 {
                    weight_views.push(ProviderViewStep::Reshape(vec![1, weights_shape[0]]));
                }
                weight_views.push(ProviderViewStep::Transpose2D);
                let Some(weights) = parameter(
                    host,
                    *layer_index,
                    ParameterRole::Weights,
                    element_type,
                    Some(vec![input_channels, output_channels]),
                    weight_views,
                    &mut parameters,
                ) else {
                    return Ok(None);
                };
                let Some(bias) = parameter(
                    host,
                    *layer_index,
                    ParameterRole::Bias,
                    element_type,
                    Some(vec![1, output_channels]),
                    vec![ProviderViewStep::Reshape(vec![1, output_channels])],
                    &mut parameters,
                ) else {
                    return Ok(None);
                };
                steps.push(step(
                    Operation::FeedForward(PhysicalStage::Dense {
                        batch,
                        input_channels,
                        output_channels,
                    }),
                    Bindings::Dense { weights, bias },
                ));
                view = view
                    .with_shape(vec![batch, output_channels])
                    .with_layout(GpuTensorLayout::Contiguous);
            }
            Stage::Conv {
                layer_index,
                request,
            } => {
                let Some(weights) = parameter(
                    host,
                    *layer_index,
                    ParameterRole::Weights,
                    element_type,
                    Some(request.weights_shape.to_vec()),
                    Vec::new(),
                    &mut parameters,
                ) else {
                    return Ok(None);
                };
                let Some(bias) = parameter(
                    host,
                    *layer_index,
                    ParameterRole::Bias,
                    element_type,
                    Some(request.bias_shape.to_vec()),
                    Vec::new(),
                    &mut parameters,
                ) else {
                    return Ok(None);
                };
                steps.push(step(
                    Operation::Conv1D(request.clone()),
                    Bindings::Conv1D { weights, bias },
                ));
                view = view
                    .with_shape(request.output_shape.clone())
                    .with_layout(GpuTensorLayout::Contiguous);
            }
            Stage::Relu | Stage::Elu { .. } => {
                let elements = view
                    .shape()
                    .iter()
                    .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| {
                        deep_learning_error(function, "selected activation size overflows")
                    })?;
                let physical = match selected {
                    Stage::Relu => PhysicalStage::Relu { elements },
                    Stage::Elu { .. } => PhysicalStage::Elu { elements },
                    _ => unreachable!(),
                };
                steps.push(step(Operation::FeedForward(physical), Bindings::None));
            }
            Stage::Softmax => {
                let [rows, columns] = view.shape() else {
                    return Err(deep_learning_error(
                        function,
                        "selected softmax route is not a matrix",
                    ));
                };
                steps.push(step(
                    Operation::FeedForward(PhysicalStage::SoftmaxRows {
                        rows: *rows,
                        columns: *columns,
                    }),
                    Bindings::None,
                ));
            }
            Stage::Normalize { stage } => {
                let request = stage.geometry.resident_request();
                let Some(scale) = parameter(
                    host,
                    stage.layer_index,
                    ParameterRole::Scale,
                    element_type,
                    None,
                    Vec::new(),
                    &mut parameters,
                ) else {
                    return Ok(None);
                };
                let Some(offset) = parameter(
                    host,
                    stage.layer_index,
                    ParameterRole::Offset,
                    element_type,
                    None,
                    Vec::new(),
                    &mut parameters,
                ) else {
                    return Ok(None);
                };
                steps.push(step(
                    Operation::LayerNorm(request),
                    Bindings::LayerNorm { scale, offset },
                ));
            }
            Stage::Pool {
                axis,
                input_axes,
                output_axes,
            } => {
                let route = pooling::route(input_axes, output_axes, output_axes, *axis, function)?;
                steps.push(step(
                    Operation::Pool1D {
                        input_shape: input_axes.shape().to_vec(),
                        route,
                    },
                    Bindings::None,
                ));
                view = view
                    .with_shape(output_axes.shape().to_vec())
                    .with_layout(GpuTensorLayout::Contiguous);
            }
        }
    }
    if view.shape() != plan.output_axes.shape() {
        return Err(deep_learning_error(
            function,
            "selected route output differs from admitted axes",
        ));
    }
    let route = DeepLearningForwardRoute {
        input,
        parameters,
        steps,
        output_view: view,
    };
    Ok(Some(route))
}
