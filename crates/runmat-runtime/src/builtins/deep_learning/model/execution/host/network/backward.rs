use super::super::{activation, conv, dense, normalization, pooling, recurrent, softmax};
use super::types::{HostError, HostTrace, HostValue, PinnedStep, StepCache};
use crate::builtins::deep_learning::model::state::{StateRole, StateUpdate};

pub(in crate::builtins::deep_learning) enum StepGradients {
    None,
    FullyConnected {
        weights: runmat_value::Tensor,
        bias: runmat_value::Tensor,
    },
    Convolution1d {
        weights: runmat_value::Tensor,
        bias: runmat_value::Tensor,
    },
    LayerNormalization {
        scale: runmat_value::Tensor,
        offset: runmat_value::Tensor,
    },
    Recurrent {
        input_weights: runmat_value::Tensor,
        recurrent_weights: runmat_value::Tensor,
        bias: runmat_value::Tensor,
        initial_hidden: runmat_value::Tensor,
        initial_cell: runmat_value::Tensor,
    },
}

pub(in crate::builtins::deep_learning) struct HostBackwardResult {
    pub input_gradient: HostValue,
    /// Numerical step index maps back to the caller's typed layer/parameter
    /// roles. No parameter is stored or mutated here.
    pub parameter_gradients: Vec<(usize, StepGradients)>,
}

pub(in crate::builtins::deep_learning) fn backward(
    trace: &HostTrace,
    output_gradient: HostValue,
    state_gradients: &[StateUpdate],
) -> Result<HostBackwardResult, HostError> {
    if !matches_axes(&output_gradient, &trace.output_axes) {
        return Err(HostError::TraceMismatch);
    }
    for (index, state) in state_gradients.iter().enumerate() {
        if !trace.records.iter().any(|record| {
            record.step_index == state.layer_index
                && matches!(&record.step, PinnedStep::Recurrent { .. })
        }) || state_gradients[..index]
            .iter()
            .any(|prior| prior.layer_index == state.layer_index && prior.role == state.role)
        {
            return Err(HostError::TraceMismatch);
        }
    }
    let mut gradient = output_gradient;
    let mut parameter_gradients = Vec::new();
    for record in trace.records.iter().rev() {
        if !matches_axes(&gradient, &record.output_axes) {
            return Err(HostError::TraceMismatch);
        }
        let step = &record.step;
        let (next, parameters) = match (step, &record.input, &gradient, &record.cache) {
            (
                PinnedStep::FullyConnected { weights, bias },
                HostValue::Real(input),
                HostValue::Real(dy),
                StepCache::None,
            ) => {
                let result = dense::backward(input, weights, bias, &record.input_axes, dy)?;
                (
                    HostValue::Real(result.input),
                    StepGradients::FullyConnected {
                        weights: result.weights,
                        bias: result.bias,
                    },
                )
            }
            (PinnedStep::Relu, HostValue::Real(input), HostValue::Real(dy), StepCache::None) => (
                HostValue::Real(activation::backward(
                    input,
                    dy,
                    activation::Activation::Relu,
                )?),
                StepGradients::None,
            ),
            (
                PinnedStep::Elu { alpha },
                HostValue::Real(input),
                HostValue::Real(dy),
                StepCache::None,
            ) => (
                HostValue::Real(activation::backward(
                    input,
                    dy,
                    activation::Activation::Elu { alpha: *alpha },
                )?),
                StepGradients::None,
            ),
            (
                PinnedStep::Softmax,
                HostValue::Real(_),
                HostValue::Real(dy),
                StepCache::Softmax(y),
            ) => (
                HostValue::Real(softmax::backward(y, dy, &record.input_axes)?),
                StepGradients::None,
            ),
            (
                PinnedStep::Convolution1d {
                    weights,
                    bias,
                    options,
                },
                HostValue::Real(input),
                HostValue::Real(dy),
                StepCache::None,
            ) => {
                let result =
                    conv::backward(input, weights, bias, dy, &record.input_axes, *options)?;
                (
                    HostValue::Real(result.input),
                    StepGradients::Convolution1d {
                        weights: result.weights,
                        bias: result.bias,
                    },
                )
            }
            (
                PinnedStep::LayerNormalization {
                    scale,
                    offset,
                    options,
                },
                HostValue::Real(input),
                HostValue::Real(dy),
                StepCache::Normalization(stats),
            ) => {
                let result = normalization::backward(
                    input,
                    scale,
                    offset,
                    dy,
                    &record.input_axes,
                    *options,
                    stats,
                )?;
                (
                    HostValue::Real(result.input),
                    StepGradients::LayerNormalization {
                        scale: result.scale,
                        offset: result.offset,
                    },
                )
            }
            (
                PinnedStep::GlobalAveragePooling1d,
                HostValue::Real(_input),
                HostValue::Real(dy),
                StepCache::None,
            ) => (
                HostValue::Real(pooling::backward(dy, &record.input_axes)?),
                StepGradients::None,
            ),
            (
                PinnedStep::GlobalAveragePooling1d,
                HostValue::Complex(input),
                HostValue::Complex(dy),
                StepCache::None,
            ) => (
                HostValue::Complex(pooling::backward_complex(input, dy, &record.input_axes)?),
                StepGradients::None,
            ),
            (
                PinnedStep::Recurrent {
                    input_weights,
                    recurrent_weights,
                    bias,
                    initial_hidden,
                    initial_cell,
                    options,
                },
                HostValue::Real(input),
                HostValue::Real(dy),
                StepCache::Recurrent(cache),
            ) => {
                let result = recurrent::backward(
                    input,
                    input_weights,
                    recurrent_weights,
                    bias,
                    initial_hidden.as_ref(),
                    initial_cell.as_ref(),
                    &record.input_axes,
                    *options,
                    cache,
                    dy,
                    state_gradients
                        .iter()
                        .find(|state| {
                            state.layer_index == record.step_index
                                && state.role == StateRole::HiddenState
                        })
                        .map(|state| &state.value),
                    state_gradients
                        .iter()
                        .find(|state| {
                            state.layer_index == record.step_index
                                && state.role == StateRole::CellState
                        })
                        .map(|state| &state.value),
                )?;
                (
                    HostValue::Real(result.input),
                    StepGradients::Recurrent {
                        input_weights: result.input_weights,
                        recurrent_weights: result.recurrent_weights,
                        bias: result.bias,
                        initial_hidden: result.initial_hidden,
                        initial_cell: result.initial_cell,
                    },
                )
            }
            _ => return Err(HostError::TraceMismatch),
        };
        parameter_gradients.push((record.step_index, parameters));
        gradient = next;
    }
    parameter_gradients.reverse();
    Ok(HostBackwardResult {
        input_gradient: gradient,
        parameter_gradients,
    })
}

fn matches_axes(
    value: &HostValue,
    axes: &crate::builtins::deep_learning::model::plan::axes::AxisPlan,
) -> bool {
    match value {
        HostValue::Real(tensor) => {
            tensor.shape() == axes.shape() && tensor.len() == axes.element_count()
        }
        HostValue::Complex(tensor) => {
            tensor.shape() == axes.shape() && tensor.len() == axes.element_count()
        }
    }
}
