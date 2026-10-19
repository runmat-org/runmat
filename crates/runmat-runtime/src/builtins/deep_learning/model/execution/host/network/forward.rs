use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::super::{activation, conv, dense, normalization, pooling, recurrent, softmax};
use super::types::{
    HostError, HostExecutionResult, HostStep, HostTrace, HostValue, PinnedStep, StepCache,
    StepRecord,
};
use crate::builtins::deep_learning::model::state::{StateRole, StateUpdate};

/// Execute a previously admitted network. Only CPU-owned tensors may reach
/// this entrypoint; the public wrapper handles device provenance and routing.
pub(in crate::builtins::deep_learning) fn execute(
    steps: &[HostStep<'_>],
    input: HostValue,
    input_axes: AxisPlan,
    capture: bool,
) -> Result<HostExecutionResult, HostError> {
    if !matches_payload_axes(&input, &input_axes) {
        return Err(HostError::Invalid(
            "host input payload does not match its admitted axes",
        ));
    }
    let mut value = input;
    let mut axes = input_axes;
    let mut records = Vec::new();
    let mut next_state = Vec::new();
    for (step_index, step) in steps.iter().enumerate() {
        if matches!(
            step,
            HostStep::Input | HostStep::ClassificationOutput | HostStep::RegressionOutput
        ) {
            continue;
        }
        let (output, output_axes, cache, state) = apply(step, &value, &axes, capture)?;
        if capture {
            records.push(StepRecord {
                step_index,
                step: PinnedStep::capture(step).expect("numerical step is not identity"),
                input: value,
                input_axes: axes,
                output_axes: output_axes.clone(),
                cache,
            });
        }
        next_state.extend(state.into_iter().map(|(role, value)| StateUpdate {
            layer_index: step_index,
            role,
            value,
        }));
        value = output;
        axes = output_axes;
    }
    Ok(HostExecutionResult {
        value,
        axes: axes.clone(),
        trace: capture.then_some(HostTrace {
            records,
            output_axes: axes,
        }),
        next_state,
    })
}

fn matches_payload_axes(value: &HostValue, axes: &AxisPlan) -> bool {
    match value {
        HostValue::Real(tensor) => {
            tensor.shape() == axes.shape() && tensor.len() == axes.element_count()
        }
        HostValue::Complex(tensor) => {
            tensor.shape() == axes.shape() && tensor.len() == axes.element_count()
        }
    }
}

type Applied = (
    HostValue,
    AxisPlan,
    StepCache,
    Vec<(StateRole, runmat_value::Tensor)>,
);

fn apply(
    step: &HostStep<'_>,
    value: &HostValue,
    axes: &AxisPlan,
    capture: bool,
) -> Result<Applied, HostError> {
    match (step, value) {
        (HostStep::FullyConnected { weights, bias }, HostValue::Real(input)) => {
            let output = dense::forward(input, weights, bias, axes)?;
            Ok((
                HostValue::Real(output.tensor),
                output.axes,
                StepCache::None,
                Vec::new(),
            ))
        }
        (HostStep::Relu, HostValue::Real(input)) => {
            let output = activation::forward(input, activation::Activation::Relu)?;
            Ok((
                HostValue::Real(output),
                axes.clone(),
                StepCache::None,
                Vec::new(),
            ))
        }
        (HostStep::Elu { alpha }, HostValue::Real(input)) => {
            let output = activation::forward(input, activation::Activation::Elu { alpha: *alpha })?;
            Ok((
                HostValue::Real(output),
                axes.clone(),
                StepCache::None,
                Vec::new(),
            ))
        }
        (HostStep::Softmax, HostValue::Real(input)) => {
            let output = softmax::forward(input, axes)?;
            let cache = if capture {
                StepCache::Softmax(output.clone())
            } else {
                StepCache::None
            };
            Ok((HostValue::Real(output), axes.clone(), cache, Vec::new()))
        }
        (
            HostStep::Convolution1d {
                weights,
                bias,
                options,
            },
            HostValue::Real(input),
        ) => {
            let output = conv::forward(input, weights, bias, axes, *options)?;
            Ok((
                HostValue::Real(output.tensor),
                output.axes,
                StepCache::None,
                Vec::new(),
            ))
        }
        (
            HostStep::LayerNormalization {
                scale,
                offset,
                options,
            },
            HostValue::Real(input),
        ) => {
            let output = normalization::forward(input, scale, offset, axes, *options)?;
            Ok((
                HostValue::Real(output.tensor),
                output.axes,
                StepCache::Normalization(output.stats),
                Vec::new(),
            ))
        }
        (HostStep::GlobalAveragePooling1d, HostValue::Real(input)) => {
            let output = pooling::forward(input, axes)?;
            Ok((
                HostValue::Real(output.tensor),
                output.axes,
                StepCache::None,
                Vec::new(),
            ))
        }
        (HostStep::GlobalAveragePooling1d, HostValue::Complex(input)) => {
            let output = pooling::forward_complex(input, axes)?;
            Ok((
                HostValue::Complex(output.tensor),
                output.axes,
                StepCache::None,
                Vec::new(),
            ))
        }
        (
            HostStep::Recurrent {
                input_weights,
                recurrent_weights,
                bias,
                initial_hidden,
                initial_cell,
                options,
            },
            HostValue::Real(input),
        ) => {
            let mut output = recurrent::forward(
                input,
                input_weights,
                recurrent_weights,
                bias,
                *initial_hidden,
                *initial_cell,
                axes,
                *options,
                capture,
            )?;
            let cache = output
                .cache
                .take()
                .map(StepCache::Recurrent)
                .unwrap_or(StepCache::None);
            let state = vec![
                (StateRole::HiddenState, output.hidden_state),
                (StateRole::CellState, output.cell_state),
            ];
            Ok((HostValue::Real(output.tensor), output.axes, cache, state))
        }
        (HostStep::Input | HostStep::ClassificationOutput | HostStep::RegressionOutput, _) => Err(
            HostError::Invalid("identity steps do not invoke numerical execution"),
        ),
        (_, HostValue::Complex(_)) => Err(HostError::Invalid(
            "this layer has no admitted complex host kernel",
        )),
    }
}
