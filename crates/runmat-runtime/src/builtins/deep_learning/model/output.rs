//! Public network output presentation after one admitted numerical execution.
//! The immutable execution trace is recorded before any wrapper is built.

use runmat_value::{ObjectInstance, Value};

use super::execution::{
    host::network::{HostExecutionResult, HostValue},
    NetworkCaller, NetworkExecutionOutcome,
};
use super::host::dlarray_format;
use super::{object, state, DLARRAY_CLASS};
use crate::builtins::deep_learning::{autodiff, deep_learning_error};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OutputPresentation {
    Forward,
    PredictFormatted,
    PredictUnformattedArray,
    PredictRaw,
}

impl OutputPresentation {
    const fn caller(self) -> NetworkCaller {
        match self {
            Self::Forward => NetworkCaller::Forward,
            Self::PredictFormatted | Self::PredictUnformattedArray | Self::PredictRaw => {
                NetworkCaller::Predict
            }
        }
    }
}

pub(super) fn present(
    network: &ObjectInstance,
    input: &Value,
    outcome: NetworkExecutionOutcome,
    presentation: OutputPresentation,
    requested_state: bool,
) -> BuiltinResult<Vec<Value>> {
    let function = presentation.caller().function();
    let NetworkExecutionOutcome {
        data,
        output_axes,
        state_updates,
        trace,
    } = outcome;
    let (data, state_updates, nodes) = if trace.is_some() {
        let value = match data {
            Value::Tensor(tensor) => HostValue::Real(tensor),
            Value::ComplexTensor(tensor) => HostValue::Complex(tensor),
            _ => {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: host trace requires a host numerical output"),
                ))
            }
        };
        let mut execution = HostExecutionResult {
            value,
            axes: output_axes.clone(),
            trace,
            next_state: state_updates,
        };
        let nodes =
            autodiff::record_network_execution(network, input, &mut execution, requested_state)?;
        let data = match execution.value {
            HostValue::Real(tensor) => Value::Tensor(tensor),
            HostValue::Complex(tensor) => Value::ComplexTensor(tensor),
        };
        (data, execution.next_state, Some(nodes))
    } else {
        (data, state_updates, None)
    };
    let is_dlarray = matches!(input, Value::Object(object) if object.class_name.is(DLARRAY_CLASS));
    let has_prediction_node = nodes
        .as_ref()
        .is_some_and(|nodes| nodes.prediction.is_some());
    let wrap = match presentation {
        OutputPresentation::Forward => is_dlarray || has_prediction_node,
        OutputPresentation::PredictFormatted => true,
        OutputPresentation::PredictUnformattedArray => true,
        OutputPresentation::PredictRaw => has_prediction_node,
    };
    let format = match presentation {
        OutputPresentation::Forward if dlarray_format(input).is_empty() => String::new(),
        OutputPresentation::PredictUnformattedArray | OutputPresentation::PredictRaw => {
            String::new()
        }
        OutputPresentation::Forward | OutputPresentation::PredictFormatted => {
            output_axes.layout().labels()
        }
    };
    let prediction = if wrap {
        object(
            "dlarray",
            vec![
                ("Data", data),
                ("Format", Value::String(format.clone())),
                ("Labels", Value::String(format)),
            ],
        )
    } else {
        data
    };
    let state = if requested_state {
        Some(state::publish_updates(network, state_updates, function)?)
    } else {
        None
    };
    let (prediction, state) = match nodes {
        Some(nodes) => autodiff::annotate_network_outputs(network, prediction, state, nodes)?,
        None => (prediction, state),
    };
    let mut output = vec![prediction];
    if let Some(state) = state {
        output.push(state);
    }
    Ok(output)
}
