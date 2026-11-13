//! Tape bindings for an already executed network result and its immutable trace.
use super::super::*;
use model::execution::host::network::{HostExecutionResult, HostTrace, HostValue};
use model::execution::NetworkCaller;
use model::parameters::ParameterRole;
use model::state::{StateKey, StateRole};
use std::collections::HashMap;

mod reverse;
#[cfg(test)]
mod tests;
pub(in crate::builtins::deep_learning::autodiff) use reverse::reverse;

pub(in crate::builtins::deep_learning) struct NetworkOutputNodes {
    pub prediction: Option<usize>,
    pub state: Vec<(StateKey, usize)>,
}

pub(super) struct ParentBinding {
    node: usize,
    shape: Vec<usize>,
    dtype: NumericDType,
}

pub(in crate::builtins::deep_learning::autodiff) struct NetworkTapeRecord {
    trace: HostTrace,
    input: Option<ParentBinding>,
    parameters: HashMap<(usize, ParameterRole), ParentBinding>,
    states: HashMap<(usize, StateRole), ParentBinding>,
    output_axes: model::plan::axes::AxisPlan,
    output_dtype: NumericDType,
    recorded_outputs: std::ops::Range<usize>,
    #[cfg(test)]
    reverse_calls: std::cell::Cell<usize>,
}

#[derive(Clone, Copy)]
pub(in crate::builtins::deep_learning::autodiff) enum NetworkOutput {
    Prediction,
    State { layer_index: usize, role: StateRole },
}

fn parent(value: &Value) -> BuiltinResult<Option<ParentBinding>> {
    let Some(node) = dlarray_node_id(value) else {
        return Ok(None);
    };
    let tensor = value_for_node(node)?;
    Ok(Some(ParentBinding {
        node,
        shape: tensor.shape().to_vec(),
        dtype: tensor.numeric_dtype(),
    }))
}

/// Record the numerical trace once; no layer operation is evaluated again.
pub(in crate::builtins::deep_learning) fn record_execution(
    network: &ObjectInstance,
    input: &Value,
    execution: &mut HostExecutionResult,
    capture_state: bool,
    caller: NetworkCaller,
) -> BuiltinResult<NetworkOutputNodes> {
    let function = caller.function();
    let mut outputs = NetworkOutputNodes {
        prediction: None,
        state: Vec::new(),
    };
    if !tape_is_active() {
        return Ok(outputs);
    }
    let mut parameters = HashMap::new();
    for row in model::parameters::view(network, function)? {
        if let Some(value) = row.value {
            if let Some(binding) = parent(value)? {
                parameters.insert((row.key.layer_index, row.key.role), binding);
            }
        }
    }
    let rows = model::state::view(network, function)?;
    let mut states = HashMap::new();
    let mut keys = HashMap::new();
    for row in rows {
        let identity = (row.key.layer_index, row.key.role);
        if let Some(binding) = parent(row.value)? {
            states.insert(identity, binding);
        }
        keys.insert(identity, row.key);
    }
    let input = parent(input)?;
    if input.is_none() && parameters.is_empty() && states.is_empty() {
        return Ok(outputs);
    }
    let HostValue::Real(output) = &execution.value else {
        return Err(unsupported_error(
            function,
            format!("{function}: complex network tape values require complex adjoint support"),
        ));
    };
    for update in execution.next_state.iter().filter(|_| capture_state) {
        if !keys.contains_key(&(update.layer_index, update.role)) {
            return Err(deep_learning_error(
                function,
                format!("{function}: traced State result has no canonical row"),
            ));
        }
    }
    let trace = execution.trace.take().ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: active network tape requires an execution trace"),
        )
    })?;
    let first_node = active_tape()
        .ok_or_else(|| deep_learning_error(function, format!("{function}: no active tape")))?
        .borrow()
        .nodes
        .len();
    let output_count = if capture_state {
        execution.next_state.len()
    } else {
        0
    };
    let end = first_node
        .checked_add(output_count)
        .and_then(|end| end.checked_add(1))
        .ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: tape output count exceeds addressable size"),
            )
        })?;
    let record = Rc::new(NetworkTapeRecord {
        trace,
        input,
        parameters,
        states,
        output_axes: execution.axes.clone(),
        output_dtype: output.numeric_dtype(),
        recorded_outputs: first_node..end,
        #[cfg(test)]
        reverse_calls: std::cell::Cell::new(0),
    });
    outputs.prediction = Some(push_node(
        output.clone(),
        NodeKind::Network {
            execution: record.clone(),
            output: NetworkOutput::Prediction,
        },
    )?);
    for update in execution.next_state.iter().filter(|_| capture_state) {
        let key = keys
            .remove(&(update.layer_index, update.role))
            .expect("admitted State output");
        let node = push_node(
            update.value.clone(),
            NodeKind::Network {
                execution: record.clone(),
                output: NetworkOutput::State {
                    layer_index: update.layer_index,
                    role: update.role,
                },
            },
        )?;
        outputs.state.push((key, node));
    }
    Ok(outputs)
}

pub(in crate::builtins::deep_learning) fn annotate_outputs(
    network: &ObjectInstance,
    prediction: Value,
    state: Option<Value>,
    nodes: NetworkOutputNodes,
    caller: NetworkCaller,
) -> BuiltinResult<(Value, Option<Value>)> {
    let function = caller.function();
    let prediction = match nodes.prediction {
        Some(node) => attach(prediction, node, caller)?,
        None => prediction,
    };
    if nodes.state.is_empty() {
        return Ok((prediction, state));
    }
    let state = state.ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: traced State nodes require a published State result"),
        )
    })?;
    let mut nodes: HashMap<_, _> = nodes.state.into_iter().collect();
    let state =
        model::state::map_result_values(network, state, function, |key, value| {
            match nodes.remove(key) {
                Some(node) => attach(value, node, caller),
                None => Ok(value),
            }
        })?;
    if !nodes.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: traced State nodes differ from the published canonical table"),
        ));
    }
    Ok((prediction, Some(state)))
}

fn attach(mut value: Value, node: usize, caller: NetworkCaller) -> BuiltinResult<Value> {
    let function = caller.function();
    if let Value::Object(object) = &mut value {
        if object.class_name().is(DLARRAY_CLASS) {
            if let Some(Value::GpuTensor(handle)) = object.properties().get("Data") {
                let super::super::tape::NodeValue::Resident(recorded) =
                    super::super::tape::node_value(node)?
                else {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: resident result has no resident tape node"),
                    ));
                };
                if handle.view_identity() != recorded.view_identity() {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: resident result differs from its tape node"),
                    ));
                }
                attach_node_to_object(object, node)?;
                return Ok(value);
            }
        }
    }
    let payload = Payload::parse(value, function)?;
    wrap_result(&payload, payload.tensor.clone(), Some(node))
}
