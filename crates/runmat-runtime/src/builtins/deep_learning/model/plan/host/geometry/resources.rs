//! Borrowed canonical parameter and state geometry; no materialization.

use std::collections::HashMap;

use runmat_value::{ObjectInstance, Value};

use crate::builtins::deep_learning::deep_learning_error;
use crate::builtins::deep_learning::model::{
    initialization,
    parameters::{self, ParameterRole, ParameterView},
    state::{self, StateRole, StateView},
};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) fn views<'a>(
    network: &'a ObjectInstance,
    function: &'static str,
) -> BuiltinResult<(Vec<ParameterView<'a>>, Vec<StateView<'a>>)> {
    let parameters = parameters::view(network, function)?;
    let state = state::view(network, function)?;
    for row in &parameters {
        let value = row
            .value
            .ok_or_else(|| fail(function, "network has uninitialized parameters"))?;
        if parameters::numeric_payload_shape(value, row.key.role.name(), function)?.contains(&0) {
            return Err(fail(function, "network has uninitialized parameters"));
        }
    }
    for row in &state {
        if parameters::numeric_payload_shape(row.value, row.key.role.name(), function)?.contains(&0)
        {
            return Err(fail(function, "network has uninitialized state"));
        }
    }
    Ok((parameters, state))
}

/// Small, owned geometry only. The payload authority remains the canonical
/// Layers and State stores; this is shared by the initialized execution path
/// and training's pre-initialization admission.
pub(super) struct ResourceShapes {
    parameters: HashMap<(usize, ParameterRole), Vec<usize>>,
    state: HashMap<(usize, StateRole), Vec<usize>>,
}

impl ResourceShapes {
    pub(super) fn filled(
        parameters: &[ParameterView<'_>],
        state: &[StateView<'_>],
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let parameters = parameters
            .iter()
            .map(|row| {
                let value = row
                    .value
                    .ok_or_else(|| fail(function, "network has uninitialized parameters"))?;
                let shape =
                    parameters::numeric_payload_shape(value, row.key.role.name(), function)?;
                if shape.contains(&0) {
                    return Err(fail(function, "network has uninitialized parameters"));
                }
                Ok(((row.key.layer_index, row.key.role), shape.to_vec()))
            })
            .collect::<BuiltinResult<HashMap<_, _>>>()?;
        let state = state
            .iter()
            .map(|row| {
                let shape =
                    parameters::numeric_payload_shape(row.value, row.key.role.name(), function)?;
                if shape.contains(&0) {
                    return Err(fail(function, "network has uninitialized state"));
                }
                Ok(((row.key.layer_index, row.key.role), shape.to_vec()))
            })
            .collect::<BuiltinResult<HashMap<_, _>>>()?;
        Ok(Self { parameters, state })
    }

    pub(super) fn before_initialization(
        layers: &[Value],
        canonical_network: Option<&ObjectInstance>,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        let rows = parameters::view_layers(layers, function)?;
        let planned = initialization::planned_parameter_shapes(layers, function)?
            .into_iter()
            .map(|(index, role, shape)| ((index, role), shape))
            .collect::<HashMap<_, _>>();
        let parameters = rows
            .iter()
            .map(|row| {
                let existing = row
                    .value
                    .map(|value| {
                        parameters::numeric_payload_shape(value, row.key.role.name(), function)
                    })
                    .transpose()?;
                let shape = match existing {
                    Some(shape) if !shape.contains(&0) => shape.to_vec(),
                    _ => planned
                        .get(&(row.key.layer_index, row.key.role))
                        .cloned()
                        .ok_or_else(|| fail(function, "initializer geometry is incomplete"))?,
                };
                Ok(((row.key.layer_index, row.key.role), shape))
            })
            .collect::<BuiltinResult<HashMap<_, _>>>()?;
        let planned_state = state::planned_initial_shapes(layers, function)?;
        let state = if let Some(network) = canonical_network {
            // A supplied network owns its State in `_State`, not in mutable
            // layer initial-condition properties. Validate and borrow those
            // rows before any initializer or provider work.
            let rows = state::view(network, function)?;
            rows.into_iter()
                .map(|row| {
                    let shape = parameters::numeric_payload_shape(
                        row.value,
                        row.key.role.name(),
                        function,
                    )?;
                    let shape = if shape.contains(&0) {
                        planned_state
                            .iter()
                            .find(|(index, role, _)| {
                                *index == row.key.layer_index && *role == row.key.role
                            })
                            .map(|(_, _, shape)| shape.clone())
                            .ok_or_else(|| fail(function, "missing planned State geometry"))?
                    } else {
                        shape.to_vec()
                    };
                    Ok(((row.key.layer_index, row.key.role), shape))
                })
                .collect::<BuiltinResult<HashMap<_, _>>>()?
        } else {
            planned_state
                .into_iter()
                .map(|(index, role, shape)| ((index, role), shape))
                .collect()
        };
        Ok(Self { parameters, state })
    }

    pub(super) fn parameter_shape(
        &self,
        index: usize,
        role: ParameterRole,
        function: &'static str,
    ) -> BuiltinResult<&[usize]> {
        self.parameters
            .get(&(index, role))
            .map(Vec::as_slice)
            .ok_or_else(|| {
                fail(
                    function,
                    format!("layer {} is missing {} geometry", index + 1, role.name()),
                )
            })
    }

    pub(super) fn state_shape(&self, index: usize, role: StateRole) -> Option<&[usize]> {
        self.state.get(&(index, role)).map(Vec::as_slice)
    }
}
