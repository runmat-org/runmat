//! Borrowed canonical parameter and state geometry; no materialization.

use runmat_value::ObjectInstance;

use crate::builtins::deep_learning::deep_learning_error;
use crate::builtins::deep_learning::model::{
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

pub(super) fn parameter_shape<'a>(
    rows: &[ParameterView<'a>],
    index: usize,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<&'a [usize]> {
    let row = rows
        .iter()
        .find(|row| row.key.layer_index == index && row.key.role == role)
        .ok_or_else(|| {
            fail(
                function,
                format!("layer {} is missing {}", index + 1, role.name()),
            )
        })?;
    let value = row.value.ok_or_else(|| {
        fail(
            function,
            format!("layer {} has uninitialized {}", index + 1, role.name()),
        )
    })?;
    let shape = parameters::numeric_payload_shape(value, role.name(), function)?;
    if shape.contains(&0) {
        return Err(fail(
            function,
            format!("layer {} has uninitialized {}", index + 1, role.name()),
        ));
    }
    Ok(shape)
}

pub(super) fn state_shape<'a>(
    rows: &[StateView<'a>],
    index: usize,
    role: StateRole,
    function: &'static str,
) -> BuiltinResult<Option<&'a [usize]>> {
    rows.iter()
        .find(|row| row.key.layer_index == index && row.key.role == role)
        .map(|row| {
            parameters::numeric_payload_shape(row.value, role.name(), function)
                .map(|shape| (!shape.contains(&0)).then_some(shape))
        })
        .transpose()
        .map(Option::flatten)
}
