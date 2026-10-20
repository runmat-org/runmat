//! Tape annotation of canonical network parameters and recurrent state.
//!
//! Admission is a read-only pass over both owners. No leaf is created until
//! every existing payload has passed host tape admission.

use super::*;
use runmat_value::NumericDType;

struct AdmittedParameter {
    key: model::parameters::ParameterKey,
    value: Value,
    payload: NodeValue,
}

struct AdmittedState {
    key: model::state::StateKey,
    value: Value,
    payload: NodeValue,
}

pub(super) fn annotate_network_object(network: &mut ObjectInstance) -> BuiltinResult<()> {
    model::validate_sequential_connections(network, "dlfeval")?;
    let parameters = model::parameters::view(network, "dlfeval")?;
    let state = model::state::view(network, "dlfeval")?;

    let parameters = parameters
        .into_iter()
        .filter_map(|entry| entry.value.map(|value| (entry.key, value)))
        .map(|(key, value)| {
            let payload = admit_derivative_parent(value, key.role.name())?;
            Ok(AdmittedParameter {
                key,
                value: value.clone(),
                payload,
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let state = state
        .into_iter()
        .map(|entry| {
            let payload = admit_derivative_parent(entry.value, entry.key.role.name())?;
            Ok(AdmittedState {
                key: entry.key,
                value: entry.value.clone(),
                payload,
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    // Staging may fail, but cannot partially change the supplied network.
    // Host COW buffers and resident handles above retain their owning leases.
    let parameters = parameters
        .into_iter()
        .map(|entry| {
            let value = annotate_payload(entry.value, entry.payload)?;
            Ok((entry.key, value))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let state = state
        .into_iter()
        .map(|entry| {
            let value = annotate_payload(entry.value, entry.payload)?;
            Ok((entry.key, value))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    // Construct and validate the new State table before either owner commits.
    let state = model::state::stage_annotated(network, state, "dlfeval")?;
    model::parameters::install_annotated(network, parameters)?;
    model::state::install_annotated(network, state)?;
    Ok(())
}

/// Canonical model storage also admits exact integers for forward execution.
/// A tape leaf is narrower: its parent must support floating derivatives.
fn admit_derivative_parent(value: &Value, label: &str) -> BuiltinResult<NodeValue> {
    let payload = match model::admission::forward_data(value) {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle)
                != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(deep_learning_error(
                    "dlfeval",
                    format!("dlfeval: {label} requires real resident storage"),
                ));
            }
            NodeValue::Resident(handle.clone())
        }
        _ => NodeValue::Host(host_tensor_from_value(value, "dlfeval", label)?),
    };
    if !matches!(
        payload.numeric_dtype(),
        Some(NumericDType::F32 | NumericDType::F64)
    ) {
        return Err(deep_learning_error(
            "dlfeval",
            format!(
                "dlfeval: {label} automatic differentiation requires single or double parent storage; explicitly convert the value before differentiation"
            ),
        ));
    }
    Ok(payload)
}

fn annotate_payload(value: Value, payload: NodeValue) -> BuiltinResult<Value> {
    match value {
        Value::Object(mut object) if object.class_name().is(DLARRAY_CLASS) => {
            annotate_dlarray_object_with_value(&mut object, payload)?;
            Ok(Value::Object(object))
        }
        value => {
            let Value::Object(mut wrapper) = crate::builtins::deep_learning::object_with_identity(
                DLARRAY_CLASS,
                [
                    ("Data", value),
                    ("Format", Value::String("".into())),
                    ("Labels", Value::String("".into())),
                ],
            )?
            else {
                unreachable!("dlarray constructor")
            };
            annotate_dlarray_object_with_value(&mut wrapper, payload)?;
            Ok(Value::Object(wrapper))
        }
    }
}

#[cfg(test)]
mod tests;
