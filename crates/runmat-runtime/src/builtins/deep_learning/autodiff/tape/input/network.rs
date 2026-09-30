//! Tape annotation of canonical network parameters and recurrent state.
//!
//! Admission is a read-only pass over both owners. No leaf is created until
//! every existing payload has passed host tape admission.

use super::*;

struct AdmittedParameter {
    key: model::parameters::ParameterKey,
    value: Value,
    tensor: Tensor,
}

struct AdmittedState {
    key: model::state::StateKey,
    value: Value,
    tensor: Tensor,
}

pub(super) fn annotate_network_object(network: &mut ObjectInstance) -> BuiltinResult<()> {
    model::validate_sequential_connections(network, "dlfeval")?;
    let parameters = model::parameters::view(network, "dlfeval")?;
    let state = model::state::view(network, "dlfeval")?;

    let parameters = parameters
        .into_iter()
        .filter_map(|entry| entry.value.map(|value| (entry.key, value)))
        .map(|(key, value)| {
            let tensor = host_tensor_from_value(value, "dlfeval", key.role.name())?;
            Ok(AdmittedParameter {
                key,
                value: value.clone(),
                tensor,
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let state = state
        .into_iter()
        .map(|entry| {
            let tensor = host_tensor_from_value(entry.value, "dlfeval", entry.key.role.name())?;
            Ok(AdmittedState {
                key: entry.key,
                value: entry.value.clone(),
                tensor,
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    // Staging may fail, but cannot partially change the supplied network.
    // Tensor cloning above retains its host allocation (copy-on-write).
    let parameters = parameters
        .into_iter()
        .map(|entry| {
            let value = annotate_payload(entry.value, entry.tensor)?;
            Ok((entry.key, value))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let state = state
        .into_iter()
        .map(|entry| {
            let value = annotate_payload(entry.value, entry.tensor)?;
            Ok((entry.key, value))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    // Construct and validate the new State table before either owner commits.
    let state = model::state::stage_annotated(network, state, "dlfeval")?;
    model::parameters::install_annotated(network, parameters);
    model::state::install_annotated(network, state);
    Ok(())
}

fn annotate_payload(value: Value, tensor: Tensor) -> BuiltinResult<Value> {
    match value {
        Value::Object(mut object) if object.class_name == DLARRAY_CLASS => {
            annotate_dlarray_object_with_tensor(&mut object, tensor)?;
            Ok(Value::Object(object))
        }
        value => {
            let mut wrapper = ObjectInstance::new(DLARRAY_CLASS.to_string());
            wrapper.properties.insert("Data".into(), value);
            wrapper
                .properties
                .insert("Format".into(), Value::String(String::new()));
            wrapper
                .properties
                .insert("Labels".into(), Value::String(String::new()));
            annotate_dlarray_object_with_tensor(&mut wrapper, tensor)?;
            Ok(Value::Object(wrapper))
        }
    }
}

#[cfg(test)]
mod tests;
