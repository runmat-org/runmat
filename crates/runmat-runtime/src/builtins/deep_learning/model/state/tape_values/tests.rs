use super::*;
use std::cell::Cell;
use std::collections::BTreeMap;

fn network() -> ObjectInstance {
    let mut layer = ObjectInstance::new(runmat_builtins::LSTM_LAYER_CLASS.owned());
    layer
        .replace_dl_property("Name", Value::String("memory".into()))
        .expect("fixture property");
    layer
        .replace_dl_property("NumHiddenUnits", Value::Num(3.0))
        .expect("fixture property");
    layer
        .replace_dl_property(
            "HiddenState",
            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
        )
        .expect("fixture property");
    layer
        .replace_dl_property(
            "CellState",
            Value::Tensor(Tensor::from_f32(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap()),
        )
        .expect("fixture property");
    let Value::Object(network) = super::super::super::network_object(
        DLNETWORK_CLASS,
        vec![Value::Object(layer)],
        BTreeMap::new(),
        "forward",
    )
    .unwrap() else {
        panic!("network object")
    };
    network
}

fn data(value: &Value) -> &Tensor {
    let Value::Object(wrapper) = value else {
        panic!("dlarray state")
    };
    let Some(Value::Tensor(data)) = wrapper.properties().get("Data") else {
        panic!("typed state Data")
    };
    data
}

#[test]
fn returned_state_mapping_uses_typed_canonical_keys_and_preserves_payloads() {
    let network = network();
    let state = project(&network, "get.State").unwrap();
    let Value::Object(mut state_object) = state else {
        panic!("State table")
    };
    state_object
        .replace_dl_property("TapeMetadata", Value::String("preserve".into()))
        .expect("fixture property");
    let original = Value::Object(state_object);
    let before = view(&network, "forward").unwrap();
    let mut roles = Vec::new();
    let mapped = map_result_values(&network, original.clone(), "forward", |key, value| {
        roles.push((key.layer_name.clone(), key.role, key.channels));
        Ok(value)
    })
    .unwrap();
    assert_eq!(roles.len(), 2);
    assert_eq!(roles[0], ("memory".into(), StateRole::HiddenState, 3));
    assert_eq!(roles[1], ("memory".into(), StateRole::CellState, 3));
    assert_eq!(mapped, original);
    let Value::Object(mapped_table) = &mapped else {
        panic!("mapped State table")
    };
    assert_eq!(
        mapped_table.properties().get("TapeMetadata"),
        Some(&Value::String("preserve".into()))
    );
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        panic!("Layers")
    };
    table::validate_canonical(layers.data(), &mapped, "forward").unwrap();
    let values = super::super::super::table_rows::values(mapped_table, "State", "forward").unwrap();
    for (before, after) in before.iter().zip(values.data()) {
        assert_eq!(
            data(before.value).numeric_dtype(),
            runmat_value::NumericDType::F32
        );
        assert_eq!(data(after).shape(), &[3, 1]);
        assert!(data(before.value)
            .host_buffer()
            .shares_allocation_with(data(after).host_buffer()));
    }
}

#[test]
fn wrong_identity_is_rejected_before_mapper_and_failure_does_not_mutate_network() {
    let network = network();
    let before = project(&network, "get.State").unwrap();
    let Value::Object(source) = &before else {
        panic!("State table")
    };
    let mut variables = crate::builtins::table::table_variables(source).unwrap();
    let Some(Value::StringArray(roles)) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Parameter")
    else {
        panic!("Parameter column")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(roles)[1] = "Bias".into();
    let invalid = crate::builtins::table::table_replace_variables_like(source, variables).unwrap();
    let calls = Cell::new(0);
    assert!(map_result_values(&network, invalid, "forward", |_, value| {
        calls.set(calls.get() + 1);
        Ok(value)
    })
    .is_err());
    assert_eq!(calls.get(), 0);
    assert_eq!(project(&network, "get.State").unwrap(), before);
}

#[test]
fn invalid_final_mapped_value_fails_without_mutating_network() {
    let network = network();
    let before = project(&network, "get.State").unwrap();
    let calls = Cell::new(0);
    assert!(
        map_result_values(&network, before.clone(), "forward", |_, value| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                Ok(Value::Tensor(
                    Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap(),
                ))
            } else {
                Ok(value)
            }
        })
        .is_err()
    );
    assert_eq!(calls.get(), 2);
    assert_eq!(project(&network, "get.State").unwrap(), before);
}
