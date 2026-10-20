use super::*;
use crate::builtins::deep_learning::layers::feature_input_layer_builtin;
use crate::builtins::deep_learning::model::dlnetwork_builtin;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use futures::executor::block_on;

fn network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(2.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::from_f32(vec![1., 2., 3., 4.], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::from_f32(vec![0., 0.], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

fn network_object(owner: &crate::execution::RootedValue) -> ObjectInstance {
    let Value::Object(network) = owner.value() else {
        panic!("network expected")
    };
    network.clone()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn empty_native_update_rejects_before_parameter_publication() {
    let network_owner = network();
    let input = network_object(&network_owner);
    let key = view(&input, "test").unwrap()[0].key.clone();
    let error = apply_updates(
        input.clone(),
        vec![ParameterUpdate {
            key,
            value: Tensor::from_f32(vec![], vec![0, 2]).unwrap(),
        }],
        "test",
    )
    .unwrap_err();
    assert!(error
        .message()
        .contains("native training update must not be empty"));
    let rows = view(&input, "test").unwrap();
    let Value::Tensor(weights) = parameter_data(rows[0].value.unwrap(), "test").unwrap() else {
        panic!("unchanged native learnable storage")
    };
    assert_eq!(weights.shape(), &[2, 2]);
    assert_eq!(weights.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
}

#[test]
fn selected_native_updates_preserve_canonical_store_metadata_and_input_alias() {
    let network_owner = network();
    let mut input = network_object(&network_owner);
    input
        .replace_dl_property("UserMarker", Value::String("keep".into()))
        .expect("fixture property");
    let original = input.clone();
    let rows = view(&input, "test").unwrap();
    let weight_key = rows[0].key.clone();
    let new_weights = Tensor::from_f32(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();
    let Value::Object(updated) = apply_updates(
        input,
        vec![ParameterUpdate {
            key: weight_key,
            value: new_weights,
        }],
        "test",
    )
    .unwrap() else {
        panic!("updated network expected")
    };
    assert_eq!(
        updated.properties().get("UserMarker"),
        Some(&Value::String("keep".into()))
    );
    assert_eq!(
        updated
            .properties()
            .get(super::super::state::STORAGE_PROPERTY),
        original
            .properties()
            .get(super::super::state::STORAGE_PROPERTY)
    );
    let updated_rows = view(&updated, "test").unwrap();
    let old_rows = view(&original, "test").unwrap();
    let updated_data = parameter_data(updated_rows[0].value.unwrap(), "test").unwrap();
    let old_data = parameter_data(old_rows[0].value.unwrap(), "test").unwrap();
    let Value::Tensor(updated_data) = updated_data else {
        panic!("updated tensor expected")
    };
    let Value::Tensor(old_data) = old_data else {
        panic!("original tensor expected")
    };
    assert_eq!(updated_data.as_f32_slice(), Some(&[5., 6., 7., 8.][..]));
    assert_eq!(old_data.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
    assert_eq!(updated_rows[1].value, old_rows[1].value);
}

#[test]
fn selected_updates_reject_entire_malformed_batch_before_install() {
    let network_owner = network();
    let input = network_object(&network_owner);
    let rows = view(&input, "test").unwrap();
    let updates = vec![
        ParameterUpdate {
            key: rows[0].key.clone(),
            value: Tensor::from_f32(vec![5., 6., 7., 8.], vec![2, 2]).unwrap(),
        },
        ParameterUpdate {
            key: rows[1].key.clone(),
            value: Tensor::from_f32(vec![1., 2., 3.], vec![1, 3]).unwrap(),
        },
    ];
    let alias = input.clone();
    assert!(apply_updates(input, updates, "test").is_err());
    let old_rows = view(&alias, "test").unwrap();
    let Value::Tensor(weights) = parameter_data(old_rows[0].value.unwrap(), "test").unwrap() else {
        panic!("original weights expected")
    };
    assert_eq!(weights.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
}

#[test]
fn selected_updates_reject_duplicate_foreign_and_precision_changing_keys() {
    let network_owner = network();
    let input = network_object(&network_owner);
    let key = view(&input, "test").unwrap()[0].key.clone();
    let value = Tensor::from_f32(vec![5.; 4], vec![2, 2]).unwrap();
    assert!(apply_updates(
        input.clone(),
        vec![
            ParameterUpdate {
                key: key.clone(),
                value: value.clone(),
            },
            ParameterUpdate {
                key: key.clone(),
                value: value.clone(),
            },
        ],
        "test",
    )
    .is_err());
    let mut foreign = key.clone();
    foreign.layer_name.push_str("-other");
    assert!(apply_updates(
        input.clone(),
        vec![ParameterUpdate {
            key: foreign,
            value: value.clone(),
        }],
        "test",
    )
    .is_err());
    assert!(apply_updates(
        input,
        vec![ParameterUpdate {
            key,
            value: Tensor::new(vec![5.; 4], vec![2, 2]).unwrap(),
        }],
        "test",
    )
    .is_err());
}

#[test]
fn selected_updates_preserve_dlarray_format_without_reusing_wrapper_identity() {
    let network_owner = network();
    let mut input = network_object(&network_owner);
    let Value::Cell(layers) = input.fixture_dl_property_mut("Layers").unwrap() else {
        panic!("Layers cell expected")
    };
    let Value::Object(dense) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
    else {
        panic!("dense layer expected")
    };
    let original = dense
        .remove_dl_property("Weights")
        .expect("fixture removal")
        .unwrap();
    let mut wrapper = ObjectInstance::new(crate::builtins::deep_learning::DLARRAY_CLASS.owned());
    wrapper
        .replace_dl_property("Data", original)
        .expect("fixture property");
    wrapper
        .replace_dl_property("Format", Value::String(String::new().into()))
        .expect("fixture property");
    wrapper
        .replace_dl_property("Labels", Value::String(String::new().into()))
        .expect("fixture property");
    dense
        .replace_dl_property("Weights", Value::Object(wrapper))
        .expect("fixture property");

    let key = view(&input, "test").unwrap()[0].key.clone();
    let Value::Object(updated) = apply_updates(
        input.clone(),
        vec![ParameterUpdate {
            key,
            value: Tensor::from_f32(vec![9.; 4], vec![2, 2]).unwrap(),
        }],
        "test",
    )
    .unwrap() else {
        panic!("updated network expected")
    };
    let rows = view(&updated, "test").unwrap();
    let Value::Object(replacement) = rows[0].value.unwrap() else {
        panic!("dlarray wrapper expected")
    };
    assert_eq!(
        replacement.properties().get("Format"),
        Some(&Value::String(String::new().into()))
    );
    assert_eq!(
        replacement.properties().get("Labels"),
        Some(&Value::String(String::new().into()))
    );
    assert_eq!(replacement.properties().len(), 3);
    let original = view(&input, "test").unwrap();
    let Value::Tensor(original) = parameter_data(original[0].value.unwrap(), "test").unwrap()
    else {
        panic!("original tensor expected")
    };
    assert_eq!(original.as_f32_slice(), Some(&[1., 2., 3., 4.][..]));
}
