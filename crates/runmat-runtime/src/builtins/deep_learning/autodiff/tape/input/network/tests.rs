use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use crate::builtins::deep_learning::model::{parameters, state};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::builtins::deep_learning::autodiff::tape::{Tape, TapeActivation};
use futures::executor::block_on;
use runmat_builtins::DeepLearningLayerKind;
use runmat_value::{IntegerStorage, NumericDType};

fn tensor(shape: &[usize], single: bool) -> Value {
    let count = shape.iter().product();
    if single {
        Value::Tensor(Tensor::from_f32(vec![1.0; count], shape.to_vec()).unwrap())
    } else {
        Value::Tensor(Tensor::new(vec![1.0; count], shape.to_vec()).unwrap())
    }
}

fn integer_storages(count: usize) -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![1; count]),
        IntegerStorage::I16(vec![1; count]),
        IntegerStorage::I32(vec![1; count]),
        IntegerStorage::I64(vec![1; count]),
        IntegerStorage::U8(vec![1; count]),
        IntegerStorage::U16(vec![1; count]),
        IntegerStorage::U32(vec![1; count]),
        IntegerStorage::U64(vec![1; count]),
    ]
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
}

fn layer(kind: DeepLearningLayerKind, name: &str, fields: &[(&str, Value)]) -> Value {
    let mut object = ObjectInstance::new(kind.class().owned());
    object
        .replace_dl_property("Name", Value::String(name.into()))
        .expect("fixture property");
    for (key, value) in fields {
        object
            .replace_dl_property(key, value.clone())
            .expect("fixture property");
    }
    Value::Object(object)
}

fn network(layers: Vec<Value>) -> ObjectInstance {
    network_with_initialize(layers, true)
}

fn network_with_initialize(layers: Vec<Value>, initialize: bool) -> ObjectInstance {
    let options = BTreeMap::from([("Initialize".into(), Value::Bool(initialize))]);
    let Value::Object(object) =
        model::network_object(model::DLNETWORK_CLASS, layers, options, "dlfeval").unwrap()
    else {
        panic!("network object expected")
    };
    object
}

fn data(value: &Value) -> &Tensor {
    let Value::Object(wrapper) = value else {
        panic!("dlarray wrapper expected")
    };
    let Some(Value::Tensor(tensor)) = wrapper.properties().get("Data") else {
        panic!("tensor data expected")
    };
    tensor
}

#[test]
fn canonical_learnables_and_recurrent_state_are_annotated_without_changing_metadata() {
    let cases = [
        network(vec![
            layer(
                DeepLearningLayerKind::FeatureInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::FullyConnected,
                "dense",
                &[
                    ("OutputSize", Value::Num(3.0)),
                    ("Weights", tensor(&[3, 2], true)),
                    ("Bias", tensor(&[3, 1], true)),
                ],
            ),
        ]),
        network(vec![
            layer(
                DeepLearningLayerKind::SequenceInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::Convolution1D,
                "conv",
                &[
                    ("FilterSize", Value::Num(3.0)),
                    ("NumFilters", Value::Num(4.0)),
                    ("Weights", tensor(&[3, 2, 4], true)),
                    ("Bias", tensor(&[1, 4], true)),
                ],
            ),
        ]),
        network(vec![
            layer(
                DeepLearningLayerKind::SequenceInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::Lstm,
                "lstm",
                &[
                    ("NumHiddenUnits", Value::Num(3.0)),
                    ("InputWeights", tensor(&[12, 2], true)),
                    ("RecurrentWeights", tensor(&[12, 3], true)),
                    ("Bias", tensor(&[12, 1], true)),
                    ("HiddenState", tensor(&[3, 1], true)),
                    ("CellState", tensor(&[3, 1], true)),
                ],
            ),
        ]),
        network(vec![
            layer(
                DeepLearningLayerKind::SequenceInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::Bilstm,
                "bilstm",
                &[
                    ("NumHiddenUnits", Value::Num(3.0)),
                    ("InputWeights", tensor(&[24, 2], true)),
                    ("RecurrentWeights", tensor(&[24, 3], true)),
                    ("Bias", tensor(&[24, 1], true)),
                    ("HiddenState", tensor(&[6, 1], true)),
                    ("CellState", tensor(&[6, 1], true)),
                ],
            ),
        ]),
        network(vec![
            layer(
                DeepLearningLayerKind::SequenceInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::LayerNormalization,
                "norm",
                &[
                    ("Scale", tensor(&[2, 1], true)),
                    ("Offset", tensor(&[2, 1], true)),
                ],
            ),
        ]),
    ];
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let mut expected_leaves = 0;
    for mut network in cases {
        let initialized = network.properties().get("Initialized").cloned();
        let before_parameters = parameters::view(&network, "dlfeval").unwrap();
        let before_state = state::view(&network, "dlfeval").unwrap();
        let originals = before_parameters
            .iter()
            .filter_map(|row| row.value.map(|value| (row.key.clone(), value.clone())))
            .collect::<Vec<_>>();
        let state_originals = before_state
            .iter()
            .map(|row| (row.key.clone(), row.value.clone()))
            .collect::<Vec<_>>();
        expected_leaves += originals.len() + state_originals.len();
        annotate_network_object(&mut network).unwrap();
        assert_eq!(
            network.properties().get("Initialized"),
            initialized.as_ref()
        );
        for (row, (key, original)) in parameters::view(&network, "dlfeval")
            .unwrap()
            .into_iter()
            .filter(|row| row.value.is_some())
            .zip(originals)
        {
            assert_eq!(row.key, key);
            let annotated = row.value.unwrap();
            let Value::Tensor(before) = original else {
                panic!("original tensor")
            };
            let after = data(annotated);
            assert_eq!(after.numeric_dtype(), NumericDType::F32);
            assert_eq!(after.shape(), before.shape());
            assert!(after
                .host_buffer()
                .shares_allocation_with(before.host_buffer()));
        }
        for (row, (key, original)) in state::view(&network, "dlfeval")
            .unwrap()
            .into_iter()
            .zip(state_originals)
        {
            assert_eq!(row.key, key);
            let before = data(&original);
            let after = data(row.value);
            assert_eq!(after.numeric_dtype(), NumericDType::F32);
            assert_eq!(after.shape(), before.shape());
            assert!(after
                .host_buffer()
                .shares_allocation_with(before.host_buffer()));
        }
        let before_repeat = tape.borrow().nodes.len();
        annotate_network_object(&mut network).unwrap();
        assert_eq!(tape.borrow().nodes.len(), before_repeat);
    }
    assert_eq!(tape.borrow().nodes.len(), expected_leaves);
}

#[test]
fn invalid_last_parameter_is_rejected_before_creating_any_leaf() {
    let mut network = network(vec![
        layer(
            DeepLearningLayerKind::SequenceInput,
            "input",
            &[("InputSize", Value::Num(2.0))],
        ),
        layer(
            DeepLearningLayerKind::LayerNormalization,
            "norm",
            &[
                ("Scale", tensor(&[2, 1], true)),
                ("Offset", tensor(&[2, 1], true)),
            ],
        ),
    ]);
    let original_initialized = network.properties().get("Initialized").cloned();
    let Some(Value::Cell(layers)) = network.fixture_dl_property_mut("Layers") else {
        panic!("Layers")
    };
    let Value::Object(last) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
    else {
        panic!("normalization layer")
    };
    last.replace_dl_property("Offset", Value::String("invalid".into()))
        .expect("fixture property");

    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    assert!(annotate_network_object(&mut network).is_err());
    assert!(tape.borrow().nodes.is_empty());
    assert_eq!(
        network.properties().get("Initialized"),
        original_initialized.as_ref()
    );
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        panic!("Layers")
    };
    let Value::Object(last) = &layers.data()[1] else {
        panic!("normalization layer")
    };
    assert_eq!(
        last.properties().get("Offset"),
        Some(&Value::String("invalid".into()))
    );
}

#[test]
fn public_learnables_setter_cannot_reuse_a_tag_after_data_replacement() {
    let network = network(vec![
        layer(
            DeepLearningLayerKind::FeatureInput,
            "input",
            &[("InputSize", Value::Num(2.0))],
        ),
        layer(
            DeepLearningLayerKind::FullyConnected,
            "dense",
            &[
                ("OutputSize", Value::Num(2.0)),
                ("Weights", tensor(&[2, 2], true)),
                ("Bias", tensor(&[2, 1], true)),
            ],
        ),
    ]);
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let annotated = annotate_tree(Value::Object(network)).unwrap();
    assert_eq!(tape.borrow().nodes.len(), 2);
    let table = block_on(model::get_learnables_builtin(annotated.clone())).unwrap();
    let Value::Object(table_object) = &table else {
        panic!("Learnables table expected")
    };
    let mut variables = crate::builtins::table::table_variables(table_object).unwrap();
    let Some(Value::Cell(values)) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value")
    else {
        panic!("Value column expected")
    };
    let Value::Object(weights) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(values)[0]
    else {
        panic!("traced Weights dlarray expected")
    };
    let old_node = dlarray_object_node_id(weights).expect("original Weights tag");
    weights
        .replace_dl_property(
            "Data",
            Value::Tensor(Tensor::from_f32(vec![2.0; 4], vec![2, 2]).unwrap()),
        )
        .expect("fixture property");
    assert!(weights.properties().contains_key(AD_NODE_PROPERTY));
    assert_eq!(dlarray_object_node_id(weights), None);
    let replacement =
        crate::builtins::table::table_replace_variables_like(table_object, variables).unwrap();
    let Value::Object(mut updated) =
        block_on(model::set_learnables_builtin(annotated, replacement)).unwrap()
    else {
        panic!("updated network expected")
    };
    let rows = parameters::view(&updated, "dlfeval").unwrap();
    assert_eq!(dlarray_node_id(rows[0].value.unwrap()), None);
    assert_eq!(dlarray_node_id(rows[1].value.unwrap()), Some(1));
    assert_eq!(
        tape.borrow().nodes[old_node]
            .value
            .host()
            .unwrap()
            .as_f32_slice()
            .unwrap(),
        &[1.0; 4]
    );
    annotate_network_object(&mut updated).unwrap();
    assert_eq!(tape.borrow().nodes.len(), 3);
    let rows = parameters::view(&updated, "dlfeval").unwrap();
    assert_eq!(dlarray_node_id(rows[0].value.unwrap()), Some(2));
    assert_eq!(dlarray_node_id(rows[1].value.unwrap()), Some(1));
}

#[test]
fn public_state_setter_detaches_edited_cell_state_without_disturbing_hidden_state() {
    let network = network(vec![
        layer(
            DeepLearningLayerKind::SequenceInput,
            "input",
            &[("InputSize", Value::Num(2.0))],
        ),
        layer(
            DeepLearningLayerKind::Lstm,
            "lstm",
            &[
                ("NumHiddenUnits", Value::Num(3.0)),
                ("InputWeights", tensor(&[12, 2], true)),
                ("RecurrentWeights", tensor(&[12, 3], true)),
                ("Bias", tensor(&[12, 1], true)),
                ("HiddenState", tensor(&[3, 1], true)),
                ("CellState", tensor(&[3, 1], true)),
            ],
        ),
    ]);
    let tape = Rc::new(RefCell::new(Tape::new()));
    let _activation = TapeActivation::enter(tape.clone());
    let annotated = annotate_tree(Value::Object(network)).unwrap();
    assert_eq!(tape.borrow().nodes.len(), 5);
    let table = block_on(model::get_state_builtin(annotated.clone())).unwrap();
    let Value::Object(table_object) = &table else {
        panic!("State table expected")
    };
    let mut variables = crate::builtins::table::table_variables(table_object).unwrap();
    let Some(Value::Cell(values)) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value")
    else {
        panic!("Value column expected")
    };
    let Value::Object(cell) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(values)[1]
    else {
        panic!("traced CellState dlarray expected")
    };
    cell.replace_dl_property(
        "Data",
        Value::Tensor(Tensor::from_f32(vec![2.0; 3], vec![3, 1]).unwrap()),
    )
    .expect("fixture property");
    let replacement =
        crate::builtins::table::table_replace_variables_like(table_object, variables).unwrap();
    let Value::Object(mut updated) =
        block_on(model::set_state_builtin(annotated, replacement)).unwrap()
    else {
        panic!("updated network expected")
    };
    let rows = state::view(&updated, "dlfeval").unwrap();
    assert_eq!(dlarray_node_id(rows[0].value), Some(3));
    assert_eq!(dlarray_node_id(rows[1].value), None);
    annotate_network_object(&mut updated).unwrap();
    assert_eq!(tape.borrow().nodes.len(), 6);
    let rows = state::view(&updated, "dlfeval").unwrap();
    assert_eq!(dlarray_node_id(rows[0].value), Some(3));
    assert_eq!(dlarray_node_id(rows[1].value), Some(5));
}

#[test]
fn all_integer_classes_including_empty_storage_are_rejected_as_derivative_parents() {
    for (count, shape) in [(1, vec![1, 1]), (0, vec![0, 0])] {
        for storage in integer_storages(count) {
            let dtype = storage.numeric_dtype();
            let value = integer_tensor(storage, shape.clone());
            let tape = Rc::new(RefCell::new(Tape::new()));
            let _activation = TapeActivation::enter(tape.clone());
            let error = admit_derivative_parent(&value, "Weights").unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("single or double parent storage"),
                "{dtype:?}: {error}"
            );
            assert!(tape.borrow().nodes.is_empty(), "{dtype:?}");
        }
    }
    assert_eq!(
        admit_derivative_parent(&tensor(&[1, 1], true), "Weights")
            .unwrap()
            .numeric_dtype(),
        Some(NumericDType::F32)
    );
    assert_eq!(
        admit_derivative_parent(&tensor(&[1, 1], false), "Weights")
            .unwrap()
            .numeric_dtype(),
        Some(NumericDType::F64)
    );
}

#[test]
fn every_integer_last_parameter_is_rejected_before_any_earlier_leaf() {
    for (count, shape) in [(2, vec![2, 1]), (0, vec![0, 0])] {
        for storage in integer_storages(count) {
            let dtype = storage.numeric_dtype();
            let mut network = network_with_initialize(
                vec![
                    layer(
                        DeepLearningLayerKind::SequenceInput,
                        "input",
                        &[("InputSize", Value::Num(2.0))],
                    ),
                    layer(
                        DeepLearningLayerKind::LayerNormalization,
                        "norm",
                        &[
                            ("Scale", tensor(&[2, 1], true)),
                            ("Offset", integer_tensor(storage, shape.clone())),
                        ],
                    ),
                ],
                count != 0,
            );
            let before = network.clone();
            let tape = Rc::new(RefCell::new(Tape::new()));
            let _activation = TapeActivation::enter(tape.clone());
            let error = annotate_network_object(&mut network).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("single or double parent storage"),
                "{dtype:?}: {error}"
            );
            assert!(tape.borrow().nodes.is_empty(), "{dtype:?}");
            assert_eq!(network.properties(), before.properties(), "{dtype:?}");
        }
    }
}

#[test]
fn every_integer_last_state_row_is_rejected_before_any_parameter_or_state_leaf() {
    for (count, shape) in [(3, vec![3, 1]), (0, vec![0, 0])] {
        for storage in integer_storages(count) {
            let dtype = storage.numeric_dtype();
            let mut network = network_with_initialize(
                vec![
                    layer(
                        DeepLearningLayerKind::SequenceInput,
                        "input",
                        &[("InputSize", Value::Num(2.0))],
                    ),
                    layer(
                        DeepLearningLayerKind::Lstm,
                        "lstm",
                        &[
                            ("NumHiddenUnits", Value::Num(3.0)),
                            ("InputWeights", tensor(&[12, 2], true)),
                            ("RecurrentWeights", tensor(&[12, 3], true)),
                            ("Bias", tensor(&[12, 1], true)),
                            ("HiddenState", tensor(&[3, 1], true)),
                            ("CellState", integer_tensor(storage, shape.clone())),
                        ],
                    ),
                ],
                count != 0,
            );
            let before = network.clone();
            let tape = Rc::new(RefCell::new(Tape::new()));
            let _activation = TapeActivation::enter(tape.clone());
            let error = annotate_network_object(&mut network).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("single or double parent storage"),
                "{dtype:?}: {error}"
            );
            assert!(tape.borrow().nodes.is_empty(), "{dtype:?}");
            assert_eq!(network.properties(), before.properties(), "{dtype:?}");
        }
    }
}

#[test]
fn uninitialized_network_preserves_its_lifecycle_and_fresh_tapes_get_fresh_leaves() {
    let mut network = network(vec![
        layer(
            DeepLearningLayerKind::FeatureInput,
            "input",
            &[("InputSize", Value::Num(2.0))],
        ),
        layer(
            DeepLearningLayerKind::FullyConnected,
            "dense",
            &[
                ("OutputSize", Value::Num(3.0)),
                ("Bias", tensor(&[3, 1], true)),
            ],
        ),
    ]);
    assert_eq!(
        network.properties().get("Initialized"),
        Some(&Value::Bool(false))
    );
    for _ in 0..2 {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        annotate_network_object(&mut network).unwrap();
        assert_eq!(tape.borrow().nodes.len(), 1);
        assert_eq!(
            network.properties().get("Initialized"),
            Some(&Value::Bool(false))
        );
        let parameters = parameters::view(&network, "dlfeval").unwrap();
        assert!(parameters[0].value.is_none());
        assert!(parameters[1].value.is_some());
    }
}

#[test]
fn resident_parameter_becomes_a_tape_leaf_without_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: &[3, 2],
            })
            .unwrap();
        let mut network = network(vec![
            layer(
                DeepLearningLayerKind::FeatureInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::FullyConnected,
                "dense",
                &[
                    ("OutputSize", Value::Num(3.0)),
                    ("Weights", Value::GpuTensor(resident.clone())),
                    ("Bias", tensor(&[3, 1], false)),
                ],
            ),
        ]);
        let initialized = network.properties().get("Initialized").cloned();
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        provider.reset_telemetry();
        annotate_network_object(&mut network).expect("resident parameter leaf");
        assert_eq!(
            network.properties().get("Initialized"),
            initialized.as_ref()
        );
        let parameters = parameters::view(&network, "dlfeval").unwrap();
        let Some(Value::Object(weights)) = parameters[0].value else {
            panic!("annotated Weights dlarray");
        };
        let Some(Value::GpuTensor(data)) = weights.properties().get("Data") else {
            panic!("resident Weights data");
        };
        assert_eq!(data.view_identity(), resident.view_identity());
        assert_eq!(data.shape(), &[3, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(data),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        assert_eq!(dlarray_object_node_id(weights), Some(0));
        let Some(Value::Object(bias)) = parameters[1].value else {
            panic!("annotated Bias dlarray");
        };
        let Some(Value::Tensor(bias_data)) = bias.properties().get("Data") else {
            panic!("native Bias data");
        };
        assert_eq!(bias_data.shape(), &[3, 1]);
        assert_eq!(bias_data.numeric_dtype(), NumericDType::F64);
        assert_eq!(bias_data.materialize_f64(), vec![1.0; 3]);
        assert_eq!(dlarray_object_node_id(bias), Some(1));
        let nodes = tape.borrow();
        assert_eq!(nodes.nodes.len(), 2);
        let NodeValue::Resident(leaf) = &nodes.nodes[0].value else {
            panic!("resident Weights tape leaf");
        };
        assert_eq!(leaf.view_identity(), resident.view_identity());
        assert_eq!(leaf.shape(), &[3, 2]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(resident.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn resident_parameter_precedes_invalid_integer_without_gather_or_partial_tape() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: &[3, 2],
            })
            .unwrap();
        let mut network = network(vec![
            layer(
                DeepLearningLayerKind::FeatureInput,
                "input",
                &[("InputSize", Value::Num(2.0))],
            ),
            layer(
                DeepLearningLayerKind::FullyConnected,
                "dense",
                &[
                    ("OutputSize", Value::Num(3.0)),
                    ("Weights", Value::GpuTensor(resident.clone())),
                    (
                        "Bias",
                        integer_tensor(IntegerStorage::U8(vec![1, 2, 3]), vec![3, 1]),
                    ),
                ],
            ),
        ]);
        let before = network.clone();
        let tape = Rc::new(RefCell::new(Tape::new()));
        let _activation = TapeActivation::enter(tape.clone());
        provider.reset_telemetry();
        let error = annotate_network_object(&mut network).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(error
            .to_string()
            .contains("single or double parent storage"));
        assert!(tape.borrow().nodes.is_empty());
        assert_eq!(network.properties(), before.properties());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(resident.allocation_lease().validate_live().is_ok());
    });
}
