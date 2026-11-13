use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use crate::builtins::deep_learning::autodiff::array_methods::dlarray_sum_builtin;
use crate::builtins::deep_learning::autodiff::tape::{annotate_tree, Tape, TapeActivation};
use futures::executor::block_on;
use runmat_builtins::DeepLearningLayerKind;
use runmat_value::StringArray;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

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

fn network(layers: Vec<Value>) -> Value {
    model::network_object(model::DLNETWORK_CLASS, layers, BTreeMap::new(), "dlfeval").unwrap()
}

fn table_columns(table: &Value) -> (Vec<String>, Vec<String>, Vec<Value>) {
    let Value::Object(table) = table else {
        panic!("table expected")
    };
    let variables = crate::builtins::table::table_variables(table).unwrap();
    let Some(Value::StringArray(layers)) = variables.fields.get("Layer") else {
        panic!("Layer column expected")
    };
    let Some(Value::StringArray(roles)) = variables.fields.get("Parameter") else {
        panic!("Parameter column expected")
    };
    let Some(Value::Cell(values)) = variables.fields.get("Value") else {
        panic!("Value column expected")
    };
    (
        layers
            .data()
            .iter()
            .map(|text| text.as_str().expect("present layer fixture").to_owned())
            .collect(),
        roles
            .data()
            .iter()
            .map(|text| text.as_str().expect("present parameter fixture").to_owned())
            .collect(),
        values.data().to_vec(),
    )
}

fn gradient_for_public_table(network: Value, state: bool) -> (Value, Value) {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let network = annotate_tree(network).unwrap();
    let table = if state {
        block_on(model::get_state_builtin(network)).unwrap()
    } else {
        block_on(model::get_learnables_builtin(network)).unwrap()
    };
    let (_, _, values) = table_columns(&table);
    let loss = dlarray_sum_builtin(
        values[0].clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("all".into())]),
    )
    .unwrap();
    let gradient = block_on(dlgradient_direct(
        loss,
        crate::builtins::common::test_support::builtin_arguments(vec![table.clone()]),
    ))
    .unwrap();
    (table, gradient)
}

#[test]
fn public_learnables_table_gradient_preserves_identities_and_mixed_precision() {
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
                (
                    "Weights",
                    Value::Tensor(Tensor::from_f32(vec![1.; 4], vec![2, 2]).unwrap()),
                ),
                (
                    "Bias",
                    Value::Tensor(Tensor::new(vec![1.; 2], vec![2, 1]).unwrap()),
                ),
            ],
        ),
    ]);
    let (target, gradient) = gradient_for_public_table(network, false);
    let (layers, roles, original) = table_columns(&target);
    let (gradient_layers, gradient_roles, values) = table_columns(&gradient);
    assert_eq!(gradient_layers, layers);
    assert_eq!(gradient_roles, roles);
    assert_eq!(roles, ["Weights", "Bias"]);
    assert_eq!(original.len(), values.len());
    let Value::Object(weights) = &values[0] else {
        panic!("Weights gradient dlarray expected")
    };
    let Some(Value::Tensor(weights)) = weights.properties().get("Data") else {
        panic!("Weights gradient tensor expected")
    };
    assert_eq!(weights.numeric_dtype(), NumericDType::F32);
    assert_eq!(weights.materialize_f64(), [1.; 4]);
    let Value::Object(bias) = &values[1] else {
        panic!("Bias gradient dlarray expected")
    };
    let Some(Value::Tensor(bias)) = bias.properties().get("Data") else {
        panic!("Bias gradient tensor expected")
    };
    assert_eq!(bias.numeric_dtype(), NumericDType::F64);
    assert_eq!(bias.materialize_f64(), [0.; 2]);
}

#[test]
fn public_state_table_gradient_preserves_identities_and_mixed_precision() {
    let network = network(vec![
        layer(
            DeepLearningLayerKind::SequenceInput,
            "input",
            &[("InputSize", Value::Num(2.0))],
        ),
        layer(
            DeepLearningLayerKind::Lstm,
            "memory",
            &[
                ("NumHiddenUnits", Value::Num(2.0)),
                (
                    "HiddenState",
                    Value::Tensor(Tensor::from_f32(vec![1.; 2], vec![2, 1]).unwrap()),
                ),
                (
                    "CellState",
                    Value::Tensor(Tensor::new(vec![1.; 2], vec![2, 1]).unwrap()),
                ),
            ],
        ),
    ]);
    let (target, gradient) = gradient_for_public_table(network, true);
    let (layers, roles, _) = table_columns(&target);
    let (gradient_layers, gradient_roles, values) = table_columns(&gradient);
    assert_eq!(gradient_layers, layers);
    assert_eq!(gradient_roles, roles);
    assert_eq!(roles, ["HiddenState", "CellState"]);
    let Value::Object(hidden) = &values[0] else {
        panic!("HiddenState gradient dlarray expected")
    };
    let Some(Value::Tensor(hidden)) = hidden.properties().get("Data") else {
        panic!("HiddenState gradient tensor expected")
    };
    assert_eq!(hidden.numeric_dtype(), NumericDType::F32);
    assert_eq!(hidden.materialize_f64(), [1.; 2]);
    let Value::Object(cell) = &values[1] else {
        panic!("CellState gradient dlarray expected")
    };
    let Some(Value::Tensor(cell)) = cell.properties().get("Data") else {
        panic!("CellState gradient tensor expected")
    };
    assert_eq!(cell.numeric_dtype(), NumericDType::F64);
    assert_eq!(cell.materialize_f64(), [0.; 2]);
}

fn network_table(layers: Vec<String>, roles: Vec<String>, values: Vec<Value>) -> Value {
    crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(StringArray::new(layers.clone(), vec![layers.len(), 1]).unwrap()),
            Value::StringArray(StringArray::new(roles.clone(), vec![roles.len(), 1]).unwrap()),
            Value::Cell(CellArray::new(values.clone(), values.len(), 1).unwrap()),
        ],
    )
    .unwrap()
}

#[test]
fn public_parameter_table_names_are_metadata_not_builtin_role_authority() {
    for (layers, roles) in [
        (vec!["custom"], vec!["CustomCoefficient"]),
        (vec!["dense", "memory"], vec!["Weights", "CellState"]),
        (vec!["dense", "dense"], vec!["Weights", "Weights"]),
    ] {
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let values = roles
            .iter()
            .map(|_| {
                annotate_tree(
                    super::super::super::object_with_identity(
                        DLARRAY_CLASS,
                        vec![
                            (
                                "Data",
                                Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
                            ),
                            ("Format", Value::String(String::new().into())),
                            ("Labels", Value::String(String::new().into())),
                        ],
                    )
                    .expect("fixture object"),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let loss = dlarray_sum_builtin(
            values[0].clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        )
        .unwrap();
        let target = network_table(
            layers.into_iter().map(String::from).collect(),
            roles.into_iter().map(String::from).collect(),
            values,
        );
        let gradient = block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![target.clone()]),
        ))
        .unwrap();
        let (layers, roles, _) = table_columns(&target);
        let (gradient_layers, gradient_roles, values) = table_columns(&gradient);
        assert_eq!(gradient_layers, layers);
        assert_eq!(gradient_roles, roles);
        for (index, value) in values.iter().enumerate() {
            let Value::Object(array) = value else {
                panic!("gradient dlarray expected")
            };
            let Some(Value::Tensor(data)) = array.properties().get("Data") else {
                panic!("gradient tensor expected")
            };
            assert_eq!(data.numeric_dtype(), NumericDType::F32);
            assert_eq!(data.shape(), &[1, 1]);
            assert_eq!(
                data.as_f32_slice().unwrap(),
                &[if index == 0 { 1.0 } else { 0.0 }]
            );
        }
    }
}

#[test]
fn empty_canonical_table_retains_its_three_columns() {
    let target = network_table(vec![], vec![], vec![]);
    let gradient = gradient_tree_for_target(&target, &HashMap::new()).unwrap();
    assert_eq!(table_columns(&gradient), table_columns(&target));
}

#[test]
fn parameter_table_column_order_does_not_change_target_admission() {
    for names in [
        ["Layer", "Parameter", "Value"],
        ["Layer", "Value", "Parameter"],
        ["Parameter", "Layer", "Value"],
        ["Parameter", "Value", "Layer"],
        ["Value", "Layer", "Parameter"],
        ["Value", "Parameter", "Layer"],
    ] {
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let traced = annotate_tree(
            super::super::super::object_with_identity(
                DLARRAY_CLASS,
                vec![
                    (
                        "Data",
                        Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
                    ),
                    ("Format", Value::String(String::new().into())),
                    ("Labels", Value::String(String::new().into())),
                ],
            )
            .expect("fixture object"),
        )
        .unwrap();
        let loss = dlarray_sum_builtin(
            traced.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        )
        .unwrap();
        let columns = names
            .iter()
            .map(|name| match *name {
                "Layer" => {
                    Value::StringArray(StringArray::new(vec!["custom".into()], vec![1, 1]).unwrap())
                }
                "Parameter" => Value::StringArray(
                    StringArray::new(vec!["coefficient".into()], vec![1, 1]).unwrap(),
                ),
                "Value" => Value::Cell(CellArray::new(vec![traced.clone()], 1, 1).unwrap()),
                _ => unreachable!("fixed public schema fixture"),
            })
            .collect();
        let target = crate::builtins::table::table_from_columns(
            names.iter().map(|name| (*name).into()).collect(),
            columns,
        )
        .unwrap();
        let gradient = block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![target]),
        ))
        .unwrap();
        let Value::Object(table) = &gradient else {
            panic!("table gradient expected")
        };
        assert_eq!(
            crate::builtins::table::table_variable_names_from_object(table).unwrap(),
            names
        );
        let (layers, parameters, values) = table_columns(&gradient);
        assert_eq!(layers, ["custom"]);
        assert_eq!(parameters, ["coefficient"]);
        let Value::Object(array) = &values[0] else {
            panic!("gradient dlarray expected")
        };
        let Some(Value::Tensor(data)) = array.properties().get("Data") else {
            panic!("gradient data expected")
        };
        assert_eq!(data.numeric_dtype(), NumericDType::F32);
        assert_eq!(data.as_f32_slice().unwrap(), &[1.0]);
    }
}

#[test]
fn canonical_metadata_requires_typed_string_columns() {
    let target = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::StringArray(StringArray::new(vec!["Weights".into()], vec![1, 1]).unwrap()),
            Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        ],
    )
    .unwrap();
    let error = gradient_tree_for_target(&target, &HashMap::new()).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
}

#[test]
fn ordinary_dlarray_table_keeps_recursive_gradient_behavior() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let traced = annotate_tree(
        super::super::super::object_with_identity(
            DLARRAY_CLASS,
            vec![
                (
                    "Data",
                    Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
                ),
                ("Format", Value::String(String::new().into())),
                ("Labels", Value::String(String::new().into())),
            ],
        )
        .expect("fixture object"),
    )
    .unwrap();
    let target = crate::builtins::table::table_from_columns(
        vec!["Payload".into()],
        vec![Value::Cell(CellArray::new(vec![traced], 1, 1).unwrap())],
    )
    .unwrap();
    let gradient = gradient_tree_for_target(&target, &HashMap::new()).unwrap();
    let Value::Object(table) = gradient else {
        panic!("gradient table expected")
    };
    let variables = crate::builtins::table::table_variables(&table).unwrap();
    let Some(Value::Cell(payload)) = variables.fields.get("Payload") else {
        panic!("Payload cell expected")
    };
    let Value::Object(dlarray) = &payload.data()[0] else {
        panic!("gradient dlarray expected")
    };
    let Some(Value::Tensor(data)) = dlarray.properties().get("Data") else {
        panic!("gradient Data expected")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F32);
    assert_eq!(data.materialize_f64(), [0.0]);
}

#[test]
fn legacy_scalar_parameter_record_preserves_extra_metadata() {
    let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
    let traced = annotate_tree(
        super::super::super::object_with_identity(
            DLARRAY_CLASS,
            vec![
                (
                    "Data",
                    Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
                ),
                ("Format", Value::String(String::new().into())),
                ("Labels", Value::String(String::new().into())),
            ],
        )
        .expect("fixture object"),
    )
    .unwrap();
    let mut record = runmat_value::UnboundedStructValueBuilder::new();
    record.insert("Layer", Value::String("custom".into()));
    record.insert("Parameter", Value::String("Gain".into()));
    record.insert("Value", traced);
    record.insert("Owner", Value::Bool(true));
    let Value::Struct(gradient) =
        gradient_tree_for_target(&Value::Struct(record.finish()), &HashMap::new()).unwrap()
    else {
        panic!("legacy record gradient expected")
    };
    assert_eq!(
        gradient.fields.get("Layer"),
        Some(&Value::String("custom".into()))
    );
    assert_eq!(
        gradient.fields.get("Parameter"),
        Some(&Value::String("Gain".into()))
    );
    assert_eq!(gradient.fields.get("Owner"), Some(&Value::Bool(true)));
    let Some(Value::Object(value)) = gradient.fields.get("Value") else {
        panic!("gradient Value must be a dlarray")
    };
    let Some(Value::Tensor(data)) = value.properties().get("Data") else {
        panic!("gradient Data expected")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F32);
    assert_eq!(data.materialize_f64(), [0.0]);
}
