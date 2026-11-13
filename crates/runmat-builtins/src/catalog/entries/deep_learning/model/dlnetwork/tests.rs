use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &DLNETWORK_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn layer() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("nnet.cnn.layer.ReLULayer").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn arbitrary_layer_namespace_identity_is_not_treated_as_supported() {
    let unsupported = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(
                StaticClassIdentity::new("nnet.cnn.layer.CustomFutureLayer").owned(),
            ),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let inferred = infer(vec![unsupported], vec![LiteralValue::Unknown]);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLNETWORK-LAYERS"));
}

#[test]
fn complete_layer_cell_with_conflicting_geometry_is_diagnosed() {
    let layers = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(layer()),
            elements: vec![layer()],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let inferred = infer(vec![layers], vec![LiteralValue::Unknown]);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLNETWORK-LAYERS"));
}

#[test]
fn static_network_identity_and_layer_geometry_follow_known_collection() {
    let result = infer(
        vec![ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(layer()),
                elements: vec![layer(), layer()],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        )],
        vec![LiteralValue::Unknown],
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("network object expected")
    };
    assert_eq!(
        network.runtime_class,
        Some(runmat_types::standard::DL_NETWORK.owned())
    );
    assert_eq!(
        network.properties["Layers"].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    let ValueKindFact::Cell(layers) = &network.properties["Layers"].kind else {
        panic!("cell fact expected")
    };
    assert_eq!(layers.elements.len(), 2);
    assert!(!network.properties_complete);
    let ValueKindFact::Object(learnables) = &network.properties["Learnables"].kind else {
        panic!("Learnables must be inferred as a table object")
    };
    assert_eq!(
        learnables.runtime_class,
        Some(runmat_types::standard::TABLE.owned())
    );
    let ValueKindFact::Struct(variables) = &learnables.properties["__table_variables"].kind else {
        panic!("Learnables table variables must be known")
    };
    assert!(variables.fields.contains_key("Layer"));
    assert!(variables.fields.contains_key("Parameter"));
    assert!(variables.fields.contains_key("Value"));
    for name in ["Layer", "Parameter"] {
        assert!(matches!(variables.fields[name].kind, ValueKindFact::String));
        assert_eq!(
            variables.fields[name].shape,
            ShapeFact::from(vec![None, Some(1)])
        );
    }
    assert!(matches!(
        variables.fields["Value"].kind,
        ValueKindFact::Cell(_)
    ));
    let ValueKindFact::Object(state) = &network.properties["State"].kind else {
        panic!("State must be inferred as a table object")
    };
    assert_eq!(
        state.runtime_class,
        Some(runmat_types::standard::TABLE.owned())
    );
}

#[test]
fn dependent_learnables_accessors_preserve_table_and_network_facts() {
    let network = infer(vec![layer()], vec![LiteralValue::Unknown])
        .outputs
        .remove(0);
    let get = crate::infer_catalog_call(
        &property_access::GET_LEARNABLES_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![network.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(get.diagnostics.is_empty());
    let ValueKindFact::Object(table) = &get.outputs[0].kind else {
        panic!("table fact expected")
    };
    assert_eq!(
        table.runtime_class,
        Some(runmat_types::standard::TABLE.owned())
    );
    let set = crate::infer_catalog_call(
        &property_access::SET_LEARNABLES_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![network.clone(), get.outputs[0].clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(set.diagnostics.is_empty());
    assert_eq!(set.outputs[0].kind, network.kind);
}

#[test]
fn dependent_state_accessors_preserve_table_and_network_facts() {
    let network = infer(vec![layer()], vec![LiteralValue::Unknown])
        .outputs
        .remove(0);
    let get = crate::infer_catalog_call(
        &state_access::GET_STATE_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![network.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(get.diagnostics.is_empty());
    assert!(matches!(&get.outputs[0].kind, ValueKindFact::Object(table)
        if table.runtime_class == Some(runmat_types::standard::TABLE.owned())));
    let set = crate::infer_catalog_call(
        &state_access::SET_STATE_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![network.clone(), get.outputs[0].clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(set.diagnostics.is_empty());
    assert_eq!(set.outputs[0].kind, network.kind);
}

#[test]
fn malformed_known_layers_and_options_are_diagnosed() {
    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let bad_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(number.clone()),
            elements: vec![number.clone()],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(!infer(vec![bad_cell], vec![LiteralValue::Unknown])
        .diagnostics
        .is_empty());
    assert!(!infer(vec![number], vec![LiteralValue::Unknown])
        .diagnostics
        .is_empty());
    assert!(!infer(
        vec![
            layer(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue)
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("InputNames".into()),
            LiteralValue::Unknown
        ],
    )
    .diagnostics
    .is_empty());
}

#[test]
fn constructor_option_identity_is_case_insensitive_in_static_admission() {
    let result = infer(
        vec![
            layer(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("outputnames".into()),
            LiteralValue::String("relu".into()),
            LiteralValue::String("INITIALIZE".into()),
            LiteralValue::Bool(false),
        ],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("network object expected")
    };
    assert_eq!(
        network.properties["Initialize"].kind,
        ValueKindFact::Logical
    );
}

#[test]
fn initialize_option_diagnoses_only_proven_invalid_numeric_values() {
    let real_scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let complex_scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let integer_scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let check = |value, literal| {
        infer(
            vec![layer(), ValueFact::scalar(ValueKindFact::String), value],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("Initialize".into()),
                literal,
            ],
        )
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLNETWORK-OPTION-VALUE")
    };
    assert!(check(real_scalar.clone(), LiteralValue::Number(2.0)));
    assert!(check(real_scalar.clone(), LiteralValue::Number(f64::NAN)));
    assert!(check(complex_scalar, LiteralValue::Unknown));
    assert!(check(
        integer_scalar.clone(),
        LiteralValue::Integer {
            text: "18446744073709551615".into(),
            class: NumericClass::UInt64,
        }
    ));
    assert!(!check(
        integer_scalar,
        LiteralValue::Integer {
            text: "1".into(),
            class: NumericClass::UInt64,
        }
    ));
    assert!(!check(real_scalar.clone(), LiteralValue::Number(0.0)));
    assert!(!check(real_scalar.clone(), LiteralValue::Number(1.0)));
    assert!(!check(real_scalar, LiteralValue::Unknown));
    assert!(!check(
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })),
        LiteralValue::Real {
            text: "1.00000001".into(),
            class: NumericClass::Single,
        }
    ));
}

#[test]
fn unknown_object_identity_does_not_imply_one_layer() {
    let mut source = layer();
    let ValueKindFact::Object(object) = &mut source.kind else {
        unreachable!()
    };
    object.runtime_class = None;
    let result = infer(vec![source], vec![LiteralValue::Unknown]);
    let ValueKindFact::Object(network) = &result.outputs[0].kind else {
        panic!("network object expected")
    };
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        network.properties["Layers"].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    let ValueKindFact::Cell(layers) = &network.properties["Layers"].kind else {
        panic!("cell fact expected")
    };
    assert!(!layers.elements_complete);
    assert!(layers.elements.is_empty());
}
