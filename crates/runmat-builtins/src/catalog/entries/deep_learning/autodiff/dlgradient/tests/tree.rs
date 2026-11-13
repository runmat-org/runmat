use super::*;
use runmat_types::{CellFact, DynamicReason, StructFact};

fn single(shape: &[usize]) -> ValueFact {
    dlarray(ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    ))
}

fn loss() -> ValueFact {
    single(&[1, 1])
}

fn infer_target(target: ValueFact) -> runmat_types::CallInference {
    crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![loss(), target],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

fn cell(element: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element.clone()),
            elements: vec![element],
            elements_complete: true,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn table(fields: BTreeMap<String, ValueFact>, complete: bool) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: BTreeMap::from([(
                "__table_variables".into(),
                ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, complete))),
            )]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn scalar_legacy_record_keeps_metadata_and_differentiates_value() {
    let record = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            ("Layer".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Parameter".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Value".into(), single(&[2, 3])),
        ]),
        true,
    )));
    let result = infer_target(record);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct gradient")
    };
    assert_eq!(output.fields["Layer"].kind, ValueKindFact::String);
    assert_eq!(output.fields["Parameter"].kind, ValueKindFact::String);
    assert_eq!(output.fields["Layer"].alias, AliasFact::Unknown);
    let ValueKindFact::Object(value) = &output.fields["Value"].kind else {
        panic!("dlarray Value gradient")
    };
    let data = &value.properties[crate::DLARRAY_DATA_PROPERTY];
    assert_eq!(data.numeric().unwrap().class, NumericClass::Single);
    assert_eq!(data.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    assert_eq!(data.alias, AliasFact::Unknown);
}

#[test]
fn scalar_legacy_record_preserves_extra_metadata_without_role_inference() {
    let record = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            ("Layer".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Parameter".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Value".into(), single(&[2, 1])),
            ("Owner".into(), ValueFact::scalar(ValueKindFact::Logical)),
        ]),
        true,
    )));
    let result = infer_target(record);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct gradient")
    };
    assert_eq!(output.fields["Layer"].kind, ValueKindFact::String);
    assert_eq!(output.fields["Owner"].kind, ValueKindFact::Logical);
    assert!(matches!(
        output.fields["Value"].kind,
        ValueKindFact::Object(_)
    ));
}

#[test]
fn struct_array_retains_position_facts_and_checks_known_invalid_member() {
    let good = BTreeMap::from([("weight".into(), single(&[1, 2]))]);
    let bad = BTreeMap::from([("weight".into(), ValueFact::scalar(ValueKindFact::String))]);
    let target = ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(
            BTreeMap::from([(
                "weight".into(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )]),
            true,
            vec![good, bad],
            true,
        )),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Opaque,
    );
    let result = infer_target(target);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"));
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct-array gradient")
    };
    assert!(output.elements_complete);
    assert_eq!(output.elements.len(), 2);
    assert!(matches!(
        output.elements[0]["weight"].kind,
        ValueKindFact::Object(_)
    ));
    assert_eq!(output.elements[1]["weight"].kind, ValueKindFact::Unknown);
}

#[test]
fn public_parameter_table_preserves_opaque_metadata_and_gradients_value_only() {
    let metadata = ValueFact::proven(ValueKindFact::String, ShapeFact::Scalar, StorageFact::Dense);
    let target = table(
        BTreeMap::from([
            ("Layer".into(), metadata.clone()),
            ("Parameter".into(), metadata),
            ("Value".into(), cell(single(&[3, 1]))),
        ]),
        true,
    );
    let result = infer_target(target);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(table) = &result.outputs[0].kind else {
        panic!("table gradient")
    };
    let ValueKindFact::Struct(variables) = &table.properties["__table_variables"].kind else {
        panic!("table Variables")
    };
    assert_eq!(variables.fields["Layer"].kind, ValueKindFact::String);
    let ValueKindFact::Cell(values) = &variables.fields["Value"].kind else {
        panic!("Value column")
    };
    let ValueKindFact::Object(value) = &values.elements[0].kind else {
        panic!("gradient leaf")
    };
    assert_eq!(
        value.properties[crate::DLARRAY_DATA_PROPERTY]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Single
    );
}

#[test]
fn generic_table_gradients_all_known_columns() {
    let target = table(
        BTreeMap::from([
            ("A".into(), cell(single(&[2, 1]))),
            ("B".into(), ValueFact::scalar(ValueKindFact::String)),
        ]),
        true,
    );
    let result = infer_target(target);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"));
    let ValueKindFact::Object(output) = &result.outputs[0].kind else {
        panic!("table gradient")
    };
    let ValueKindFact::Struct(variables) = &output.properties["__table_variables"].kind else {
        panic!("table Variables")
    };
    assert!(matches!(variables.fields["A"].kind, ValueKindFact::Cell(_)));
    assert_eq!(variables.fields["B"].kind, ValueKindFact::Unknown);
}

#[test]
fn network_target_projects_a_table_without_guessing_rows() {
    let network = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::DL_NETWORK.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    let result = infer_target(network);
    let ValueKindFact::Object(table) = &result.outputs[0].kind else {
        panic!("network gradient table")
    };
    assert_eq!(table.runtime_class, Some(standard::TABLE.owned()));
    assert!(!table.properties_complete);
    assert!(!table.properties.contains_key("__table_variables"));
}

#[test]
fn ambiguous_singleton_struct_metadata_stays_dynamic() {
    let leaf = single(&[1, 1]);
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            ("Layer".into(), leaf.clone()),
            ("Parameter".into(), leaf.clone()),
            ("Value".into(), leaf),
        ]),
        true,
    )));
    let result = infer_target(target);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct gradient")
    };
    // A scalar StructValue preserves metadata; a one-element StructArray
    // differentiates every field. StructFact does not distinguish them.
    assert_eq!(output.fields["Layer"].kind, ValueKindFact::Unknown);
    assert!(matches!(
        output.fields["Value"].kind,
        ValueKindFact::Object(_)
    ));
}

#[test]
fn nonempty_open_struct_reports_invalid_common_field_without_guessing_members() {
    let target = ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(
            BTreeMap::from([("value".into(), ValueFact::scalar(ValueKindFact::String))]),
            false,
            Vec::new(),
            false,
        )),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Opaque,
    );
    let result = infer_target(target);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"));
}

#[test]
fn malformed_known_parameter_table_reports_schema_before_runtime() {
    let target = table(
        BTreeMap::from([
            (
                "Layer".into(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ),
            ("Parameter".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Value".into(), cell(single(&[1, 1]))),
        ]),
        true,
    );
    let result = infer_target(target);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TABLE"));
}

#[test]
fn known_network_projection_checks_value_column_but_not_names() {
    let learnables = table(
        BTreeMap::from([
            ("Layer".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Parameter".into(), ValueFact::scalar(ValueKindFact::String)),
            (
                "Value".into(),
                cell(ValueFact::scalar(ValueKindFact::String)),
            ),
        ]),
        true,
    );
    let network = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::DL_NETWORK.owned()),
        properties: BTreeMap::from([("Learnables".into(), learnables)]),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    let result = infer_target(network);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLGRADIENT-TARGET"));
}

#[test]
fn discarded_table_tree_is_not_admitted_or_diagnosed() {
    let invalid_table = table(
        BTreeMap::from([("bad".into(), ValueFact::scalar(ValueKindFact::String))]),
        true,
    );
    let arguments = vec![loss(), single(&[1, 1]), invalid_table];
    let result = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments: arguments.clone(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 1);
    let selected = crate::infer_catalog_call(
        &DLGRADIENT_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(selected.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-DL-DLGRADIENT-TARGET" && diagnostic.argument == Some(2)
    }));
}
