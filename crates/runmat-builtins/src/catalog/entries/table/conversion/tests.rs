use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn all_six_bindings_are_canonical_and_documented() {
    for entry in ENTRIES {
        assert_eq!(
            entry.bindings, &REQUIRED_DEFAULT_BINDING,
            "{}",
            entry.identity.name
        );
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(
            !entry.documentation.examples.is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
}

#[test]
fn constructors_infer_scalar_table_identity_without_invented_schema() {
    for entry in [
        &array2table::ENTRY,
        &cell2table::ENTRY,
        &struct2table::ENTRY,
    ] {
        let result = infer_catalog_call(entry, &call(vec![unknown()], 1));
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("{} must infer table identity", entry.identity.name);
        };
        assert_eq!(
            object.runtime_class,
            Some(runmat_types::standard::TABLE.owned())
        );
        assert!(!object.properties_complete);
    }
}

#[test]
fn array2table_does_not_reject_admitted_object_valued_columns() {
    let datetime = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DATETIME.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Opaque,
    );
    let result = infer_catalog_call(&array2table::ENTRY, &call(vec![datetime], 1));
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn conversion_outputs_are_sound_when_table_schema_is_unknown() {
    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TABLE.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let array = infer_catalog_call(&table2array::ENTRY, &call(vec![table.clone()], 1));
    assert!(matches!(array.outputs[0].kind, ValueKindFact::Unknown));
    let cell = infer_catalog_call(&table2cell::ENTRY, &call(vec![table.clone()], 1));
    assert!(matches!(cell.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(cell.outputs[0].shape.known_dims(), Some(vec![None, None]));
    let structure = infer_catalog_call(&table2struct::ENTRY, &call(vec![table], 1));
    assert!(matches!(
        structure.outputs[0].kind,
        ValueKindFact::Struct(_)
    ));
    assert_eq!(structure.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn impossible_known_inputs_and_missing_inputs_are_diagnosed() {
    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for entry in ENTRIES {
        assert!(!infer_catalog_call(entry, &call(Vec::new(), 1))
            .diagnostics
            .is_empty());
    }
    for entry in [
        &cell2table::ENTRY,
        &struct2table::ENTRY,
        &table2array::ENTRY,
        &table2cell::ENTRY,
        &table2struct::ENTRY,
    ] {
        assert!(!infer_catalog_call(entry, &call(vec![number.clone()], 1))
            .diagnostics
            .is_empty());
    }
}
