use super::*;
use crate::catalog::inference::infer_catalog_call;
use runmat_types::*;
use std::collections::BTreeMap;

fn call(
    entry: &crate::BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    outputs: usize,
) -> CallInference {
    infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

fn receiver() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::super::data_family::DATA_ARRAY_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn all_ten_array_identities_are_sorted_source_linked_and_bound() {
    let names: Vec<_> = ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(
        names,
        [
            "DataArray.chunk_shape",
            "DataArray.codec",
            "DataArray.dtype",
            "DataArray.fill",
            "DataArray.name",
            "DataArray.rank",
            "DataArray.read",
            "DataArray.resize",
            "DataArray.shape",
            "DataArray.write",
        ]
    );
    assert!(ENTRIES.iter().all(|entry| !entry.bindings.is_empty()));
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn read_slice_and_mutation_controls_do_not_admit_ignored_extras() {
    assert_eq!(DATAARRAY_RESIZE_DESCRIPTOR.signatures[0].inputs.len(), 2);
    assert_eq!(DATAARRAY_FILL_DESCRIPTOR.signatures[0].inputs.len(), 2);
    let value = ValueFact::unknown(DynamicReason::RuntimeValue);
    for (entry, valid, invalid) in [
        (&DATAARRAY_READ_CATALOG_ENTRY, 2, 3),
        (&DATAARRAY_RESIZE_CATALOG_ENTRY, 2, 3),
        (&DATAARRAY_FILL_CATALOG_ENTRY, 2, 3),
    ] {
        let args = [receiver(), value.clone(), value.clone()];
        assert!(call(entry, args[..valid].to_vec(), 1)
            .diagnostics
            .is_empty());
        assert!(call(entry, args[..invalid].to_vec(), 1)
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-DATA-ARRAY-ARITY"));
    }
}

#[test]
fn metadata_and_payload_facts_do_not_guess_stored_dtype() {
    let shape = call(&DATAARRAY_SHAPE_CATALOG_ENTRY, vec![receiver()], 1);
    assert_eq!(shape.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    assert!(matches!(
        shape.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
    let read = call(&DATAARRAY_READ_CATALOG_ENTRY, vec![receiver()], 1);
    assert_eq!(read.outputs[0].residency, ResidencyFact::Host);
    assert!(matches!(read.outputs[0].kind, ValueKindFact::Unknown));
    assert_eq!(DATAARRAY_READ_CATALOG_ENTRY.integer_capabilities.len(), 1);
    assert_eq!(DATAARRAY_WRITE_CATALOG_ENTRY.descriptor.signatures.len(), 2);
    assert_eq!(DATAARRAY_RESIZE_CATALOG_ENTRY.integer_capabilities.len(), 1);
}

#[test]
fn known_invalid_receiver_slice_shape_value_and_output_are_diagnosed() {
    let synthetic = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::super::data_family::DATA_ARRAY_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    assert!(call(&DATAARRAY_NAME_CATALOG_ENTRY, vec![synthetic], 1)
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-ARRAY-RECEIVER"));
    let bad_slice = call(
        &DATAARRAY_READ_CATALOG_ENTRY,
        vec![receiver(), ValueFact::scalar(ValueKindFact::Logical)],
        1,
    );
    assert!(bad_slice
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-ARRAY-SLICE"));
    let bad_shape = call(
        &DATAARRAY_RESIZE_CATALOG_ENTRY,
        vec![receiver(), ValueFact::scalar(ValueKindFact::String)],
        1,
    );
    assert!(bad_shape
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-ARRAY-SHAPE"));
    let bad_value = call(
        &DATAARRAY_FILL_CATALOG_ENTRY,
        vec![receiver(), ValueFact::scalar(ValueKindFact::String)],
        1,
    );
    assert!(bad_value
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-DATA-ARRAY-VALUE"));
    assert!(!call(&DATAARRAY_RANK_CATALOG_ENTRY, vec![receiver()], 2)
        .diagnostics
        .is_empty());
}
