use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn constructor_proves_canonical_interpolant_class_and_properties() {
    let values = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let inferred =
        inference::constructor(&request(vec![values]), &GRIDDED_INTERPOLANT_CATALOG_ENTRY);
    let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
        panic!("constructor must produce an object fact");
    };
    assert_eq!(
        object.runtime_class,
        Some(GRIDDED_INTERPOLANT_CLASS_IDENTITY.owned())
    );
    for name in ["GridVectors", "Values", "Method", "ExtrapolationMethod"] {
        assert!(object.properties.contains_key(name));
    }
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn indexing_keeps_dynamic_result_and_rejects_known_wrong_receiver() {
    let mut constructed =
        inference::constructor(&request(vec![]), &GRIDDED_INTERPOLANT_CATALOG_ENTRY);
    let object = constructed.outputs.remove(0);
    let path = ValueFact::unknown(DynamicReason::RuntimeValue);
    let inferred = inference::subsref(
        &request(vec![object, path.clone()]),
        &GRIDDED_INTERPOLANT_SUBSREF_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);

    let wrong = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(StaticClassIdentity::new("containers.Map").owned()),
        properties: std::collections::BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(true),
    }));
    let rejected = inference::subsref(
        &request(vec![wrong, path]),
        &GRIDDED_INTERPOLANT_SUBSREF_CATALOG_ENTRY,
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRIDDED-SUBSREF-RECEIVER"));
}
