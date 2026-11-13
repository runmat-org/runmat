use super::*;
use runmat_types::*;

fn request(input: Option<ValueFact>, literal: Option<&str>, outputs: usize) -> CallRequest {
    let literals = literal.map_or_else(LiteralContext::default, |literal| {
        LiteralContext::new(vec![LiteralValue::String(literal.into())])
    });
    CallRequest {
        arguments: input.into_iter().collect(),
        literals,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn open_query_list_and_unknown_forms_keep_distinct_output_facts() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let open = inference::infer(
        &request(Some(text.clone()), Some("input.txt"), 2),
        &FOPEN_CATALOG_ENTRY,
    );
    assert!(open.diagnostics.is_empty());
    assert!(matches!(open.outputs[0].kind, ValueKindFact::Numeric(_)));
    assert!(matches!(open.outputs[1].kind, ValueKindFact::Character));
    assert!(open
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    let query = inference::infer(&request(Some(numeric), None, 4), &FOPEN_CATALOG_ENTRY);
    assert!(query
        .outputs
        .iter()
        .all(|fact| matches!(fact.kind, ValueKindFact::Character)));
    assert!(!query.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(!query
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    let list = inference::infer(
        &request(Some(text.clone()), Some("all"), 4),
        &FOPEN_CATALOG_ENTRY,
    );
    assert!(matches!(list.outputs[0].kind, ValueKindFact::Numeric(_)));
    assert!(matches!(list.outputs[1].kind, ValueKindFact::Cell(_)));
    assert!(!list.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(!list
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    let unknown = inference::infer(&request(Some(text), None, 1), &FOPEN_CATALOG_ENTRY);
    assert!(matches!(unknown.outputs[0].kind, ValueKindFact::Unknown));
    assert!(unknown.diagnostics.is_empty());
    assert!(unknown
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    let no_args = inference::infer(&request(None, None, 1), &FOPEN_CATALOG_ENTRY);
    assert!(matches!(no_args.outputs[0].kind, ValueKindFact::Numeric(_)));
}

#[test]
fn invalid_first_argument_and_extra_slots_are_not_guessed() {
    let invalid = inference::infer(
        &request(Some(ValueFact::scalar(ValueKindFact::Logical)), None, 1),
        &FOPEN_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    let padded = inference::infer(
        &request(Some(ValueFact::scalar(ValueKindFact::String)), Some("x"), 5),
        &FOPEN_CATALOG_ENTRY,
    );
    assert!(padded.diagnostics.is_empty());
    assert_eq!(padded.outputs.len(), 5);
    assert_eq!(
        padded.outputs[4],
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    );
}

#[test]
fn known_character_rows_are_valid_paths_and_controls() {
    let row = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(9)]),
        StorageFact::Dense,
    );
    let mut call = request(Some(row.clone()), Some("input.txt"), 1);
    call.arguments.push(row);
    let inferred = inference::infer(&call, &FOPEN_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));

    call.arguments[1] = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert!(!inference::infer(&call, &FOPEN_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
