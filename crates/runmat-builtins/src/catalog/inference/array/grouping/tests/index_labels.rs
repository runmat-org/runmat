use std::collections::BTreeMap;

use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(input: ValueFact, outputs: RequestedOutputCount) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("grp2idx").expect("grp2idx catalog entry"),
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn numeric_input_types_indices_names_and_exact_levels() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(5)]),
        StorageFact::Dense,
    );
    let inferred = infer(input, RequestedOutputCount::Exactly(3));
    assert_eq!(inferred.outputs.len(), 3);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(5), Some(1)])
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert!(matches!(inferred.outputs[1].kind, ValueKindFact::Cell(_)));
    assert_eq!(
        inferred.outputs[2].numeric().map(|numeric| numeric.class),
        Some(NumericClass::UInt64)
    );
}

#[test]
fn string_levels_are_cellstr_and_character_observations_are_rows() {
    let strings = infer(
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        ),
        RequestedOutputCount::Exactly(3),
    );
    assert_eq!(
        strings.outputs[0].shape,
        ShapeFact::from(vec![Some(6), Some(1)])
    );
    assert!(matches!(strings.outputs[2].kind, ValueKindFact::Cell(_)));

    let characters = infer(
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(4), Some(8)]),
            StorageFact::Dense,
        ),
        RequestedOutputCount::One,
    );
    assert_eq!(
        characters.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
}

#[test]
fn known_unrelated_object_is_rejected_without_erasing_its_fact() {
    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let inferred = infer(table, RequestedOutputCount::One);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GRP2IDX-INPUT"));
}
