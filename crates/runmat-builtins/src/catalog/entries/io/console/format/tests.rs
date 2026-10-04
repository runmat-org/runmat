use super::*;
use runmat_types::ShapeFact;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(argument: Option<ValueFact>, literal: Option<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: argument.into_iter().collect(),
        literals: literal.map_or_else(LiteralContext::default, |value| {
            LiteralContext::new(vec![value])
        }),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    }
}

#[test]
fn no_public_output_and_known_modes_match_runtime_surface() {
    let entry = &FORMAT_CATALOG_ENTRY;
    let no_args = inference::infer(&request(None, None), entry);
    assert!(no_args.outputs.is_empty());
    assert!(no_args.diagnostics.is_empty());
    let text = ValueFact::scalar(ValueKindFact::String);
    assert!(inference::infer(
        &request(
            Some(text.clone()),
            Some(LiteralValue::String("SHORT".into()))
        ),
        entry
    )
    .diagnostics
    .is_empty());
    assert_eq!(
        inference::infer(
            &request(Some(text), Some(LiteralValue::String("unknown".into()))),
            entry
        )
        .diagnostics
        .len(),
        1
    );
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert_eq!(
        inference::infer(&request(Some(numeric), None), entry)
            .diagnostics
            .len(),
        1
    );
}

#[test]
fn character_pages_are_invalid_but_trailing_singleton_is_possible() {
    let entry = &FORMAT_CATALOG_ENTRY;
    let character = |shape| {
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(shape),
            runmat_types::StorageFact::Dense,
        )
    };
    assert!(inference::infer(
        &request(Some(character(vec![Some(1), Some(2), Some(2)])), None),
        entry
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-FORMAT-TEXT"));
    assert!(inference::infer(
        &request(Some(character(vec![Some(1), Some(2), Some(1)])), None),
        entry
    )
    .diagnostics
    .is_empty());
}
