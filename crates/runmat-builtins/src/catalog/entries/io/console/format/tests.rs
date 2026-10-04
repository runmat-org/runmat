use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, OutputSelection,
    RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

fn request(argument: Option<ValueFact>, literal: Option<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: argument.into_iter().collect(),
        literals: literal.map_or_else(LiteralContext::default, |value| {
            LiteralContext::new(vec![value])
        }),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn fixed_empty_result_and_known_modes_match_runtime_surface() {
    let entry = &FORMAT_CATALOG_ENTRY;
    let no_args = inference::infer(&request(None, None), entry);
    assert_eq!(
        no_args.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
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
