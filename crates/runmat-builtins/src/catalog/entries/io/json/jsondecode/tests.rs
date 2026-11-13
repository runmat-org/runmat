use super::*;
use runmat_types::*;

fn infer(input: ValueFact, literal: LiteralValue) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![literal]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &JSONDECODE_CATALOG_ENTRY,
    )
}

#[test]
fn literal_scalar_facts_and_invalid_json_are_diagnostic() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let number = infer(text.clone(), LiteralValue::String("42".into()));
    assert!(number.diagnostics.is_empty(), "{:?}", number.diagnostics);
    assert_eq!(
        number.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    let logical = infer(text.clone(), LiteralValue::String("true".into()));
    assert_eq!(logical.outputs[0].kind, ValueKindFact::Logical);
    let malformed = infer(text, LiteralValue::String("{".into()));
    assert!(!malformed.diagnostics.is_empty());
}

#[test]
fn numeric_and_known_multiline_char_shapes_reject_but_unknown_is_not_rejected() {
    let numeric = infer(
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        LiteralValue::Unknown,
    );
    assert!(!numeric.diagnostics.is_empty());
    let character = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert!(!infer(character, LiteralValue::Unknown)
        .diagnostics
        .is_empty());
    assert!(infer(
        ValueFact::unknown(DynamicReason::RuntimeValue),
        LiteralValue::Unknown,
    )
    .diagnostics
    .is_empty());
}
