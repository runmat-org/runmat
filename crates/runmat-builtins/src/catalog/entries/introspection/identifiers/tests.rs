use super::*;
use runmat_types::*;
#[test]
fn unknown_keyword_text_does_not_invent_a_truth_value() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_iskeyword(&request, &ISKEYWORD_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
}
#[test]
fn identifier_output_facts_follow_forms() {
    let zero = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let names = inference::infer_iskeyword(&zero, &ISKEYWORD_CATALOG_ENTRY);
    assert!(matches!(names.outputs[0].kind, ValueKindFact::Cell(_)));
    let one = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        ..zero.clone()
    };
    let predicate = inference::infer_iskeyword(&one, &ISKEYWORD_CATALOG_ENTRY);
    assert_eq!(
        predicate.outputs[0],
        ValueFact::scalar(ValueKindFact::Logical)
    );
    let length = inference::infer_namelengthmax(&zero, &NAMELENGTHMAX_CATALOG_ENTRY);
    assert!(matches!(length.outputs[0].kind, ValueKindFact::Numeric(_)));
}
