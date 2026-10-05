use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact,
    ValueKindFact,
};

fn request(argument: ValueFact, literal: LiteralValue) -> CallRequest {
    CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::new(vec![literal]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn warning_contract_covers_state_and_message_forms() {
    assert_eq!(WARNING_CATALOG_ENTRY.bindings.len(), 1);
    assert_eq!(WARNING_EXTENSIONS.len(), 1);
    assert_eq!(WARNING_INTEGER_CAPABILITIES.len(), 2);
    assert!(WARNING_DESCRIPTOR.signatures.len() >= 17);
}

#[test]
fn warning_inference_keeps_option_dependent_output_facts() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let query = inference::infer(
        &request(text.clone(), LiteralValue::String("query".into())),
        &WARNING_CATALOG_ENTRY,
    );
    assert_eq!(query.outputs.len(), 1);
    assert!(matches!(query.outputs[0].kind, ValueKindFact::Cell(_)));

    let state = inference::infer(
        &request(text.clone(), LiteralValue::String("off".into())),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(matches!(state.outputs[0].kind, ValueKindFact::Struct(_)));

    let message = inference::infer(
        &request(text, LiteralValue::String("hello".into())),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(matches!(message.outputs[0].kind, ValueKindFact::Numeric(_)));

    let status = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String("status".into()),
        ),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(matches!(status.outputs[0].kind, ValueKindFact::Cell(_)));

    let backtrace = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String("backtrace".into()),
        ),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(matches!(backtrace.outputs[0].kind, ValueKindFact::String));

    let reset = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String("reset".into()),
        ),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(matches!(reset.outputs[0].kind, ValueKindFact::Numeric(_)));

    let targeted_query = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("query".into()),
            LiteralValue::String("runmat:demo:test".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let targeted = inference::infer(&targeted_query, &WARNING_CATALOG_ENTRY);
    assert!(matches!(targeted.outputs[0].kind, ValueKindFact::Struct(_)));

    let mut all_query = targeted_query;
    all_query.literals.literal_args[1] = LiteralValue::String("all".into());
    let all = inference::infer(&all_query, &WARNING_CATALOG_ENTRY);
    assert!(matches!(all.outputs[0].kind, ValueKindFact::Cell(_)));
}
