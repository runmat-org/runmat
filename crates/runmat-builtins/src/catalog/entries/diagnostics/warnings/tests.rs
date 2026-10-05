use super::*;
use runmat_types::{
    CallRequest, ExceptionFact, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
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

    let mut extra_mode = all_query;
    extra_mode.literals.literal_args[0] = LiteralValue::String("off".into());
    extra_mode
        .arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    extra_mode.literals.literal_args.push(LiteralValue::Unknown);
    assert!(inference::infer(&extra_mode, &WARNING_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WARNING-MODE-ARITY"));
}

#[test]
fn warning_inference_accepts_exception_and_rejects_invalid_text_and_targets() {
    let exception = request(
        ValueFact::scalar(ValueKindFact::Exception(ExceptionFact { identifier: None })),
        LiteralValue::Unknown,
    );
    let inferred = inference::infer(&exception, &WARNING_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(_)
    ));

    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let invalid_message = inference::infer(
        &request(numeric.clone(), LiteralValue::Unknown),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(invalid_message
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WARNING-TEXT"));

    let mut invalid_text = ValueFact::scalar(ValueKindFact::String);
    invalid_text.shape = ShapeFact::from(vec![Some(1), Some(2)]);
    let invalid_text = inference::infer(
        &request(invalid_text, LiteralValue::Unknown),
        &WARNING_CATALOG_ENTRY,
    );
    assert!(invalid_text
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WARNING-TEXT"));

    let invalid_target = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String), numeric],
        literals: LiteralContext::new(vec![
            LiteralValue::String("off".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    assert!(inference::infer(&invalid_target, &WARNING_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-WARNING-TARGET-TEXT"));

    let invalid_special_mode = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("once".into()),
            LiteralValue::String(" BACKTRACE ".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    assert!(
        inference::infer(&invalid_special_mode, &WARNING_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-WARNING-SPECIAL-MODE")
    );
}
