use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn all_runtime_timer_bindings_have_one_catalog_identity() {
    assert_eq!(ENTRIES.len(), 21);
    for (index, entry) in ENTRIES.iter().enumerate() {
        assert_eq!(entry.bindings.len(), REQUIRED_DEFAULT_BINDING.len());
        assert!(ENTRIES[..index]
            .iter()
            .all(|previous| previous.identity.name != entry.identity.name));
    }
    assert_eq!(TIMER.documentation.examples.len(), 2);
    assert_eq!(TIMERFIND.documentation.examples.len(), 1);
    assert_eq!(TIMERFINDALL.documentation.examples.len(), 1);
}

#[test]
fn timer_inference_owns_handle_fact_and_rejects_late_invalid_property() {
    let constructor = inference::constructor(&request(Vec::new(), Vec::new()), &TIMER);
    assert!(matches!(
        constructor.outputs[0].kind,
        ValueKindFact::Object(_)
    ));
    let invalid = inference::constructor(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            vec![
                LiteralValue::String("Period".into()),
                LiteralValue::Number(1.0),
                LiteralValue::String("Running".into()),
            ],
        ),
        &TIMER,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMER-READONLY"));
}

#[test]
fn startat_accepts_optional_delay_other_methods_do_not() {
    let receiver = ValueFact::unknown(DynamicReason::RuntimeValue);
    let delay = ValueFact::unknown(DynamicReason::RuntimeValue);
    let call = request(vec![receiver, delay], Vec::new());
    assert!(inference::startat(&call, &STARTAT).diagnostics.is_empty());
    assert!(inference::method(&call, &START)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMER-METHOD-ARITY"));
}
