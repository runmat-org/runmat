use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(argument: Option<(ValueFact, LiteralValue)>) -> CallRequest {
    let (arguments, literals) = match argument {
        Some((fact, literal)) => (vec![fact], vec![literal]),
        None => (Vec::new(), Vec::new()),
    };
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn pause_inference_distinguishes_state_and_empty_wait() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let query = inference::infer(
        &request(Some((text.clone(), LiteralValue::String("query".into())))),
        &PAUSE_CATALOG_ENTRY,
    );
    assert!(matches!(query.outputs[0].kind, ValueKindFact::Character));
    let duration = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let waited = inference::infer(
        &request(Some((duration, LiteralValue::Unknown))),
        &PAUSE_CATALOG_ENTRY,
    );
    assert_eq!(waited.outputs[0].shape.element_count(), Some(0));
    let invalid = inference::infer(
        &request(Some((text, LiteralValue::String("bad".into())))),
        &PAUSE_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-PAUSE-COMMAND"));

    let real = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let negative = inference::infer(
        &request(Some((real, LiteralValue::Number(-1.0)))),
        &PAUSE_CATALOG_ENTRY,
    );
    assert!(negative
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-PAUSE-DURATION"));
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    assert!(inference::infer(
        &request(Some((complex, LiteralValue::Unknown))),
        &PAUSE_CATALOG_ENTRY,
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-PAUSE-DURATION"));
}
