use super::*;
use runmat_types::*;

fn request(value: ValueFact, outputs: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments: vec![value],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn scalar_logical_fact_for_known_and_unknown_inputs() {
    for value in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    ] {
        let inferred = inference::infer(
            &request(value, RequestedOutputCount::Exactly(1)),
            &ANYMISSING_CATALOG_ENTRY,
        );
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(inferred.outputs[0].shape.element_count(), Some(1));
        assert!(inferred.diagnostics.is_empty());
    }
}

#[test]
fn rejects_too_many_outputs_and_missing_input() {
    let value = ValueFact::scalar(ValueKindFact::String);
    assert!(!inference::infer(
        &request(value, RequestedOutputCount::Exactly(2)),
        &ANYMISSING_CATALOG_ENTRY,
    )
    .diagnostics
    .is_empty());
    let mut missing_input = request(
        ValueFact::unknown(DynamicReason::RuntimeValue),
        RequestedOutputCount::Exactly(1),
    );
    missing_input.arguments.clear();
    assert!(!inference::infer(&missing_input, &ANYMISSING_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
