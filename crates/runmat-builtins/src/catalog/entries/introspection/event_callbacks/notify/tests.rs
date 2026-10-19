use super::*;
use runmat_types::*;
#[test]
fn status_callback_effects_and_output_limit() {
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &NOTIFY_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
    assert!(result.effects.0.contains(&EffectKind::HostCallback));
    assert!(result.effects.0.contains(&EffectKind::Unknown));
    assert!(result.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&request, &NOTIFY_CATALOG_ENTRY)
        .outputs
        .is_empty());
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &NOTIFY_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&request, &NOTIFY_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
