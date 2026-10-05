use super::*;
use runmat_types::*;

#[test]
fn caller_dependent_name_is_character_and_non_numeric_is_rejected() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &INPUTNAME_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &INPUTNAME_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    for residency in [
        ResidencyFact::Device { provider: None },
        ResidencyFact::Remote { pool: None },
        ResidencyFact::Unknown,
    ] {
        request.arguments[0].residency = residency;
        assert!(!inference::infer(&request, &INPUTNAME_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    request.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(1)]);
    assert!(!inference::infer(&request, &INPUTNAME_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.arguments[0].shape = ShapeFact::Unknown;
    assert!(inference::infer(&request, &INPUTNAME_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
