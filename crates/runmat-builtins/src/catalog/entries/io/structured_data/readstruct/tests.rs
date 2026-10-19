use super::*;
use runmat_types::*;

#[test]
fn readstruct_inference_preserves_scalar_struct_and_rejects_known_bad_controls() {
    let valid = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&valid, &READSTRUCT_CATALOG_ENTRY);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Struct(_)));
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let invalid = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
        ..valid
    };
    assert!(!inference::infer(&invalid, &READSTRUCT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
