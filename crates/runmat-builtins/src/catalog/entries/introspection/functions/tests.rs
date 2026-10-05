use super::*;
use runmat_types::*;

#[test]
fn callable_input_has_three_string_metadata_fields() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &FUNCTIONS_CATALOG_ENTRY);
    let ValueKindFact::Struct(info) = &result.outputs[0].kind else {
        panic!("struct metadata")
    };
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(info.fields.len(), 3);
    assert!(info
        .fields
        .values()
        .all(|fact| fact.kind == ValueKindFact::String));
}

#[test]
fn definitely_noncallable_input_is_rejected_statically() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(!inference::infer(&request, &FUNCTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    for residency in [
        ResidencyFact::Device { provider: None },
        ResidencyFact::Remote { pool: None },
        ResidencyFact::Unknown,
    ] {
        request.arguments[0].residency = residency;
        assert!(!inference::infer(&request, &FUNCTIONS_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
