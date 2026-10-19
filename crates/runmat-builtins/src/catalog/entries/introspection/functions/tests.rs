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

#[test]
fn known_callable_on_device_or_remote_is_rejected_without_rejecting_unknown_residency() {
    let mut callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: false,
    }));
    let mut request = CallRequest {
        arguments: vec![callable.clone()],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    for residency in [
        ResidencyFact::Device { provider: None },
        ResidencyFact::Remote { pool: None },
    ] {
        callable.residency = residency;
        request.arguments[0] = callable.clone();
        assert!(inference::infer(&request, &FUNCTIONS_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-FUNCTIONS-RESIDENCY"));
    }
    callable.residency = ResidencyFact::Unknown;
    request.arguments[0] = callable;
    assert!(inference::infer(&request, &FUNCTIONS_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
