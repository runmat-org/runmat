use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn bound_method_scope_and_output_are_declared_without_exposing_helper() {
    let mut call = request(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        1,
    );
    let inferred = inference::infer(&call, &CALL_BOUND_METHOD_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        CALL_BOUND_METHOD_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::HiddenInternal
    );
    assert_eq!(
        CALL_BOUND_METHOD_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    assert_eq!(
        CALL_BOUND_METHOD_ERRORS[0].identifier,
        Some("RunMat:CallMethodScopeInvalid")
    );

    call.arguments[2] = ValueFact::scalar(ValueKindFact::Character);
    assert!(!inference::infer(&call, &CALL_BOUND_METHOD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.arguments[2] = ValueFact::unknown(DynamicReason::RuntimeValue);
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&call, &CALL_BOUND_METHOD_CATALOG_ENTRY)
        .outputs
        .is_empty());
}
