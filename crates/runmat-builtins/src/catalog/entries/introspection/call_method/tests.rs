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
fn method_result_is_dynamic_and_helper_remains_hidden() {
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut call = request(vec![object, ValueFact::scalar(ValueKindFact::String)], 2);
    let inferred = inference::infer(&call, &CALL_METHOD_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 2);
    assert!(inferred
        .outputs
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));
    assert_eq!(
        CALL_METHOD_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::HiddenInternal
    );
    assert_eq!(
        CALL_METHOD_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    assert_eq!(
        CALL_METHOD_CATALOG_ENTRY.link.execution_stack,
        ExecutionStackRequirement::Process
    );

    call.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&call, &CALL_METHOD_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&call, &CALL_METHOD_CATALOG_ENTRY)
        .outputs
        .is_empty());
}
