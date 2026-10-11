use super::*;
use runmat_types::*;
fn call(receiver: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![receiver, ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn metadata_handle_fact_and_invalid_receiver() {
    let valid = ValueFact::proven(
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
    let mut request = call(valid, 1);
    let result = inference::infer(&request, &ADDPROP_CATALOG_ENTRY);
    assert_eq!(result.outputs.len(), 1);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("metadata object fact")
    };
    assert_eq!(
        object
            .runtime_class
            .as_ref()
            .map(|class| class.display_name()),
        Some("matlab.metadata.DynamicProperty")
    );
    assert_eq!(object.handle_semantics, Some(true));
    assert!(result.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&request, &ADDPROP_CATALOG_ENTRY)
        .outputs
        .is_empty());
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &ADDPROP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(inference::infer(&request, &ADDPROP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&request, &ADDPROP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
