use super::*;
use runmat_types::*;
#[test]
fn no_result_and_metadata_receiver_admission() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(ClassIdentity::from("matlab.metadata.DynamicProperty")),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(true),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        )],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
    };
    let result = inference::infer(&request, &DYNAMIC_PROPERTY_DELETE_CATALOG_ENTRY);
    assert!(result.outputs.is_empty());
    assert!(result.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    assert!(
        !inference::infer(&request, &DYNAMIC_PROPERTY_DELETE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(
        !inference::infer(&request, &DYNAMIC_PROPERTY_DELETE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(
        inference::infer(&request, &DYNAMIC_PROPERTY_DELETE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
