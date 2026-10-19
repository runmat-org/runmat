use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
fn handle() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
#[test]
fn known_unknown_invalid_and_discarded_registration() {
    let mut call = request(
        vec![
            handle(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        1,
    );
    let result = inference::infer(&call, &ADDLISTENER_CATALOG_ENTRY);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert!(result.diagnostics.is_empty());
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&call, &ADDLISTENER_CATALOG_ENTRY)
        .outputs
        .is_empty());
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&call, &ADDLISTENER_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(inference::infer(&call, &ADDLISTENER_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&call, &ADDLISTENER_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
