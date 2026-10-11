use super::*;
use runmat_types::*;
#[test]
fn unknown_property_class_and_known_invalid_target() {
    let mut request = CallRequest {
        arguments: vec![
            ValueFact::proven(
                ValueKindFact::Object(ObjectFact {
                    class: None,
                    runtime_class: None,
                    properties: Default::default(),
                    properties_complete: false,
                    handle_semantics: Some(false),
                }),
                ShapeFact::Scalar,
                StorageFact::Opaque,
            ),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &FINDPROP_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert!(result.diagnostics.is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&request, &FINDPROP_CATALOG_ENTRY)
        .outputs
        .is_empty());
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &FINDPROP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(inference::infer(&request, &FINDPROP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
