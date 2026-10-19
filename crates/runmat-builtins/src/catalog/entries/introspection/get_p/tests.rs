use super::*;
use runmat_types::*;

#[test]
fn known_backing_member_preserves_nested_class_and_residency() {
    let mut backing = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    backing.residency = ResidencyFact::Device {
        provider: Some("test".into()),
    };
    backing.alias = AliasFact::Unknown;
    let object = ObjectFact {
        class: None,
        runtime_class: None,
        properties: [("p_backing".into(), backing.clone())].into(),
        properties_complete: true,
        handle_semantics: Some(false),
    };
    let mut request = CallRequest {
        arguments: vec![ValueFact::proven(
            ValueKindFact::Object(object),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        )],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert_eq!(
        inference::infer(&request, &GET_P_CATALOG_ENTRY).outputs[0],
        backing
    );
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert_eq!(
        inference::infer(&request, &GET_P_CATALOG_ENTRY).outputs[0].kind,
        ValueKindFact::Unknown
    );
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(!inference::infer(&request, &GET_P_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&request, &GET_P_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
