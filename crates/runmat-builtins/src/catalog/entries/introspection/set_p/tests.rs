use super::*;
use runmat_types::*;

#[test]
fn replacement_updates_member_facts_without_losing_class() {
    let identity = StaticClassIdentity::new("decomposition").owned();
    let receiver = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(identity.clone()),
            properties: Default::default(),
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let payload = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let mut request = CallRequest {
        arguments: vec![receiver, payload.clone()],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &SET_P_CATALOG_ENTRY);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("object");
    };
    assert_eq!(object.runtime_class, Some(identity));
    assert_eq!(object.properties["p_backing"], payload);
    request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert_eq!(
        inference::infer(&request, &SET_P_CATALOG_ENTRY).outputs[0].kind,
        ValueKindFact::Unknown
    );
}
