use super::*;
use runmat_types::*;

#[test]
fn opaque_overload_subscript_is_not_diagnosed_as_a_default_path() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let object = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    for target in [unknown, object, ValueFact::scalar(ValueKindFact::Logical)] {
        let default_route = matches!(target.kind, ValueKindFact::Logical);
        let request = CallRequest {
            arguments: vec![
                target,
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("Expression".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = inference::infer(&request, &NUM_ARGUMENTS_FROM_SUBSCRIPT_CATALOG_ENTRY);
        assert_eq!(!result.diagnostics.is_empty(), default_route);
    }
}

#[test]
fn default_count_is_double_but_unresolved_overload_is_dynamic() {
    let mut request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); 3],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Statement".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert_eq!(
        inference::infer(&request, &NUM_ARGUMENTS_FROM_SUBSCRIPT_CATALOG_ENTRY).outputs[0].kind,
        ValueKindFact::Unknown
    );
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let output =
        &inference::infer(&request, &NUM_ARGUMENTS_FROM_SUBSCRIPT_CATALOG_ENTRY).outputs[0];
    assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    assert_eq!(output.shape, ShapeFact::Scalar);
    request.literals.literal_args[2] = LiteralValue::String("Loop".into());
    assert!(
        !inference::infer(&request, &NUM_ARGUMENTS_FROM_SUBSCRIPT_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
