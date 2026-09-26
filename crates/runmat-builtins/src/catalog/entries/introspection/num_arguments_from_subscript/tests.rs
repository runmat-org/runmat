use super::*;
use runmat_types::*;

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
