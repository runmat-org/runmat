use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &COV2CORR_CATALOG_ENTRY,
    )
}

#[test]
fn standard_deviations_are_first_and_form_a_row() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(3)]),
        StorageFact::Dense,
    );
    let one = call(input.clone(), 1);
    assert!(one.diagnostics.is_empty(), "{:?}", one.diagnostics);
    assert_eq!(
        one.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    let two = call(input, 2);
    assert_eq!(
        two.outputs[1].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
}

#[test]
fn unknown_class_does_not_claim_a_proven_double_result() {
    let mut input = ValueFact::unknown(DynamicReason::RuntimeValue);
    input.shape = ShapeFact::from(vec![Some(2), Some(2)]);
    let result = call(input, 2);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
