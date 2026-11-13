use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &RMMISSING_CATALOG_ENTRY,
    )
}

#[test]
fn floating_result_keeps_class_but_does_not_claim_a_removed_extent() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = call(input, 2);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(inferred.outputs[1].kind, ValueKindFact::Logical);
    assert_eq!(inferred.outputs[1].shape, ShapeFact::Unknown);
}

#[test]
fn symbolic_rank_is_preserved_without_materializing_dimensions() {
    let mut input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    input.shape = ShapeFact::Ranked { rank: usize::MAX };
    assert_eq!(call(input.clone(), 1).outputs[0].shape, input.shape);
}

#[test]
fn integer_input_cannot_lose_rows_to_standard_missing_values() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert_eq!(call(input.clone(), 1).outputs[0].shape, input.shape);
    assert!(call(input, 3)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
