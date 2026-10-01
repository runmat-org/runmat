use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &CORRCOV_CATALOG_ENTRY,
    )
}

fn matrix(class: NumericClass, rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

#[test]
fn class_shapes_and_requested_outputs_follow_corrcov_contract() {
    let result = call(matrix(NumericClass::Single, 2, 2), 2);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        call(matrix(NumericClass::UInt64, 2, 2), 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Double
    );
    assert!(!call(matrix(NumericClass::Double, 2, 3), 1)
        .diagnostics
        .is_empty());
    assert!(!call(matrix(NumericClass::Double, 2, 2), 3)
        .diagnostics
        .is_empty());
}

#[test]
fn resident_provider_choice_is_not_claimed_statically() {
    let mut input = matrix(NumericClass::Double, 2, 2);
    input.residency = ResidencyFact::Device { provider: None };
    let result = call(input, 2);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[1].residency, ResidencyFact::Unknown);
}
