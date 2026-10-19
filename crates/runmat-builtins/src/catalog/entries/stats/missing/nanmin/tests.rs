use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &NANMIN_CATALOG_ENTRY,
    )
}
#[test]
fn known_single_input_has_single_numeric_output_and_reduced_shape() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = call(input, 1);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
}
#[test]
fn all_selector_proves_scalar_output_shape() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![input, ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMIN_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
}
#[test]
fn known_nonnumeric_input_has_a_type_diagnostic() {
    assert!(call(ValueFact::scalar(ValueKindFact::String), 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM.MISSING.UNSUPPORTED_TYPE"));
}
#[test]
fn unknown_input_and_excess_outputs_do_not_claim_success() {
    let inferred = call(ValueFact::unknown(DynamicReason::RuntimeValue), 3);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn empty_placeholder_and_dimension_are_reduction_not_pairwise() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![
                input,
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![Some(0), Some(0)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Empty,
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMIN_CATALOG_ENTRY,
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn pairwise_same_class_integer_input_keeps_exact_class_and_shape() {
    let operand = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![operand.clone(), operand.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMIN_CATALOG_ENTRY,
    );
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(inferred.outputs[0].shape, operand.shape);
}

#[test]
fn reduction_indices_are_double_with_reduced_shape() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = call(input, 2);
    assert_eq!(inferred.outputs.len(), 2);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        inferred.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[1].shape, inferred.outputs[0].shape);
}

#[test]
fn all_selector_and_dimension_do_not_prove_a_reduction() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![
                input,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
        &NANMIN_CATALOG_ENTRY,
    );
    assert!(inferred.outputs.iter().all(|output| {
        output.kind == ValueKindFact::Unknown && output.shape == ShapeFact::Unknown
    }));
}

#[test]
fn resident_or_complex_second_argument_does_not_claim_pairwise_min() {
    let operand = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let mut resident = operand.clone();
    resident.residency = ResidencyFact::Device { provider: None };
    let mut complex = operand.clone();
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Complex,
    });
    for second in [resident, complex] {
        let inferred = inference::infer(
            &CallRequest {
                arguments: vec![operand.clone(), second],
                literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
            &NANMIN_CATALOG_ENTRY,
        );
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    }
}
