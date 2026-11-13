use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &NANMEAN_CATALOG_ENTRY,
    )
}

#[test]
fn abstract_rank_and_trailing_dimension_do_not_allocate_shape_metadata() {
    let mut input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    input.shape = ShapeFact::Ranked { rank: usize::MAX };
    assert_eq!(call(input.clone(), 1).outputs[0].shape, input.shape);
    input.shape = ShapeFact::from(vec![Some(3), Some(2)]);
    let inferred = inference::infer(
        &CallRequest {
            arguments: vec![
                input.clone(),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Integer {
                    text: usize::MAX.to_string(),
                    class: NumericClass::UInt64,
                },
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMEAN_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].shape, input.shape);
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
        &NANMEAN_CATALOG_ENTRY,
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
    let inferred = call(ValueFact::unknown(DynamicReason::RuntimeValue), 2);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn integer_output_class_obeys_native_and_default_options() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    assert_eq!(
        call(input.clone(), 1).outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let native = inference::infer(
        &CallRequest {
            arguments: vec![input.clone(), ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("native".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMEAN_CATALOG_ENTRY,
    );
    assert_eq!(
        native.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    let like = inference::infer(
        &CallRequest {
            arguments: vec![input, ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMEAN_CATALOG_ENTRY,
    );
    assert_eq!(like.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn conflicting_native_and_double_options_do_not_prove_an_output_class() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
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
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("native".into()),
                LiteralValue::String("double".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMEAN_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn duplicate_all_selector_does_not_prove_scalar_shape() {
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
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
                LiteralValue::String("all".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &NANMEAN_CATALOG_ENTRY,
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
}
