use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &NANMEDIAN_CATALOG_ENTRY,
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
        &NANMEDIAN_CATALOG_ENTRY,
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
