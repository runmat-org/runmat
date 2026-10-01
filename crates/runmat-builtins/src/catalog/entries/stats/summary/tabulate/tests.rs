use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn infer(input: ValueFact) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &TABULATE_CATALOG_ENTRY,
    )
}

#[test]
fn numeric_class_and_output_columns_follow_runtime_without_guessing_levels() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let result = infer(input);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        result.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    );
}

#[test]
fn wide_integer_and_unknown_contents_do_not_claim_numeric_label_conversion() {
    let integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(integer);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    assert_eq!(
        infer(ValueFact::unknown(DynamicReason::RuntimeValue)).outputs[0].kind,
        ValueKindFact::Unknown
    );
}

#[test]
fn invalid_input_does_not_prove_a_table() {
    let result = infer(ValueFact::scalar(ValueKindFact::Symbolic));
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}
