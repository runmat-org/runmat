use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &FILLMISSING_CATALOG_ENTRY,
    )
}

#[test]
fn known_numeric_input_preserves_class_and_shape_with_logical_mask() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let result = call(
        vec![input.clone(), ValueFact::scalar(ValueKindFact::String)],
        2,
    );
    assert_eq!(result.outputs[0].shape, input.shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[1].kind, ValueKindFact::Logical);
    assert_eq!(result.outputs[1].shape, input.shape);
}

#[test]
fn unknown_aggregate_and_excessive_outputs_are_not_claimed_proven() {
    let result = call(
        vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::String),
        ],
        3,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[1].kind, ValueKindFact::Unknown);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
