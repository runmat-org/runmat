use super::*;
use runmat_types::*;

fn call(input: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![
                input,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(3.0)]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &MOVMAD_CATALOG_ENTRY,
    )
}

#[test]
fn integer_input_produces_double_with_same_shape() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5), Some(1)]),
        StorageFact::Dense,
    );
    let inferred = call(input.clone(), 1);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(inferred.outputs[0].shape, input.shape);
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn unknown_data_and_excessive_outputs_stay_unproven() {
    let inferred = call(ValueFact::unknown(DynamicReason::RuntimeValue), 2);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}
