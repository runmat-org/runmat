use super::*;
use runmat_types::*;

fn call(value: ValueFact, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![value],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &ISMISSING_CATALOG_ENTRY,
    )
}

#[test]
fn numeric_and_string_array_masks_retain_known_shape() {
    let numeric = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let inferred = call(numeric, 1);
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3), Some(4)])
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);

    let string_array = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(0), Some(3)]),
        StorageFact::Dense,
    );
    assert_eq!(
        call(string_array, 1).outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(3)])
    );
}

#[test]
fn character_rows_and_unknown_aggregate_stay_sound() {
    let chars = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(5), Some(12)]),
        StorageFact::Dense,
    );
    assert_eq!(
        call(chars, 1).outputs[0].shape,
        ShapeFact::from(vec![Some(5), Some(1)])
    );
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert_eq!(call(unknown, 1).outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn excessive_requested_outputs_are_diagnosed() {
    let value = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!call(value, 2).diagnostics.is_empty());
}
