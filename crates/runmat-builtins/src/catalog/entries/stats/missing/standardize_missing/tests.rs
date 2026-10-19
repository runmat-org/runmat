use super::*;
use runmat_types::*;

fn call(data: ValueFact, indicator: ValueFact) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![data, indicator],
            literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &STANDARDIZE_MISSING_CATALOG_ENTRY,
    )
}

#[test]
fn numeric_class_and_shape_survive_host_standardization() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let indicator = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let result = call(data, indicator);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn character_rows_change_to_strings_but_unknown_values_stay_unknown() {
    let characters = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(4), Some(8)]),
        StorageFact::Dense,
    );
    let marker = ValueFact::scalar(ValueKindFact::String);
    let result = call(characters, marker.clone());
    assert_eq!(result.outputs[0].kind, ValueKindFact::String);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );

    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert_eq!(
        call(unknown, marker).outputs[0].kind,
        ValueKindFact::Unknown
    );
}
