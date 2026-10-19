use super::*;
use runmat_types::*;
fn call(input: ValueFact, literal: LiteralValue, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![literal]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &GAUSPULS_CATALOG_ENTRY,
    )
}
fn number(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
#[test]
fn single_nd_samples_preserve_class_shape_but_materialize_host() {
    let mut input = number(
        NumericClass::Single,
        ShapeFact::from(vec![Some(2), Some(0), Some(3)]),
    );
    input.residency = ResidencyFact::Device {
        provider: Some("owner".into()),
    };
    let out = call(input.clone(), LiteralValue::Unknown, 3);
    assert!(out.diagnostics.is_empty());
    for output in out.outputs {
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.numeric().unwrap().class, NumericClass::Single);
        assert_eq!(output.residency, ResidencyFact::Host);
    }
}
#[test]
fn cutoff_and_padding_are_double_scalars() {
    let out = call(
        ValueFact::scalar(ValueKindFact::Character),
        LiteralValue::Character("cutoff".into()),
        2,
    );
    assert!(out.diagnostics.is_empty());
    assert!(out
        .outputs
        .iter()
        .all(|o| o.is_scalar() && o.numeric().unwrap().class == NumericClass::Double));
}
#[test]
fn unknown_or_dynamic_mode_does_not_invent_sample_shape() {
    let out = call(
        ValueFact::unknown(DynamicReason::RuntimeValue),
        LiteralValue::Unknown,
        1,
    );
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    let out = call(
        ValueFact::scalar(ValueKindFact::String),
        LiteralValue::Unknown,
        1,
    );
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
}
#[test]
fn complex_sample_input_is_diagnosed() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    assert!(!call(input, LiteralValue::Unknown, 1).diagnostics.is_empty());
}
#[test]
fn integer_samples_convert_to_double_and_discarded_output_remains_empty() {
    let input = number(
        NumericClass::UInt64,
        ShapeFact::from(vec![Some(2), Some(1)]),
    );
    assert_eq!(
        call(input.clone(), LiteralValue::Unknown, 1).outputs[0]
            .numeric()
            .unwrap()
            .class,
        NumericClass::Double
    );
    assert!(call(input, LiteralValue::Unknown, 0).outputs.is_empty());
}
