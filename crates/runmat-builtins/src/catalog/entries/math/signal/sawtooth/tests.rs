use super::*;
use runmat_types::*;

fn real(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &SAWTOOTH_CATALOG_ENTRY,
    )
}

#[test]
fn all_admitted_numeric_sample_classes_produce_double_shape() {
    let shape = ShapeFact::from(vec![Some(2), Some(0), Some(3)]);
    for class in [
        NumericClass::Double,
        NumericClass::Single,
        NumericClass::Int64,
    ] {
        let result = infer(
            vec![real(class, shape.clone())],
            vec![LiteralValue::Unknown],
            1,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn invalid_peak_and_complex_input_cannot_prove_output() {
    let scalar = real(NumericClass::Double, ShapeFact::Scalar);
    for (peak, literal) in [
        (
            real(
                NumericClass::Double,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
            LiteralValue::Unknown,
        ),
        (scalar.clone(), LiteralValue::Number(-0.5)),
    ] {
        let result = infer(
            vec![scalar.clone(), peak],
            vec![LiteralValue::Unknown, literal],
            1,
        );
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let result = infer(vec![complex], vec![LiteralValue::Unknown], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_sample_and_discarded_output_are_conservative() {
    let result = infer(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        1,
    );
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    let result = infer(
        vec![real(NumericClass::Double, ShapeFact::Scalar)],
        vec![LiteralValue::Unknown],
        0,
    );
    assert!(result.outputs.is_empty());
}
