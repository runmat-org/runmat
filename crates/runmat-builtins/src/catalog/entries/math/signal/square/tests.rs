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
        &SQUARE_CATALOG_ENTRY,
    )
}

#[test]
fn real_single_shape_and_class_are_retained_but_integer_produces_double() {
    let shape = ShapeFact::from(vec![Some(2), Some(0), Some(3)]);
    for (class, expected) in [
        (NumericClass::Single, NumericClass::Single),
        (NumericClass::Int64, NumericClass::Double),
    ] {
        let result = infer(
            vec![real(class, shape.clone())],
            vec![LiteralValue::Unknown],
            1,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(result.outputs[0].numeric().unwrap().class, expected);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn invalid_duty_and_complex_input_cannot_prove_output() {
    let scalar = real(NumericClass::Double, ShapeFact::Scalar);
    for (duty, literal) in [
        (
            real(
                NumericClass::Double,
                ShapeFact::from(vec![Some(1), Some(2)]),
            ),
            LiteralValue::Unknown,
        ),
        (scalar.clone(), LiteralValue::Number(101.0)),
    ] {
        let result = infer(
            vec![scalar.clone(), duty],
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

#[test]
fn known_sparse_sample_times_are_diagnosed() {
    let mut value = real(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(2)]),
    );
    value.storage = StorageFact::Sparse;
    let result = infer(vec![value], vec![LiteralValue::Unknown], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn partial_nonscalar_duty_and_inexact_integer_times_do_not_prove_success() {
    let scalar = real(NumericClass::Double, ShapeFact::Scalar);
    for extent in [0, 2] {
        let duty = real(
            NumericClass::Double,
            ShapeFact::from(vec![Some(extent), None]),
        );
        let result = infer(
            vec![scalar.clone(), duty],
            vec![LiteralValue::Unknown; 2],
            1,
        );
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
    let result = infer(
        vec![real(NumericClass::UInt64, ShapeFact::Scalar)],
        vec![LiteralValue::Integer {
            text: "9007199254740993".into(),
            class: NumericClass::UInt64,
        }],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
