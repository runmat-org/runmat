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
        &TRIPULS_CATALOG_ENTRY,
    )
}

#[test]
fn waveform_class_and_shape_follow_sample_times() {
    for (class, expected) in [
        (NumericClass::Single, NumericClass::Single),
        (NumericClass::Double, NumericClass::Double),
        (NumericClass::Int64, NumericClass::Double),
    ] {
        let times = real(class, ShapeFact::from(vec![Some(2), Some(0), Some(3)]));
        let out = infer(vec![times.clone()], vec![LiteralValue::Unknown], 1);
        assert!(out.diagnostics.is_empty());
        assert_eq!(out.outputs[0].shape, times.shape);
        assert_eq!(out.outputs[0].numeric().unwrap().class, expected);
        assert_eq!(out.outputs[0].residency, ResidencyFact::Unknown);
    }
}

#[test]
fn invalid_controls_and_complex_times_have_no_success_facts() {
    let scalar = real(NumericClass::Double, ShapeFact::Scalar);
    let vector = real(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    for (arguments, literals) in [
        (vec![scalar.clone(), vector], vec![LiteralValue::Unknown; 2]),
        (
            vec![scalar.clone(), scalar.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
        ),
        (
            vec![scalar.clone(), scalar.clone(), scalar.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
            ],
        ),
    ] {
        let out = infer(arguments, literals, 1);
        assert!(!out.diagnostics.is_empty());
        assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    }
    let complex = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    let out = infer(vec![complex], vec![LiteralValue::Unknown], 1);
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn unknown_sample_and_discarded_output_do_not_claim_value() {
    let out = infer(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        1,
    );
    assert_eq!(out.outputs[0].kind, ValueKindFact::Unknown);
    let out = infer(
        vec![real(NumericClass::Single, ShapeFact::Scalar)],
        vec![LiteralValue::Unknown],
        0,
    );
    assert!(out.outputs.is_empty());
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
fn partial_nonscalar_controls_and_inexact_integer_literals_do_not_prove_success() {
    let scalar = real(NumericClass::Double, ShapeFact::Scalar);
    for extent in [0, 2] {
        let control = real(
            NumericClass::Double,
            ShapeFact::from(vec![Some(extent), None]),
        );
        for arguments in [
            vec![scalar.clone(), control.clone()],
            vec![scalar.clone(), scalar.clone(), control],
        ] {
            let result = infer(arguments, vec![LiteralValue::Unknown; 3], 1);
            assert!(!result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        }
    }
    let inexact = LiteralValue::Integer {
        text: "9007199254740993".into(),
        class: NumericClass::UInt64,
    };
    for (arguments, literals) in [
        (
            vec![real(NumericClass::UInt64, ShapeFact::Scalar)],
            vec![inexact.clone()],
        ),
        (
            vec![
                scalar.clone(),
                real(NumericClass::UInt64, ShapeFact::Scalar),
            ],
            vec![LiteralValue::Unknown, inexact],
        ),
    ] {
        let result = infer(arguments, literals, 1);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}
