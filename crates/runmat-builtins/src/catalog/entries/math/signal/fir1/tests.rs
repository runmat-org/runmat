use super::*;
use runmat_types::*;

fn real(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(
    order: LiteralValue,
    cutoff: ValueFact,
    options: &[(ValueFact, LiteralValue)],
) -> CallInference {
    let mut arguments = vec![real(ShapeFact::Scalar), cutoff];
    let mut literals = vec![order, LiteralValue::Unknown];
    for (value, literal) in options {
        arguments.push(value.clone());
        literals.push(literal.clone());
    }
    crate::infer_catalog_call(
        &FIR1_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

#[test]
fn highpass_odd_order_adjusts_width_without_changing_class_or_residency() {
    let result = infer(
        LiteralValue::Number(5.0),
        real(ShapeFact::Scalar),
        &[(
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String("high".into()),
        )],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(7)])
    );
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
}

#[test]
fn unknown_filter_type_does_not_assert_odd_order_width() {
    let result = infer(
        LiteralValue::Number(5.0),
        real(ShapeFact::Scalar),
        &[(
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::Unknown,
        )],
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}

#[test]
fn known_bad_window_and_cutoff_are_diagnosed() {
    let window = real(ShapeFact::from(vec![Some(1), Some(4)]));
    let invalid_window = infer(
        LiteralValue::Number(4.0),
        real(ShapeFact::Scalar),
        &[(window, LiteralValue::Unknown)],
    );
    assert!(invalid_window
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FIR1-WINDOW"));
    let invalid_order = infer(LiteralValue::Number(-1.0), real(ShapeFact::Scalar), &[]);
    assert!(invalid_order
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FIR1-ORDER"));
}

#[test]
fn zero_order_matches_runtime_scalarization() {
    let result = infer(LiteralValue::Number(0.0), real(ShapeFact::Scalar), &[]);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
}

#[test]
fn row_and_column_cutoff_literals_match_runtime_vector_admission() {
    for (shape, literal) in [
        (
            ShapeFact::from(vec![Some(1), Some(2)]),
            LiteralValue::Matrix(vec![vec![
                LiteralValue::Number(0.2),
                LiteralValue::Number(0.4),
            ]]),
        ),
        (
            ShapeFact::from(vec![Some(2), Some(1)]),
            LiteralValue::Matrix(vec![
                vec![LiteralValue::Number(0.2)],
                vec![LiteralValue::Number(0.4)],
            ]),
        ),
    ] {
        let result = crate::infer_catalog_call(
            &FIR1_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![real(ShapeFact::Scalar), real(shape)],
                literals: LiteralContext::new(vec![LiteralValue::Number(4.0), literal]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(5)])
        );
    }
}

#[test]
fn complex_zero_imaginary_cutoff_is_not_rejected_but_known_nonzero_window_is() {
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let accepted = crate::infer_catalog_call(
        &FIR1_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![real(ShapeFact::Scalar), complex.clone()],
            literals: LiteralContext::new(vec![
                LiteralValue::Number(4.0),
                LiteralValue::Complex {
                    real: "0.25".into(),
                    imaginary: "0".into(),
                    class: NumericClass::Double,
                },
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );

    let rejected = infer(
        LiteralValue::Number(4.0),
        real(ShapeFact::Scalar),
        &[(
            ValueFact::proven(
                complex.kind,
                ShapeFact::from(vec![Some(1), Some(5)]),
                StorageFact::Dense,
            ),
            LiteralValue::Vector(vec![LiteralValue::Complex {
                real: "1".into(),
                imaginary: "0.1".into(),
                class: NumericClass::Double,
            }]),
        )],
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FIR1-WINDOW"));
}
