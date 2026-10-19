use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
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
        &ENVELOPE_CATALOG_ENTRY,
    )
}

#[test]
fn exact_two_output_shape_class_without_false_residency() {
    let shape = ShapeFact::from(vec![Some(4), Some(2)]);
    for (class, expected) in [
        (NumericClass::Single, NumericClass::Single),
        (NumericClass::Double, NumericClass::Double),
        (NumericClass::Int16, NumericClass::Double),
    ] {
        let result = infer(
            vec![numeric(class, shape.clone())],
            vec![LiteralValue::Unknown],
            2,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 2);
        for output in &result.outputs {
            assert_eq!(output.shape, shape);
            assert_eq!(
                output.numeric(),
                Some(NumericFact {
                    class: expected,
                    domain: NumericDomain::Real
                })
            );
            assert_eq!(output.residency, ResidencyFact::Unknown);
        }
    }
}

#[test]
fn known_invalid_controls_and_shapes_cannot_claim_outputs() {
    let signal = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    for (arguments, literals, outputs) in [
        (
            vec![signal.clone(), scalar.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
            1,
        ),
        (
            vec![
                signal.clone(),
                scalar.clone(),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Number(3.0),
                LiteralValue::String("other".into()),
            ],
            2,
        ),
        (vec![signal.clone()], vec![LiteralValue::Unknown], 3),
    ] {
        let result = infer(arguments, literals, outputs);
        assert!(!result.diagnostics.is_empty());
        assert!(result
            .outputs
            .iter()
            .all(|output| output.kind == ValueKindFact::Unknown));
    }
    let rank_three = infer(
        vec![numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(2), Some(2), Some(2)]),
        )],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(!rank_three.diagnostics.is_empty());
    let symbolic_rank_three = infer(
        vec![numeric(NumericClass::Double, ShapeFact::Ranked { rank: 3 })],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(!symbolic_rank_three.diagnostics.is_empty());
    assert!(symbolic_rank_three
        .outputs
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));
    let unknown_rank = infer(
        vec![numeric(NumericClass::Double, ShapeFact::Unknown)],
        vec![LiteralValue::Unknown],
        1,
    );
    assert!(unknown_rank.diagnostics.is_empty());
    assert!(unknown_rank.outputs[0].shape == ShapeFact::Unknown);
}

#[test]
fn typed_single_length_rounding_matches_runtime_tolerance() {
    let signal = numeric(
        NumericClass::Single,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let control = numeric(NumericClass::Single, ShapeFact::Scalar);
    let rounded = infer(
        vec![signal.clone(), control.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Real {
                text: "16777217".into(),
                class: NumericClass::Single,
            },
        ],
        1,
    );
    assert!(rounded.diagnostics.is_empty());
    let near_integer = infer(
        vec![signal.clone(), control.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Number(3.0 + 5e-10)],
        1,
    );
    assert!(near_integer.diagnostics.is_empty());
    let too_far = infer(
        vec![signal, control],
        vec![LiteralValue::Unknown, LiteralValue::Number(3.0 + 2e-9)],
        1,
    );
    assert!(!too_far.diagnostics.is_empty());

    let wide_data = infer(
        vec![numeric(NumericClass::UInt64, ShapeFact::Scalar)],
        vec![LiteralValue::Integer {
            text: "9007199254740993".into(),
            class: NumericClass::UInt64,
        }],
        1,
    );
    assert!(!wide_data.diagnostics.is_empty());
}
