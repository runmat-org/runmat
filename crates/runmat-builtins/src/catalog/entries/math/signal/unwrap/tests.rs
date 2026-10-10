use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &UNWRAP_CATALOG_ENTRY,
    )
}

#[test]
fn exact_shape_class_and_host_placement_follow_phase_input() {
    let shape = ShapeFact::from(vec![Some(2), Some(3)]);
    for (kind, class) in [
        (
            numeric(NumericClass::Single, NumericDomain::Real, shape.clone()),
            NumericClass::Single,
        ),
        (
            numeric(NumericClass::Double, NumericDomain::Real, shape.clone()),
            NumericClass::Double,
        ),
        (
            numeric(NumericClass::Int64, NumericDomain::Real, shape.clone()),
            NumericClass::Double,
        ),
        (
            ValueFact::proven(ValueKindFact::Logical, shape.clone(), StorageFact::Dense),
            NumericClass::Double,
        ),
    ] {
        let result = infer(vec![kind], vec![LiteralValue::Unknown]);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(
            result.outputs[0].numeric(),
            Some(NumericFact {
                class,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
}

#[test]
fn invalid_known_controls_do_not_claim_proven_output() {
    let input = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let scalar = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    for (literals, arguments) in [
        (
            vec![LiteralValue::Unknown, LiteralValue::Number(-1.0)],
            vec![input.clone(), scalar.clone()],
        ),
        (
            vec![
                LiteralValue::Unknown,
                LiteralValue::Empty,
                LiteralValue::Number(0.0),
            ],
            vec![input.clone(), scalar.clone(), scalar.clone()],
        ),
        (
            vec![
                LiteralValue::Unknown,
                LiteralValue::Empty,
                LiteralValue::Number(1.5),
            ],
            vec![input.clone(), scalar.clone(), scalar.clone()],
        ),
    ] {
        let result = infer(arguments, literals);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
    let complex = infer(
        vec![numeric(
            NumericClass::Double,
            NumericDomain::Complex,
            ShapeFact::Scalar,
        )],
        vec![LiteralValue::Unknown],
    );
    assert!(!complex.diagnostics.is_empty());
}

#[test]
fn typed_rounding_and_dimension_epsilon_match_runtime_control_admission() {
    let signal = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let scalar = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let within = infer(
        vec![signal.clone(), scalar.clone(), scalar.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
            LiteralValue::Number(2.0 + 5e-7),
        ],
    );
    assert!(within.diagnostics.is_empty());
    let beyond = infer(
        vec![signal.clone(), scalar.clone(), scalar.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0),
            LiteralValue::Number(2.0 + 2e-6),
        ],
    );
    assert!(!beyond.diagnostics.is_empty());

    let exact = infer(
        vec![signal.clone(), scalar.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "9007199254740992".into(),
                class: NumericClass::UInt64,
            },
        ],
    );
    assert!(exact.diagnostics.is_empty());
    let inexact = infer(
        vec![signal, scalar],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "9007199254740993".into(),
                class: NumericClass::UInt64,
            },
        ],
    );
    assert!(!inexact.diagnostics.is_empty());

    let wide_phase = infer(
        vec![numeric(
            NumericClass::UInt64,
            NumericDomain::Real,
            ShapeFact::Scalar,
        )],
        vec![LiteralValue::Integer {
            text: "9007199254740993".into(),
            class: NumericClass::UInt64,
        }],
    );
    assert!(!wide_phase.diagnostics.is_empty());
}
