use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
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
        &HILBERT_CATALOG_ENTRY,
    )
}

#[test]
fn complex_domain_and_precision_follow_the_signal() {
    let shape = ShapeFact::from(vec![Some(1), Some(4)]);
    for domain in [NumericDomain::Real, NumericDomain::Complex] {
        let result = infer(
            vec![numeric(NumericClass::Single, domain, shape.clone())],
            vec![LiteralValue::Unknown],
            1,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(
            result.outputs[0].numeric(),
            Some(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    }
}

#[test]
fn integer_and_logical_inputs_produce_double_complex() {
    let shape = ShapeFact::from(vec![Some(4), Some(2)]);
    for input in [
        numeric(NumericClass::UInt64, NumericDomain::Real, shape.clone()),
        ValueFact::proven(ValueKindFact::Logical, shape.clone(), StorageFact::Dense),
    ] {
        let result = infer(vec![input], vec![LiteralValue::Unknown], 1);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].numeric(),
            Some(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(result.outputs[0].shape, shape);
    }
}

#[test]
fn literal_length_changes_only_the_transform_axis() {
    let signal = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(4), Some(2)]),
    );
    let length = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let result = infer(
        vec![signal.clone(), length.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Number(6.0)],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(6), Some(2)])
    );

    let unknown = infer(
        vec![signal.clone(), length.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        1,
    );
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None, Some(2)])
    );
    let empty = infer(
        vec![signal.clone(), length],
        vec![LiteralValue::Unknown, LiteralValue::Empty],
        1,
    );
    assert_eq!(empty.outputs[0].shape, signal.shape);

    let singleton = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(1)]),
    );
    let singleton_length = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let singleton_result = infer(
        vec![singleton, singleton_length],
        vec![LiteralValue::Unknown, LiteralValue::Number(3.0)],
        1,
    );
    assert_eq!(
        singleton_result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
}

#[test]
fn typed_single_length_and_runtime_epsilon_rounding_are_exactly_projected() {
    let signal = numeric(
        NumericClass::Single,
        NumericDomain::Real,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let single_control = numeric(NumericClass::Single, NumericDomain::Real, ShapeFact::Scalar);
    let rounded_single = infer(
        vec![signal.clone(), single_control],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Real {
                text: "16777217".into(),
                class: NumericClass::Single,
            },
        ],
        1,
    );
    assert!(rounded_single.diagnostics.is_empty());
    assert_eq!(
        rounded_single.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(16_777_216)])
    );

    let double_control = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let within_epsilon = infer(
        vec![signal.clone(), double_control.clone()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0 + f64::EPSILON),
        ],
        1,
    );
    assert!(within_epsilon.diagnostics.is_empty());
    assert_eq!(
        within_epsilon.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    let beyond_epsilon = infer(
        vec![signal, double_control],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(1.0 + 2.0 * f64::EPSILON),
        ],
        1,
    );
    assert!(!beyond_epsilon.diagnostics.is_empty());
    assert_eq!(beyond_epsilon.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn invalid_controls_and_unsupported_inputs_do_not_get_proven_facts() {
    let signal = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let control = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    for literal in [LiteralValue::Number(-1.0), LiteralValue::Number(1.5)] {
        let result = infer(
            vec![signal.clone(), control.clone()],
            vec![LiteralValue::Unknown, literal],
            1,
        );
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
    let bad = numeric(
        NumericClass::UInt64,
        NumericDomain::Complex,
        ShapeFact::Scalar,
    );
    let result = infer(vec![bad], vec![LiteralValue::Unknown], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn discarded_and_surplus_outputs_do_not_invent_analytic_signals() {
    let signal = numeric(NumericClass::Double, NumericDomain::Real, ShapeFact::Scalar);
    let discarded = infer(vec![signal.clone()], vec![LiteralValue::Unknown], 0);
    assert!(discarded.outputs.is_empty());
    let surplus = infer(vec![signal], vec![LiteralValue::Unknown], 2);
    assert_eq!(surplus.outputs.len(), 2);
    assert_eq!(surplus.outputs[1].kind, ValueKindFact::Unknown);
}
