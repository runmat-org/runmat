use super::*;
use runmat_types::*;

fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
fn known(dims: &[usize]) -> ShapeFact {
    ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>())
}
fn call(
    data: ShapeFact,
    requests: ShapeFact,
    control: Option<(ValueFact, LiteralValue)>,
) -> CallInference {
    let mut arguments = vec![numeric(data), numeric(requests)];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    if let Some((fact, literal)) = control {
        arguments.push(fact);
        literals.push(literal);
    }
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &QUANTILE_CATALOG_ENTRY,
    )
}

#[test]
fn vector_request_changes_only_reduced_axis_and_returns_double() {
    let result = call(known(&[4, 3]), known(&[1, 2]), None);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, known(&[2, 3]));
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        call(
            known(&[4, 3]),
            known(&[1, 2]),
            Some((numeric(ShapeFact::Scalar), LiteralValue::Number(2.0)))
        )
        .outputs[0]
            .shape,
        known(&[4, 2])
    );
}

#[test]
fn unknown_request_length_and_runtime_dimension_do_not_invent_output_shape() {
    assert_eq!(
        call(known(&[4, 3]), ShapeFact::Ranked { rank: 2 }, None).outputs[0].shape,
        ShapeFact::Ranked { rank: 2 }
    );
    assert_eq!(
        call(
            known(&[4, 3]),
            known(&[1, 2]),
            Some((numeric(ShapeFact::Scalar), LiteralValue::Unknown))
        )
        .outputs[0]
            .shape,
        ShapeFact::Unknown
    );
    assert_eq!(
        call(ShapeFact::from(vec![None, Some(4)]), known(&[1, 2]), None).outputs[0].shape,
        ShapeFact::Unknown
    );
}

#[test]
fn all_empty_and_invalid_controls_are_distinguished() {
    let all = call(
        known(&[2, 3]),
        known(&[1, 2]),
        Some((
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String("all".into()),
        )),
    );
    assert_eq!(all.outputs[0].shape, known(&[2, 1]));
    assert_eq!(
        call(known(&[0, 3]), known(&[1, 2]), None).outputs[0].shape,
        known(&[2, 1])
    );
    let bad = call(
        known(&[2, 3]),
        known(&[1, 2]),
        Some((numeric(ShapeFact::Scalar), LiteralValue::Number(0.0))),
    );
    assert!(!bad.diagnostics.is_empty());
    let bad = call(
        known(&[2, 3]),
        known(&[1, 2]),
        Some((
            ValueFact::scalar(ValueKindFact::String),
            LiteralValue::String("unexpected".into()),
        )),
    );
    assert!(!bad.diagnostics.is_empty());
}

#[test]
fn count_and_vecdim_literals_are_shape_proven_without_runtime_string_matching() {
    let data = numeric(known(&[2, 2, 2]));
    let count = numeric(ShapeFact::Scalar);
    let axes = numeric(known(&[1, 2]));
    let request = CallRequest {
        arguments: vec![data, count, axes],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(4.0),
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)]),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &QUANTILE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, known(&[4, 1, 2]));
}

#[test]
fn wide_integer_dimensions_remain_exact_without_materializing_a_huge_shape() {
    let wide = (1_u64 << 53) + 1;
    let result = call(
        known(&[2, 3]),
        ShapeFact::Scalar,
        Some((
            numeric(ShapeFact::Scalar),
            LiteralValue::Integer {
                text: wide.to_string(),
                class: NumericClass::UInt64,
            },
        )),
    );
    if usize::BITS >= 64 {
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::Ranked {
                rank: wide as usize
            }
        );
    } else {
        assert!(!result.diagnostics.is_empty());
    }

    let invalid = call(
        known(&[2, 3]),
        ShapeFact::Scalar,
        Some((
            numeric(ShapeFact::Scalar),
            LiteralValue::Number(usize::MAX as f64),
        )),
    );
    assert!(!invalid.diagnostics.is_empty());
    // A nonliteral scalar request can select either one probability or a
    // count; retain rank, but do not invent the selected extent.
    assert_eq!(invalid.outputs[0].shape, ShapeFact::Ranked { rank: 2 });
}

#[test]
fn empty_and_duplicate_vector_dimensions_are_rejected_statically() {
    for axes in [
        vec![],
        vec![LiteralValue::Number(1.0), LiteralValue::Number(1.0)],
    ] {
        let result = call(
            known(&[2, 3]),
            ShapeFact::Scalar,
            Some((
                numeric(ShapeFact::Ranked { rank: 2 }),
                LiteralValue::Vector(axes),
            )),
        );
        assert!(!result.diagnostics.is_empty());
    }
}
