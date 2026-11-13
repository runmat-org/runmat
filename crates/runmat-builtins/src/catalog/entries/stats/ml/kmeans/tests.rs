use super::*;
use runmat_types::*;

fn numeric(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    let requested = match outputs {
        0 => RequestedOutputCount::Zero,
        1 => RequestedOutputCount::One,
        2 => RequestedOutputCount::Exactly(2),
        3 => RequestedOutputCount::Exactly(3),
        4 => RequestedOutputCount::Exactly(4),
        _ => RequestedOutputCount::Exactly(outputs),
    };
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(requested),
        },
        &KMEANS_CATALOG_ENTRY,
    )
}

#[test]
fn matrix_and_vector_outputs_are_double_with_requested_shapes() {
    let result = infer(
        vec![numeric(6, 3), scalar()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::Int64,
            },
        ],
        4,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 4);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(6), Some(1)])
    );
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        result.outputs[3].shape,
        ShapeFact::from(vec![Some(6), Some(2)])
    );
    assert!(result.outputs.iter().all(|fact| fact
        .numeric()
        .is_some_and(|numeric| numeric.class == NumericClass::Double)));

    let vector = infer(
        vec![numeric(1, 5), scalar()],
        vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
        2,
    );
    assert_eq!(
        vector.outputs[0].shape,
        ShapeFact::from(vec![Some(5), Some(1)])
    );
    assert_eq!(
        vector.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn numeric_start_supplies_empty_cluster_count() {
    let result = infer(
        vec![numeric(5, 3), numeric(0, 0), text(), numeric(2, 3)],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Empty,
            LiteralValue::String("Start".into()),
            LiteralValue::Unknown,
        ],
        2,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    let named = infer(
        vec![numeric(5, 3), numeric(0, 0), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Empty,
            LiteralValue::String("Start".into()),
            LiteralValue::String("plus".into()),
        ],
        1,
    );
    assert!(!named.diagnostics.is_empty());
}

#[test]
fn known_invalid_controls_are_diagnosed_without_inventing_unknown_values() {
    for literal in [
        LiteralValue::Number(0.0),
        LiteralValue::Number(1.5),
        LiteralValue::Integer {
            text: "18446744073709551616".into(),
            class: NumericClass::UInt64,
        },
    ] {
        let result = infer(
            vec![numeric(4, 2), scalar()],
            vec![LiteralValue::Unknown, literal],
            1,
        );
        assert!(!result.diagnostics.is_empty());
    }
    let invalid_distance = infer(
        vec![numeric(4, 2), scalar(), text(), text()],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("Distance".into()),
            LiteralValue::String("unknown".into()),
        ],
        1,
    );
    assert!(!invalid_distance.diagnostics.is_empty());
    let dynamic = infer(vec![numeric(4, 2), scalar(), text(), text()], vec![], 1);
    assert!(dynamic.diagnostics.is_empty(), "{:?}", dynamic.diagnostics);
}
