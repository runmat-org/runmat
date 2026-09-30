use super::*;
use runmat_types::*;

fn value(shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallInference {
    super::inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &HISTCOUNTS2_CATALOG_ENTRY,
    )
}

#[test]
fn explicit_edges_determine_matrix_edges_and_original_index_shapes() {
    let result = infer(
        vec![
            value(&[2, 2]),
            value(&[2, 2]),
            value(&[1, 4]),
            value(&[1, 6]),
        ],
        vec![LiteralValue::Unknown; 4],
        5,
    );
    assert!(result.diagnostics.is_empty());
    for (slot, shape) in [[3, 5], [1, 4], [1, 6], [2, 2], [2, 2]].iter().enumerate() {
        assert_eq!(result.outputs[slot].shape, value(shape).shape);
        assert_eq!(
            result.outputs[slot].numeric().unwrap().class,
            NumericClass::Double
        );
    }
}

#[test]
fn scalar_counts_and_unknown_controls_are_conservative() {
    let counts = infer(
        vec![
            value(&[4, 1]),
            value(&[4, 1]),
            value(&[1, 1]),
            value(&[1, 1]),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(2.),
            LiteralValue::Number(4.),
        ],
        3,
    );
    assert_eq!(counts.outputs[0].shape, value(&[2, 4]).shape);
    let dynamic = infer(
        vec![value(&[4, 1]), value(&[4, 1])],
        vec![LiteralValue::Unknown; 2],
        3,
    );
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
    assert!(!infer(
        vec![value(&[4, 1]), value(&[4, 1])],
        vec![LiteralValue::Unknown; 2],
        6
    )
    .diagnostics
    .is_empty());
}

#[test]
fn descriptor_covers_core_forms() {
    let labels: Vec<&str> = HISTCOUNTS2_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"N = histcounts2(X, Y)"));
    assert!(labels.contains(&"N = histcounts2(X, Y, binsX, binsY)"));
    assert!(labels.contains(&"[N, Xedges, Yedges] = histcounts2(X, Y, Name, Value, ...)"));
}
