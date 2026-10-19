use super::*;
use runmat_types::*;

#[test]
fn tolerant_fractional_dimension_remains_conservative_without_false_rejection() {
    let result = infer(
        shape(&[1, 2]),
        shape(&[1, 2]),
        Some(LiteralValue::Number(1.0000001)),
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}
fn number(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
fn infer(x: ShapeFact, e: ShapeFact, dim: Option<LiteralValue>) -> CallInference {
    let mut arguments = vec![number(x), number(e)];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    if let Some(dim) = dim {
        arguments.push(number(ShapeFact::Scalar));
        literals.push(dim);
    }
    super::inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
        &HISTC_CATALOG_ENTRY,
    )
}
fn shape(dims: &[usize]) -> ShapeFact {
    ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>())
}
#[test]
fn known_default_explicit_and_edge_orientation_shapes() {
    for (x, e, d, expected) in [
        (vec![3, 2], vec![4, 1], None, vec![4, 2]),
        (
            vec![2, 3],
            vec![1, 5],
            Some(LiteralValue::Number(2.)),
            vec![2, 5],
        ),
        (vec![1, 3], vec![4, 1], None, vec![4, 1]),
        (vec![3, 2], vec![4, 2], None, vec![4, 2]),
    ] {
        let out = infer(shape(&x), shape(&e), d);
        assert!(out.diagnostics.is_empty());
        assert_eq!(out.outputs[0].shape, shape(&expected));
        assert_eq!(out.outputs[1].shape, shape(&x));
        assert_eq!(
            out.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
    }
}
#[test]
fn unknown_data_shape_does_not_invent_a_vector() {
    assert_eq!(
        infer(ShapeFact::Unknown, shape(&[1, 3]), None).outputs[0].shape,
        ShapeFact::Unknown
    );
    assert_eq!(
        infer(shape(&[2, 3]), shape(&[1, 3]), Some(LiteralValue::Unknown)).outputs[0].shape,
        ShapeFact::Unknown
    );
}
#[test]
fn empty_edges_empty_indices_and_invalid_dimension() {
    let out = infer(shape(&[2, 3]), shape(&[0, 0]), None);
    assert_eq!(out.outputs[1].shape, shape(&[0, 0]));
    assert!(!infer(
        shape(&[2, 3]),
        shape(&[1, 3]),
        Some(LiteralValue::Number(0.))
    )
    .diagnostics
    .is_empty());
}
