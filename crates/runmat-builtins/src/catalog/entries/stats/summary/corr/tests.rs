use super::*;
use runmat_types::*;
fn shape(dims: &[usize]) -> ShapeFact {
    ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>())
}
fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        shape,
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
        &CORR_CATALOG_ENTRY,
    )
}
#[test]
fn matrix_and_cross_matrix_outputs_are_double() {
    let result = infer(
        vec![numeric(shape(&[5, 2])), numeric(shape(&[5, 3]))],
        vec![],
        3,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, shape(&[2, 3]));
    assert_eq!(result.outputs[1].shape, shape(&[2, 3]));
    assert_eq!(result.outputs[2].shape, ShapeFact::Scalar);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}
#[test]
fn unknown_second_argument_does_not_invent_column_count() {
    let result = infer(
        vec![
            numeric(shape(&[5, 2])),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        vec![],
        1,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Unknown]
        }
    );
}
#[test]
fn rejects_known_bad_rank_and_observation_count() {
    assert!(!infer(vec![numeric(shape(&[2, 2, 2]))], vec![], 1)
        .diagnostics
        .is_empty());
    assert!(!infer(
        vec![numeric(shape(&[2, 2])), numeric(shape(&[3, 2]))],
        vec![],
        1
    )
    .diagnostics
    .is_empty());
}
