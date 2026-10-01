use super::*;
use runmat_types::*;

fn infer(shape: ShapeFact, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                }),
                shape,
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &TIEDRANK_CATALOG_ENTRY,
    )
}

#[test]
fn vector_matrix_nd_and_requested_outputs_match_rank_geometry() {
    let vector = infer(ShapeFact::from(vec![Some(1), Some(4)]), 2);
    assert_eq!(
        vector.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    assert_eq!(vector.outputs[1].shape, ShapeFact::Scalar);
    let matrix = infer(ShapeFact::from(vec![Some(3), Some(2)]), 2);
    assert_eq!(
        matrix.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    let nd = infer(ShapeFact::from(vec![Some(3), Some(2), Some(4)]), 3);
    assert_eq!(
        nd.outputs[1].shape,
        ShapeFact::from(vec![Some(3), Some(1), Some(4)])
    );
    assert_eq!(nd.outputs[2].shape, ShapeFact::Scalar);
    assert_eq!(nd.outputs[0].numeric().unwrap().class, NumericClass::Double);
    assert!(infer(ShapeFact::Unknown, 0).outputs.is_empty());
}

#[test]
fn dynamic_shape_does_not_guess_vector_tie_adjustment() {
    assert_eq!(
        infer(ShapeFact::from(vec![None, Some(2)]), 2).outputs[1].shape,
        ShapeFact::Unknown
    );
}

#[test]
fn rank_only_facts_remain_symbolic_without_dimension_allocation() {
    let shape = ShapeFact::Ranked { rank: usize::MAX };
    let inferred = infer(shape.clone(), 2);
    assert_eq!(inferred.outputs[0].shape, shape);
    assert_eq!(inferred.outputs[1].shape, ShapeFact::Unknown);
    assert!(inferred.diagnostics.is_empty());
}

#[test]
fn kendall_mode_has_three_corrections_and_unknown_flags_keep_shape_dynamic() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5), Some(1)]),
        StorageFact::Dense,
    );
    let flag = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![data, flag],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(1.0)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let known = inference::infer(&request, &TIEDRANK_CATALOG_ENTRY);
    assert!(known.diagnostics.is_empty(), "{:?}", known.diagnostics);
    assert_eq!(
        known.outputs[1].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let mut unknown_request = request;
    unknown_request.literals = LiteralContext::default();
    let unknown = inference::infer(&unknown_request, &TIEDRANK_CATALOG_ENTRY);
    assert_eq!(unknown.outputs[1].shape, ShapeFact::Unknown);
}
