use super::MLDIVIDE_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
fn numeric(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}
#[test]
fn infers_left_solve_shape() {
    let request = CallRequest {
        arguments: vec![
            numeric(vec![Some(3), Some(2)]),
            numeric(vec![Some(3), Some(4)]),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = infer_catalog_call(&MLDIVIDE_CATALOG_ENTRY, &request);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
}

#[test]
fn sparse_complete_solve_inference_preserves_native_class_and_solution_geometry() {
    for class in [NumericClass::Single, NumericClass::Double] {
        for domain in [NumericDomain::Real, NumericDomain::Complex] {
            for (m, n) in [(3, 2), (2, 3), (2, 2), (0, 3), (2, 0)] {
                let kind = ValueKindFact::Numeric(NumericFact { class, domain });
                let request = CallRequest {
                    arguments: vec![
                        ValueFact::proven(
                            kind.clone(),
                            ShapeFact::from(vec![Some(m), Some(n)]),
                            StorageFact::Sparse,
                        ),
                        ValueFact::proven(
                            kind.clone(),
                            ShapeFact::from(vec![Some(m), Some(2)]),
                            StorageFact::Dense,
                        ),
                    ],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                };
                let inferred = infer_catalog_call(&MLDIVIDE_CATALOG_ENTRY, &request);
                assert!(inferred.diagnostics.is_empty());
                assert_eq!(
                    inferred.outputs[0].shape,
                    ShapeFact::from(vec![Some(n), Some(2)])
                );
                assert_eq!(inferred.outputs[0].kind, kind);
                assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
            }
        }
    }
}
