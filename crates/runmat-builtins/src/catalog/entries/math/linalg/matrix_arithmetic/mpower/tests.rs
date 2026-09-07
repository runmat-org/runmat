use super::MPOWER_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
fn numeric(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
#[test]
fn preserves_square_matrix_shape() {
    let request = CallRequest {
        arguments: vec![
            numeric(ShapeFact::from(vec![Some(3), Some(3)])),
            numeric(ShapeFact::Scalar),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = infer_catalog_call(&MPOWER_CATALOG_ENTRY, &request);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
}

#[test]
fn diagnoses_proven_nonsquare_base() {
    let request = CallRequest {
        arguments: vec![
            numeric(ShapeFact::from(vec![Some(2), Some(3)])),
            numeric(ShapeFact::Scalar),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = infer_catalog_call(&MPOWER_CATALOG_ENTRY, &request);
    assert!(!inferred.diagnostics.is_empty());
}
