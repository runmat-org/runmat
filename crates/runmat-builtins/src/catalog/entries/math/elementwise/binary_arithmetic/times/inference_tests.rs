use super::super::inference::test_support::{numeric, request};
use super::TIMES_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{NumericClass, NumericDomain, NumericFact, ShapeFact};

#[test]
fn preserves_broadcast_shape_and_single_precision() {
    let inferred = infer_catalog_call(
        &TIMES_CATALOG_ENTRY,
        &request(vec![
            numeric(
                NumericClass::Single,
                NumericDomain::Real,
                &[Some(3), Some(1)],
            ),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                &[Some(1), Some(4)],
            ),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
    assert_eq!(
        inferred.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
}
