use super::super::inference::test_support::{numeric, numeric_scalar, request};
use super::LDIVIDE_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{NumericClass, NumericDomain, NumericFact, ShapeFact};

#[test]
fn preserves_broadcast_shape_and_single_precision() {
    let inferred = infer_catalog_call(
        &LDIVIDE_CATALOG_ENTRY,
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

#[test]
fn scalar_inputs_produce_a_double_scalar_fact() {
    let inferred = infer_catalog_call(
        &LDIVIDE_CATALOG_ENTRY,
        &request(vec![
            numeric_scalar(NumericClass::Double, NumericDomain::Real),
            numeric_scalar(NumericClass::Double, NumericDomain::Real),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred.outputs[0]
        .shape
        .is_proven_equivalent(&ShapeFact::Scalar));
    assert_eq!(
        inferred.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}
