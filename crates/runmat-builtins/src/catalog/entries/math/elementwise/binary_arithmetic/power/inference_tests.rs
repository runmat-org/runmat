use super::super::inference::test_support::{numeric, numeric_scalar, request};
use super::POWER_CATALOG_ENTRY;
use crate::infer_catalog_call;
use runmat_types::{CertaintyFact, DynamicReason, NumericClass, NumericDomain, ShapeFact};

#[test]
fn keeps_known_shape_without_inventing_a_result_domain() {
    let inferred = infer_catalog_call(
        &POWER_CATALOG_ENTRY,
        &request(vec![
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                &[Some(2), Some(1)],
            ),
            numeric(
                NumericClass::Double,
                NumericDomain::Real,
                &[Some(1), Some(3)],
            ),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(inferred.outputs[0].numeric(), None);
    assert!(matches!(
        inferred.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));
}

#[test]
fn scalar_inputs_keep_scalar_shape_without_inventing_a_result_domain() {
    let inferred = infer_catalog_call(
        &POWER_CATALOG_ENTRY,
        &request(vec![
            numeric_scalar(NumericClass::Double, NumericDomain::Real),
            numeric_scalar(NumericClass::Double, NumericDomain::Real),
        ]),
    );
    assert!(inferred.diagnostics.is_empty());
    assert!(inferred.outputs[0]
        .shape
        .is_proven_equivalent(&ShapeFact::Scalar));
    assert_eq!(inferred.outputs[0].numeric(), None);
    assert!(matches!(
        inferred.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));
}
