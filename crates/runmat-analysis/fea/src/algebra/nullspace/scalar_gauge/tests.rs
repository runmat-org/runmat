use super::*;
use crate::algebra::{spmv, MatrixProperties, OperatorRole};
fn matrix() -> RealSparseMatrix {
    RealSparseMatrix::from_dense(
        &[
            2.0, -2.0, 0.0, 0.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0, 0.0, 0.0, -5.0, 5.0,
        ],
        4,
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap()
}
#[test]
fn disconnected_constant_modes_lower_without_changing_original_compatible_equations() {
    let a = matrix();
    let b = [2.0, -2.0, 10.0, -10.0];
    let (prepared, rhs) = prepare_scalar_component_gauges(&a, &b, &[0, 2], &[0.0; 4]).unwrap();
    let solution = [0.0, -1.0, 0.0, -2.0];
    assert_eq!(spmv(&prepared, &solution).unwrap(), rhs);
    assert_eq!(spmv(&a, &solution).unwrap(), b);
    assert!(prepare_scalar_component_gauges(&a, &b, &[0, 1], &[0.0; 4]).is_err());
    assert!(prepare_scalar_component_gauges(&a, &b, &[0, 0], &[0.0; 4]).is_err());
    assert!(prepare_scalar_component_gauges(&a, &b, &[4], &[0.0; 4]).is_err());
}
#[test]
fn incompatible_source_or_nonconstant_operator_cannot_hide_in_replaced_row() {
    let a = matrix();
    assert!(
        prepare_scalar_component_gauges(&a, &[2.0, -1.0, 10.0, -10.0], &[0, 2], &[0.0; 4]).is_err()
    );
    let bad = RealSparseMatrix::from_dense(
        &[3.0, -2.0, -2.0, 2.0],
        2,
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    assert!(prepare_scalar_component_gauges(&bad, &[0.0, 0.0], &[0], &[0.0; 2]).is_err());
}

#[test]
fn source_construction_roundoff_is_explicit_and_never_changes_the_original_rhs() {
    let a = matrix();
    let b = [1e-30, -9e-31, 0.0, 0.0];
    assert!(prepare_scalar_component_gauges(&a, &b, &[0, 2], &[0.0; 4]).is_err());
    let (_, rhs) =
        prepare_scalar_component_gauges(&a, &b, &[0, 2], &[2e-31, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(rhs[1], b[1]);
    assert!(prepare_scalar_component_gauges(&a, &b, &[0, 2], &[f64::NAN; 4]).is_err());
}
