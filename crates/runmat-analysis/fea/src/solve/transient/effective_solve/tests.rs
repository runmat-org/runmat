use runmat_execution::Digest;

use super::*;

#[test]
fn correction_equation_normalization_is_invariant_to_physical_scale() {
    for scale in [1.0e-12, 1.0, 1.0e12] {
        let operator = RealSparseMatrix::diagonal(
            vec![2.0 * scale],
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let (operator_scale, solution_scale) =
            active_correction_scales(&operator, &[6.0 * scale], &[false], None).unwrap();

        assert_eq!(operator_scale, 2.0 * scale);
        assert_eq!(solution_scale, 3.0);
    }
}

#[test]
fn direct_constraint_reactions_do_not_set_correction_scale() {
    let operator = RealSparseMatrix::diagonal(
        vec![1.0e12, 2.0],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let (operator_scale, solution_scale) =
        active_correction_scales(&operator, &[1.0e12, 6.0], &[true, false], None).unwrap();

    assert_eq!(operator_scale, 2.0);
    assert_eq!(solution_scale, 3.0);
}

#[test]
fn affine_constraint_reactions_do_not_set_correction_scale() {
    let operator = RealSparseMatrix::diagonal(
        vec![2.0, 2.0],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let constraint = ResolvedConstraintMatrix {
        full_dof_count: 2,
        row_offsets: vec![0, 2],
        column_indices: vec![0, 1],
        coefficients: vec![1.0, -1.0],
        right_hand_side: vec![0.0],
        row_identities: vec![Digest::sha256(b"equal-motion")],
        constraint_ids: vec!["equal_motion".into()],
    };
    let reduction = constraint.affine_reduction().unwrap();
    let (operator_scale, solution_scale) = active_correction_scales(
        &operator,
        &[1.0e12, -1.0e12 + 6.0],
        &[false, false],
        Some(&reduction),
    )
    .unwrap();

    assert_eq!(operator_scale, 4.0);
    assert_eq!(solution_scale, 1.5);
}
