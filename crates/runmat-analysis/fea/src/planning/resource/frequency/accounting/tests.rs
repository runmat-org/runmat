use super::*;

#[test]
fn direct_workspace_counts_right_scaling_without_changing_modal_inventory() {
    let reduction = crate::constraints::AffineConstraintReduction {
        full_dof_count: 3,
        reduced_dof_count: 2,
        free_dof_indices: vec![0, 2],
        pivot_dof_indices: vec![1],
        recovery_row_offsets: vec![0, 1, 1, 2],
        recovery_column_indices: vec![0, 1],
        recovery_coefficients: vec![1.0, 1.0],
        offset: vec![0.0; 3],
    };
    reduction.validate().unwrap();
    // Three reduced CSR operators: 9 row offsets + 4 value/index pairs.
    // Recovery: 9 indices + 5 floating values. Both store 8-byte entries.
    let operator_and_map = 9 * 8 + 4 * 16 + (9 + 5) * 8;
    for restart in [2, 4, 32] {
        let block = 4;
        let ordinary_vectors = (restart + 9) * block;
        let right_scaling_vectors = 3 * block;
        let hessenberg = (restart + 1) * restart;
        let projected_vectors = 3 * restart + 1;
        let full_vectors = 14 * 3;
        let expected = operator_and_map
            + 8 * (ordinary_vectors
                + right_scaling_vectors
                + hessenberg
                + projected_vectors
                + full_vectors);
        assert_eq!(
            direct_workspace_bytes(3, 2, 4, &reduction, restart).unwrap(),
            expected
        );
        assert_eq!(right_scaling_vectors * 8, 96);
    }
    // Modal projection retains four full-by-mode arrays, eleven dense arrays,
    // fourteen full vectors and eight reduced vectors; no Krylov diagonal.
    assert_eq!(
        modal_workspace_bytes(3, 2).unwrap(),
        8 * (4 * 6 + 11 * 4 + 14 * 3 + 8 * 2)
    );
    assert!(direct_workspace_bytes(3, u64::MAX, 4, &reduction, 32).is_err());
    assert!(direct_workspace_bytes(3, 2, 4, &reduction, u64::MAX).is_err());
}
