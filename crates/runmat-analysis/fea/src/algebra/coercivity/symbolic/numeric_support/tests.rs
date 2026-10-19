use super::*;
use crate::algebra::{CoercivityCertificate, MatrixProperties, OperatorRole};

fn dense_pattern_matrix(dimension: usize, value: impl Fn(usize, usize) -> f64) -> RealSparseMatrix {
    let offsets = (0..=dimension)
        .map(|row| (row * dimension) as u64)
        .collect();
    let columns = (0..dimension)
        .flat_map(|_| (0..dimension).map(|column| column as u64))
        .collect();
    let pattern =
        SparsePattern::try_new(dimension as u64, dimension as u64, offsets, columns).unwrap();
    let values = (0..dimension)
        .flat_map(|row| {
            (0..dimension)
                .map(|column| value(row, column))
                .collect::<Vec<_>>()
        })
        .collect();
    RealSparseMatrix::try_new(
        pattern,
        values,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn stored_zero_edges_do_not_create_dense_proof_fill_or_change_source_identity() {
    const N: usize = 32;
    // Positive tridiagonal: eigenvalues 4-2cos(k pi/(N+1)) are strictly positive.
    // Full CSR deliberately retains numerical zero entries, as constrained assembly does.
    let matrix = dense_pattern_matrix(N, |row, column| {
        if row == column {
            4.0
        } else if row.abs_diff(column) == 1 {
            -1.0
        } else {
            0.0
        }
    });
    let original = matrix.clone();
    let exact = CoercivitySymbolicPlan::derive_for_matrix(&matrix).unwrap();
    let structural = CoercivitySymbolicPlan::derive(matrix.pattern()).unwrap();
    exact.validate_for(matrix.pattern()).unwrap();
    assert_eq!(exact.factor_nonzero_count(), (2 * N - 1) as u64);
    assert_eq!(exact.update_count(), (N - 1) as u64);
    assert_eq!(structural.factor_nonzero_count(), (N * (N + 1) / 2) as u64);
    let certificate = CoercivityCertificate::prove(&matrix).unwrap();
    certificate.validate_for(&matrix).unwrap();
    assert_eq!(certificate.work_counts().1, exact.factor_nonzero_count());
    assert_eq!(certificate.work_counts().2, exact.update_count());
    let envelope = CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap();
    assert!(certificate.work_counts().1 <= envelope.factor_nonzero_limit());
    assert!(certificate.work_counts().2 <= envelope.update_limit());
    assert_eq!(matrix, original, "proof must not modify the source matrix");
}

#[test]
fn every_nonzero_coefficient_is_retained_without_a_drop_tolerance() {
    for coupling in [
        f64::from_bits(1),
        -f64::from_bits(1),
        1e-300,
        -1e-300,
        1e-12,
        -1e-12,
    ] {
        let matrix = dense_pattern_matrix(4, |row, column| {
            if row == column {
                2.0
            } else if [row, column] == [0, 3] || [row, column] == [3, 0] {
                coupling
            } else {
                -0.0
            }
        });
        let exact = CoercivitySymbolicPlan::derive_for_matrix(&matrix).unwrap();
        assert_eq!(exact.factor_nonzero_count(), 5);
        assert_eq!(exact.update_count(), 1);
        let certificate = CoercivityCertificate::prove(&matrix).unwrap();
        let changed = dense_pattern_matrix(4, |row, column| if row == column { 2.0 } else { 0.0 });
        assert_eq!(
            certificate.validate_for(&changed),
            Err(CoercivityError::BindingMismatch)
        );
        assert_eq!(
            CoercivitySymbolicPlan::derive_for_matrix(&changed)
                .unwrap()
                .factor_nonzero_count(),
            4
        );
    }
}

#[test]
fn exact_zero_support_still_rejects_singular_indefinite_and_asymmetric_operators() {
    for coupling in [1.0, 2.0] {
        let matrix = dense_pattern_matrix(4, |row, column| {
            if row == column {
                1.0
            } else if row < 2 && column < 2 {
                coupling
            } else {
                0.0
            }
        });
        assert!(matches!(
            CoercivityCertificate::prove(&matrix),
            Err(CoercivityError::NonPositivePivot { .. })
        ));
    }
    let asymmetric = dense_pattern_matrix(4, |row, column| {
        if row == column {
            2.0
        } else if [row, column] == [0, 3] {
            1.0
        } else {
            0.0
        }
    });
    assert_eq!(
        CoercivityCertificate::prove(&asymmetric),
        Err(CoercivityError::AsymmetricValues)
    );
}

#[test]
fn actual_zero_subgraph_can_require_more_updates_than_its_structural_upper_graph() {
    let rows: &[&[u64]] = &[
        &[0, 1, 2, 3, 4],
        &[0, 1, 2, 3, 4],
        &[0, 1, 2],
        &[0, 1, 3],
        &[0, 1, 4],
    ];
    let mut offsets = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for (row, entries) in rows.iter().enumerate() {
        for &column in *entries {
            columns.push(column);
            values.push(if row as u64 == column {
                6.0
            } else if [row as u64, column] == [0, 2] || [row as u64, column] == [2, 0] {
                0.0
            } else {
                -1.0
            });
        }
        offsets.push(columns.len() as u64);
    }
    // Strict diagonal dominance with positive diagonal independently establishes SPD.
    let matrix = RealSparseMatrix::try_new(
        SparsePattern::try_new(5, 5, offsets, columns).unwrap(),
        values,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        Vec::new(),
    )
    .unwrap();
    let upper = CoercivitySymbolicPlan::derive(matrix.pattern()).unwrap();
    let exact = CoercivitySymbolicPlan::derive_for_matrix(&matrix).unwrap();
    assert_eq!(upper.update_count(), 10);
    assert_eq!(exact.update_count(), 11);
    let certificate = CoercivityCertificate::prove(&matrix).unwrap();
    certificate.validate_for(&matrix).unwrap();
    assert_eq!(certificate.work_counts().2, 11);
    let envelope = CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap();
    assert!(certificate.work_counts().3 <= envelope.resident_bytes_limit());
}
