use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
use crate::solve::krylov::right_ilu::SparseRightIlu;

fn matrix(values: [[f64; 2]; 2]) -> RealSparseMatrix {
    let mut a = RealSparseAssembler::new(
        2,
        2,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for (row, entries) in values.iter().enumerate() {
        for (column, value) in entries.iter().enumerate() {
            a.add(row, column, *value);
        }
    }
    a.finish().unwrap()
}

fn verdict(a: &RealSparseMatrix, columns: [[f64; 2]; 2]) -> RightInverseVerdict {
    let mut check = RightInverseCheck::new(
        a,
        RightInverseWorkEnvelope::derive(2, a.values().len()).unwrap(),
    )
    .unwrap();
    for (index, column) in columns.iter().enumerate() {
        check.push_column(index, column).unwrap();
    }
    check.finish().unwrap()
}

#[test]
fn independent_indefinite_inverse_and_scale_invariance() {
    // A=[[1,2],[3,4]], det=-2, exact B=[[-2,1],[1.5,-.5]].
    for scale in [1e-120, 1.0, 1e120] {
        let a = matrix([[scale, 2.0 * scale], [3.0 * scale, 4.0 * scale]]);
        let RightInverseVerdict::Verified(certificate) = verdict(
            &a,
            [[-2.0 / scale, 1.5 / scale], [1.0 / scale, -0.5 / scale]],
        ) else {
            panic!("independent inverse not certified");
        };
        assert_eq!(
            certificate.operator_identity(),
            a.numerical_identity().unwrap()
        );
        assert_eq!(certificate.dimension(), 2);
        assert!(certificate.defect_infinity_upper() < 1e-12);
        // The actual shared ILU is merely the candidate provider; it is not itself proof.
        let ilu = SparseRightIlu::factor(&a, || false).unwrap();
        let cols = [
            ilu.apply(&[1.0, 0.0], || false).unwrap(),
            ilu.apply(&[0.0, 1.0], || false).unwrap(),
        ];
        assert!(matches!(
            verdict(
                &a,
                [
                    cols[0].clone().try_into().unwrap(),
                    cols[1].clone().try_into().unwrap()
                ]
            ),
            RightInverseVerdict::Verified(_)
        ));
    }
}

#[test]
fn exact_null_vector_and_bad_candidate_are_indeterminate_not_singular_claims() {
    let singular = matrix([[1.0, -1.0], [-1.0, 1.0]]);
    // Analytic null vector [1,1]; no candidate can establish an inverse.
    assert!(matches!(
        verdict(&singular, [[1.0, 0.0], [0.0, 1.0]]),
        RightInverseVerdict::Indeterminate { .. }
    ));
    let invertible = matrix([[1.0, 0.0], [0.0, 2.0]]);
    assert!(matches!(
        verdict(&invertible, [[0.0, 0.0], [0.0, 0.0]]),
        RightInverseVerdict::Indeterminate { .. }
    ));
}

#[test]
fn outward_bound_never_certifies_rounding_hidden_defect() {
    // Nearly rank-one; a wrong candidate must not be rescued by cancellation.
    let a = matrix([[1.0, 1.0], [1.0, 1.0 + f64::EPSILON]]);
    assert!(matches!(
        verdict(&a, [[1e300, -1e300], [-1e300, 1e300]]),
        RightInverseVerdict::Indeterminate { .. }
    ));
    let overflow = matrix([[1e308, 0.0], [0.0, 1e308]]);
    assert!(matches!(
        verdict(&overflow, [[1e308, 0.0], [0.0, 1e308]]),
        RightInverseVerdict::Indeterminate { .. }
    ));
}

#[test]
fn original_inventory_and_admission_precede_work() {
    let a = matrix([[1.0, 0.0], [0.0, 2.0]]);
    let envelope = RightInverseWorkEnvelope::derive(2, a.values().len()).unwrap();
    assert!(matches!(
        RightInverseCheck::new(
            &a,
            RightInverseWorkEnvelope {
                scalar_operations: envelope.scalar_operations - 1,
                ..envelope
            }
        ),
        Err(RightInverseError::Admission)
    ));
    let mut check = RightInverseCheck::new(&a, envelope).unwrap();
    assert_eq!(
        check.push_column(1, &[0.0, 0.5]),
        Err(RightInverseError::Column)
    );
    assert_eq!(
        check.push_column(0, &[f64::NAN, 0.0]),
        Err(RightInverseError::Column)
    );
    check.push_column(0, &[1.0, 0.0]).unwrap();
    assert_eq!(
        check.push_column(0, &[1.0, 0.0]),
        Err(RightInverseError::Column)
    );
    assert_eq!(check.finish(), Err(RightInverseError::Column));
    assert_eq!(
        RightInverseWorkEnvelope::derive(usize::MAX, usize::MAX),
        Err(RightInverseError::Overflow)
    );
}

#[test]
fn cold_operator_reverification_preserves_exact_evidence_not_trusted_deserialization() {
    let a = matrix([[1.0, 2.0], [3.0, 4.0]]);
    let columns = [[-2.0, 1.5], [1.0, -0.5]];
    let expected = verdict(&a, columns);
    let bytes = serde_json::to_vec(&a).unwrap();
    let restored: RealSparseMatrix = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(verdict(&restored, columns), expected);
    let changed = matrix([[1.0, 2.0], [3.0, 4.125]]);
    let RightInverseVerdict::Verified(original) = expected else {
        panic!("original inverse");
    };
    let RightInverseVerdict::Verified(substituted) = verdict(&changed, columns) else {
        panic!("bounded changed operator");
    };
    assert_ne!(
        original.operator_identity(),
        substituted.operator_identity()
    );
    assert_eq!(original.inverse_identity(), substituted.inverse_identity());
    assert_ne!(
        original.defect_infinity_upper(),
        substituted.defect_infinity_upper()
    );
    // No Deserialize implementation grants a portable certificate authority.
    assert!(!serde_json::to_vec(&original).unwrap().is_empty());
}

#[test]
fn sparse_full_inventory_is_certified_without_dense_inverse_storage() {
    let n = 128;
    let mut assembler = RealSparseAssembler::new(
        n,
        n,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for row in 0..n {
        assembler.add(row, row, 4.0);
        if row > 0 {
            assembler.add(row, row - 1, -1.0);
        }
        if row + 1 < n {
            assembler.add(row, row + 1, -1.0);
        }
    }
    let a = assembler.finish().unwrap();
    let envelope = RightInverseWorkEnvelope::derive(n, a.values().len()).unwrap();
    let mut check = RightInverseCheck::new(&a, envelope).unwrap();
    let mut column = vec![0.0; n];
    for index in 0..n {
        column[index] = 0.25;
        check.push_column(index, &column).unwrap();
        column[index] = 0.0;
    }
    let RightInverseVerdict::Verified(certificate) = check.finish().unwrap() else {
        panic!("strict diagonal dominance");
    };
    // Independent diagonal B gives exact off-diagonal defect row sum 1/2.
    assert!(certificate.defect_infinity_upper() >= 0.5);
    assert!(certificate.defect_infinity_upper() < 0.5 + 1e-12);
    assert_eq!(certificate.dimension(), n);
}
