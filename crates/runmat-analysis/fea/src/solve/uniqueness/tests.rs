use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
mod prepared;
mod record;

fn matrix(scale: f64, singular: bool) -> RealSparseMatrix {
    let values = if singular {
        [[1.0, -1.0], [-1.0, 1.0]]
    } else {
        [[1.0, 2.0], [3.0, 4.0]]
    };
    let mut a = RealSparseAssembler::new(
        2,
        2,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for (row, entries) in values.iter().enumerate() {
        for (column, value) in entries.iter().enumerate() {
            a.add(row, column, value * scale);
        }
    }
    a.finish().unwrap()
}
fn policy() -> SparseUniquenessPolicy {
    SparseUniquenessPolicy {
        candidate_relative_tolerance: 1e-3,
        restart: 2,
        maximum_iterations_per_column: 8,
    }
}

#[test]
fn complete_shared_candidates_prove_independent_indefinite_scaled_operator() {
    for scale in [1e-120, 1.0, 1e120] {
        let a = matrix(scale, false);
        let envelope = SparseUniquenessEnvelope::derive(&a, policy()).unwrap();
        let SparseUniquenessVerdict::Verified {
            certificate,
            iterations,
        } = certify_sparse_uniqueness(&a, policy(), envelope).unwrap()
        else {
            panic!("independent nonsingular matrix");
        };
        assert_eq!(
            certificate.operator_identity(),
            a.numerical_identity().unwrap()
        );
        assert_eq!(certificate.dimension(), 2);
        assert!(certificate.defect_infinity_upper() < 1e-12);
        assert!(iterations > 0 && iterations <= 16);
    }
}

#[test]
fn exact_null_vector_stays_indeterminate_and_admission_precedes_factorization() {
    let a = matrix(1.0, true);
    let envelope = SparseUniquenessEnvelope::derive(&a, policy()).unwrap();
    assert_eq!(
        certify_sparse_uniqueness(
            &a,
            policy(),
            SparseUniquenessEnvelope {
                resident_bytes: envelope.resident_bytes - 1,
                ..envelope
            }
        ),
        Err(SparseUniquenessError::Admission)
    );
    assert!(matches!(
        certify_sparse_uniqueness(&a, policy(), envelope).unwrap(),
        SparseUniquenessVerdict::Indeterminate {
            reason: SparseUniquenessUncertainty::FactorUnavailable,
            ..
        }
    ));
    assert!(SparseUniquenessEnvelope::derive(
        &a,
        SparseUniquenessPolicy {
            maximum_iterations_per_column: 0,
            ..policy()
        }
    )
    .is_err());
}

#[test]
fn cancellation_never_yields_a_certificate() {
    let a = matrix(1.0, false);
    let envelope = SparseUniquenessEnvelope::derive(&a, policy()).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            handler: None,
            cancellation: token,
            attempt_id: None,
        });
    assert_eq!(
        certify_sparse_uniqueness(&a, policy(), envelope),
        Err(SparseUniquenessError::Cancelled)
    );
}

#[test]
fn count_only_planning_and_actual_operator_rederivation_share_one_authority() {
    let a = matrix(1.0, false);
    let exact = SparseUniquenessEnvelope::derive(&a, policy()).unwrap();
    assert_eq!(
        exact,
        SparseUniquenessEnvelope::for_counts(2, 4, 2, policy()).unwrap()
    );
    assert!(SparseUniquenessEnvelope::for_counts(2, 5, 2, policy()).is_err());
    assert!(
        SparseUniquenessEnvelope::for_counts(usize::MAX, usize::MAX, usize::MAX, policy()).is_err()
    );
    assert_eq!(
        certify_sparse_uniqueness(
            &a,
            policy(),
            SparseUniquenessEnvelope {
                scalar_operations: exact.scalar_operations - 1,
                ..exact
            }
        ),
        Err(SparseUniquenessError::Admission)
    );
}
