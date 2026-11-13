use super::*;

#[test]
fn retained_candidate_factor_matches_public_proof_and_original_operator() {
    let a = matrix(1.0, false);
    let envelope = SparseUniquenessEnvelope::derive(&a, policy()).unwrap();
    let original = certify_sparse_uniqueness(&a, policy(), envelope).unwrap();
    let (prepared, factor) = prepare_sparse_uniqueness(&a, policy(), envelope).unwrap();
    assert_eq!(prepared, original);
    let factor = factor.unwrap();
    for rhs in [[1., 0.], [0., 1.], [2., -3.]] {
        let x = factor.apply(&rhs, || false).unwrap();
        let action = crate::algebra::spmv(&a, &x).unwrap();
        for (v, r) in action.iter().zip(rhs) {
            assert!((v - r).abs() < 1e-12);
        }
    }
}

#[test]
fn unavailable_original_factor_remains_explicitly_absent() {
    let a = matrix(1.0, true);
    let envelope = SparseUniquenessEnvelope::derive(&a, policy()).unwrap();
    let (verdict, factor) = prepare_sparse_uniqueness(&a, policy(), envelope).unwrap();
    assert!(factor.is_none());
    assert!(matches!(
        verdict,
        SparseUniquenessVerdict::Indeterminate {
            reason: SparseUniquenessUncertainty::FactorUnavailable,
            ..
        }
    ));
}
