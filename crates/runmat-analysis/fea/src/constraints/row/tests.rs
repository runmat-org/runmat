use super::*;

fn term(node: u8, coefficient: f64) -> CanonicalStructuralConstraintTerm {
    CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes([node; 32])),
        dof: CanonicalStructuralDof::TRANSLATIONS[0],
        coefficient,
    }
}

#[test]
fn sparse_combination_reuses_storage_and_preserves_addition_order() {
    let terms = vec![
        term(2, 4.0),
        term(1, 1e16),
        term(3, 7.0),
        term(1, -1e16),
        term(2, -4.0),
        term(1, 1.0),
        term(3, 2.0),
    ];
    let pointer = terms.as_ptr();
    let capacity = terms.capacity();
    let row = new_sparse("combined", terms, 0.0).unwrap();
    assert_eq!(row.terms.as_ptr(), pointer);
    assert_eq!(row.terms.capacity(), capacity);
    assert_eq!(row.terms, vec![term(1, 1.0), term(3, 9.0)]);
    let expected = new_sparse("combined", vec![term(1, 1.0), term(3, 9.0)], 0.0).unwrap();
    assert_eq!(row, expected);
}

#[test]
fn cancellation_and_nonfinite_equations_still_fail() {
    assert!(new_sparse("cancelled", vec![term(1, 4.0), term(1, -4.0)], 0.0).is_err());
    assert!(new_sparse("rhs", vec![term(1, 1.0)], f64::INFINITY).is_err());
    assert!(new_sparse("overflow", vec![term(1, f64::MAX), term(1, f64::MAX)], 0.0).is_err());
}

#[test]
fn sparse_combination_does_not_retain_lowering_growth_slack() {
    let mut terms = Vec::with_capacity(1024);
    terms.extend([term(1, 2.0), term(1, 3.0), term(2, 4.0)]);
    let original_len = terms.len();
    let row = new_sparse("slack", terms, 0.0).unwrap();
    assert_eq!(row.terms, vec![term(1, 5.0), term(2, 4.0)]);
    assert_eq!(row.terms.capacity(), original_len);
}

#[test]
fn large_duplicate_rows_preserve_standard_stable_summation_and_encoding_budget() {
    use super::super::plan::{PLAN_LIMITS, PLAN_PREFIX};
    use runmat_canonical_codec::{
        canonical_payload_materialization_resident_bytes_estimate, encode_contract,
    };
    let input: Vec<_> = (0..128)
        .flat_map(|_| [term(2, 3.0), term(1, 1e16), term(1, -1e16), term(1, 1.0)])
        .collect();
    let mut reference = input.clone();
    reference.sort_by_key(term_key);
    let sum: f64 = reference
        .iter()
        .filter(|t| t.owner == term(1, 1.0).owner)
        .map(|t| t.coefficient)
        .sum();
    let p = constraint_row_workspace_profile("duplicate", &input, 0.0).unwrap();
    assert_eq!(
        p.ordering_bytes,
        (input.len() * std::mem::size_of::<usize>()) as u64
    );
    let actual = new_sparse("duplicate", input, 0.0).unwrap();
    assert_eq!(actual.terms, vec![term(1, sum), term(2, 384.0)]);
    let expected = new_sparse("duplicate", vec![term(1, sum), term(2, 384.0)], 0.0).unwrap();
    assert_eq!(actual.identity, expected.identity);
    for recovery in [
        ReactionRecovery::UnconstrainedEquilibriumResidual,
        ReactionRecovery::ConstraintMultiplier,
        ReactionRecovery::RedundantEquation,
    ] {
        for quantity in [
            ConstraintResidualQuantity::Displacement,
            ConstraintResidualQuantity::Rotation,
        ] {
            let bytes = encode_contract(
                PLAN_PREFIX,
                ROW_DOMAIN,
                &(
                    &actual.constraint_id,
                    &actual.terms,
                    actual.right_hand_side,
                    quantity,
                    recovery,
                ),
                PLAN_LIMITS,
            )
            .unwrap();
            assert!(
                p.identity_materialization_bytes
                    >= canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap()
            );
        }
    }
}

#[test]
fn cancelled_row_finalization_rejects_before_mutation() {
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        new_sparse("cancelled", vec![term(1, 1.0)], 0.0).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}
