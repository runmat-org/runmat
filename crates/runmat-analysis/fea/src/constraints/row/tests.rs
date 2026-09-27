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
