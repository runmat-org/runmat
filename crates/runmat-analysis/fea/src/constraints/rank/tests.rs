use runmat_analysis_core::StructuralTranslationDof;
use runmat_meshing_core::StableDigest;

use super::*;
use crate::constraints::{row, CanonicalStructuralConstraintTerm};

fn equation(id: &str, coefficients: &[(u8, f64)], rhs: f64) -> CanonicalStructuralConstraintRow {
    let terms = coefficients
        .iter()
        .map(|&(node, coefficient)| CanonicalStructuralConstraintTerm {
            owner: CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes(
                [node; 32],
            )),
            dof: CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
            coefficient,
        })
        .collect();
    row::new_sparse(id, terms, rhs).unwrap()
}

fn counts(rows: &[CanonicalStructuralConstraintRow]) -> (usize, usize) {
    let result = classify(rows).unwrap();
    (result.active_row_count, result.redundant_row_count)
}

#[test]
fn empty_classification_and_profile_allocate_no_slots() {
    assert_eq!(counts(&[]), (0, 0));
    let profile = constraint_rank_workspace_profile(&[]).unwrap();
    assert_eq!(profile.total_requested_bytes, 0);
    assert_eq!(profile.graph_bytes, 0);
    assert_eq!(profile.component_count, 0);
}

#[test]
fn graph_connects_transitive_dofs_and_keeps_independent_rows_separate() {
    let rows = [
        equation("a", &[(1, 1.0), (2, 1.0)], 0.0),
        equation("independent", &[(9, 1.0)], 0.0),
        equation("b", &[(3, 1.0), (4, 1.0)], 0.0),
        equation("bridge", &[(2, 1.0), (3, 1.0)], 0.0),
    ];
    let graph = graph::Components::derive(&rows).unwrap();
    assert_eq!(graph.candidate_count, 7);
    assert_eq!(graph.dof_count, 5);
    assert_eq!(graph.components.len(), 2);
    assert_eq!(graph.row_components, [0, 1, 0, 0]);
    assert_eq!((graph.components[0].rows, graph.components[0].dofs), (3, 4));
    assert_eq!((graph.components[1].rows, graph.components[1].dofs), (1, 1));
    let profile = constraint_rank_workspace_profile(&rows).unwrap();
    assert_eq!(profile.basis_row_capacity, 4);
    assert_eq!(profile.basis_coefficient_slots, 13);
    assert_eq!(profile.basis_source_slots, 10);
    assert!(constraint_rank_graph_workspace_bytes(&rows).unwrap() >= profile.graph_bytes);
    assert_eq!(counts(&rows), (4, 0));
}

#[test]
fn independent_rows_have_linear_not_global_dense_workspace() {
    let rows: Vec<_> = (0..128u8)
        .map(|node| equation("same-id", &[(node, 1.0)], 0.0))
        .collect();
    let profile = constraint_rank_workspace_profile(&rows).unwrap();
    assert_eq!(profile.component_count, rows.len());
    assert_eq!(profile.basis_row_capacity, rows.len());
    assert_eq!(profile.basis_coefficient_slots, rows.len());
    assert_eq!(profile.basis_source_slots, rows.len());
    assert_eq!(counts(&rows), (128, 0));
}

#[test]
fn connected_elimination_fills_new_columns_and_preserves_borrowed_rows() {
    let rows = [
        equation("first", &[(1, 1.0), (3, 1.0)], 0.0),
        equation("fill", &[(1, 1.0), (2, 1.0)], 0.0),
        equation("last", &[(2, 1.0)], 0.0),
        equation("redundant", &[(3, 1.0)], 0.0),
    ];
    let original = rows.clone();
    let result = classify(&rows).unwrap();
    assert_eq!(
        (result.active_row_count, result.redundant_row_count),
        (3, 1)
    );
    assert_eq!(
        result.classifications,
        [
            ReactionRecovery::ConstraintMultiplier,
            ReactionRecovery::ConstraintMultiplier,
            ReactionRecovery::UnconstrainedEquilibriumResidual,
            ReactionRecovery::RedundantEquation,
        ]
    );
    assert_eq!(rows, original);
    let mut classified = rows;
    assert_eq!(row::classify(&mut classified).unwrap(), (3, 1));
    assert_eq!(
        row::validate_row_classification(&classified).unwrap(),
        (3, 1)
    );
    classified[3].reaction_recovery = ReactionRecovery::UnconstrainedEquilibriumResidual;
    assert!(row::validate_row_classification(&classified).is_err());
}

#[test]
fn repeated_equations_deduplicate_borrowed_source_ids() {
    let rows: Vec<_> = (0..256)
        .map(|_| equation("same-id", &[(1, 1.0), (2, 1.0)], 0.0))
        .collect();
    let profile = constraint_rank_workspace_profile(&rows).unwrap();
    assert_eq!(profile.component_count, 1);
    assert_eq!(profile.basis_row_capacity, 2);
    assert_eq!(profile.basis_source_slots, 512);
    assert_eq!(counts(&rows), (1, 255));
}

#[test]
fn provenance_conflict_is_lexical_and_cannot_cross_components() {
    let rows = [
        equation("z-first", &[(1, 1.0), (2, 1.0)], 0.0),
        equation("000-unrelated", &[(9, 1.0)], 0.0),
        equation("a-second", &[(1, 1.0)], 0.0),
        equation("conflict", &[(2, 1.0)], 1.0),
    ];
    assert!(
        matches!(classify(&rows), Err(ConstraintPlanError::ConflictingRow {
        first_constraint_id, second_constraint_id, ..
    }) if first_constraint_id == "a-second" && second_constraint_id == "conflict")
    );
}

#[test]
fn workspace_does_not_copy_constraint_id_payloads() {
    let short = [
        equation("a", &[(1, 1.0), (2, 1.0)], 0.0),
        equation("b", &[(2, 1.0)], 0.0),
    ];
    let mut long = short.clone();
    for (index, row) in long.iter_mut().enumerate() {
        row.constraint_id = format!("{index}{}", "x".repeat(8192));
    }
    assert_eq!(
        constraint_rank_workspace_profile(&short).unwrap(),
        constraint_rank_workspace_profile(&long).unwrap()
    );
    assert_eq!(counts(&short), counts(&long));
}

#[test]
fn scaled_conflicting_equations_are_not_redundant() {
    let rows = [
        equation("fixed_x", &[(1, 1.0)], 0.0),
        equation("scaled_x", &[(1, 1.0e-12)], 1.0e-12),
    ];
    assert!(matches!(
        classify(&rows),
        Err(ConstraintPlanError::ConflictingRow { .. })
    ));
}

#[test]
fn small_pivot_elimination_preserves_its_right_hand_side_contribution() {
    let rows = [
        equation("large_x", &[(1, 1.0)], 1.0e12),
        equation("mixed", &[(1, 1.0e-12), (2, 1.0)], 2.0),
        equation("y", &[(2, 1.0)], 1.0),
    ];
    assert_eq!(counts(&rows), (2, 1));
    assert_eq!(
        classify(&rows).unwrap().classifications[2],
        ReactionRecovery::RedundantEquation
    );
}

#[test]
fn large_prescribed_values_do_not_hide_resolvable_conflicts() {
    let consistent = [
        equation("first", &[(1, 1.0)], 1.0e12),
        equation("same", &[(1, 1.0)], 1.0e12),
    ];
    assert_eq!(counts(&consistent), (1, 1));
    let rows = [
        equation("first", &[(1, 1.0)], 1.0e12),
        equation("second", &[(1, 1.0)], 1.0e12 + 1.0),
    ];
    assert!(matches!(
        classify(&rows),
        Err(ConstraintPlanError::ConflictingRow { .. })
    ));
}

#[test]
fn empty_zero_and_nonfinite_equations_remain_invalid() {
    let valid = equation("source", &[(1, 1.0)], 0.0);
    let mut empty = valid.clone();
    empty.terms.clear();
    assert!(classify(&[empty]).is_err());
    for coefficient in [0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut row = valid.clone();
        row.terms[0].coefficient = coefficient;
        assert!(classify(&[row]).is_err());
    }
    let mut mixed_nan = equation("mixed-nan", &[(1, 1.0), (2, 1.0)], 0.0);
    mixed_nan.terms[1].coefficient = f64::NAN;
    assert!(classify(&[mixed_nan]).is_err());
    for rhs in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut row = valid.clone();
        row.right_hand_side = rhs;
        assert!(classify(&[row]).is_err());
    }
}

#[test]
fn normalization_overflow_is_rejected() {
    let mut row = equation("overflow", &[(1, 1.0)], 1.0);
    row.terms[0].coefficient = f64::MIN_POSITIVE;
    row.right_hand_side = f64::MAX;
    assert!(classify(&[row]).is_err());
}

#[test]
fn reservation_failure_and_capacity_exhaustion_return_errors() {
    assert!(reserved::<DofKey>(usize::MAX).is_err());
    let mut values = reserved(1).unwrap();
    bounded_insert(&mut values, 0, 42usize, 1).unwrap();
    assert!(bounded_insert(&mut values, 1, 43, 1).is_err());
    assert_eq!(values, [42]);
}
