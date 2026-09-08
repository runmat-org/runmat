//! Canonical sparse constraint-row construction, ordering, and rank classification.

use std::cmp::Ordering;

use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use super::contract::{
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintEnforcement, ConstraintPlanError, ReactionRecovery,
};
use super::{plan::canonical_digest, rank};

const ROW_DOMAIN: &str = "structural-constraint-row";

pub(super) fn new_singleton(
    constraint_id: &str,
    node_identity: StableDigest,
    dof: CanonicalStructuralDof,
    coefficient: f64,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    new_sparse(
        constraint_id,
        vec![CanonicalStructuralConstraintTerm {
            owner: CanonicalStructuralDofOwner::MeshNode(node_identity),
            dof,
            coefficient,
        }],
        right_hand_side,
    )
}

pub(super) fn new_direction(
    constraint_id: &str,
    node_identity: StableDigest,
    direction: [f64; 3],
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    let terms = CanonicalStructuralDof::TRANSLATIONS
        .into_iter()
        .zip(direction)
        .filter(|(_, coefficient)| *coefficient != 0.0)
        .map(|(dof, coefficient)| CanonicalStructuralConstraintTerm {
            owner: CanonicalStructuralDofOwner::MeshNode(node_identity),
            dof,
            coefficient,
        })
        .collect();
    new_sparse(constraint_id, terms, right_hand_side)
}

pub(super) fn new_sparse(
    constraint_id: &str,
    mut terms: Vec<CanonicalStructuralConstraintTerm>,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    terms.sort_by_key(term_key);
    let mut combined: Vec<CanonicalStructuralConstraintTerm> = Vec::with_capacity(terms.len());
    for term in terms {
        if let Some(previous) = combined
            .last_mut()
            .filter(|previous| term_key(previous) == term_key(&term))
        {
            previous.coefficient += term.coefficient;
        } else {
            combined.push(term);
        }
    }
    combined.retain(|term| term.coefficient != 0.0);
    if combined.is_empty() || !right_hand_side.is_finite() {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` compiles to an empty or non-finite equation"
        )));
    }
    let reaction_recovery = if combined.len() == 1 {
        ReactionRecovery::UnconstrainedEquilibriumResidual
    } else {
        ReactionRecovery::ConstraintMultiplier
    };
    let mut row = CanonicalStructuralConstraintRow {
        identity: Digest::from_bytes([0; 32]),
        constraint_id: constraint_id.into(),
        terms: combined,
        right_hand_side,
        reaction_recovery,
    };
    row.identity = row_identity(&row)?;
    Ok(row)
}

pub(super) fn row_identity(
    row: &CanonicalStructuralConstraintRow,
) -> Result<Digest, ConstraintPlanError> {
    canonical_digest(
        ROW_DOMAIN,
        &(
            &row.constraint_id,
            &row.terms,
            row.right_hand_side,
            row.reaction_recovery,
        ),
    )
}

pub(super) fn term_key(
    term: &CanonicalStructuralConstraintTerm,
) -> (CanonicalStructuralDofOwner, CanonicalStructuralDof) {
    (term.owner, term.dof)
}

pub(super) fn compare_rows(
    left: &CanonicalStructuralConstraintRow,
    right: &CanonicalStructuralConstraintRow,
) -> Ordering {
    left.constraint_id
        .cmp(&right.constraint_id)
        .then_with(|| {
            for (left, right) in left.terms.iter().zip(&right.terms) {
                let ordering = term_key(left)
                    .cmp(&term_key(right))
                    .then_with(|| left.coefficient.total_cmp(&right.coefficient));
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            left.terms.len().cmp(&right.terms.len())
        })
        .then_with(|| left.right_hand_side.total_cmp(&right.right_hand_side))
}

pub(super) fn classify(
    rows: &mut [CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let result = rank::classify(rows)?;
    for row in rows {
        row.identity = row_identity(row)?;
    }
    Ok(result)
}

pub(super) fn validate_row_classification(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let mut expected = rows
        .iter()
        .map(|row| {
            let mut row = row.clone();
            row.reaction_recovery = if row.terms.len() == 1 {
                ReactionRecovery::UnconstrainedEquilibriumResidual
            } else {
                ReactionRecovery::ConstraintMultiplier
            };
            row
        })
        .collect::<Vec<_>>();
    let result = rank::classify(&mut expected)?;
    for (actual, expected) in rows.iter().zip(expected) {
        if actual.reaction_recovery != expected.reaction_recovery {
            return Err(ConstraintPlanError::Invalid(
                "constraint row rank classification is inconsistent".into(),
            ));
        }
    }
    Ok(result)
}

pub(super) fn enforcement_matches_rows(
    enforcement: ConstraintEnforcement,
    rows: &[&CanonicalStructuralConstraintRow],
) -> bool {
    match enforcement {
        ConstraintEnforcement::DirectElimination => {
            !rows.is_empty() && rows.iter().all(|row| row.terms.len() == 1)
        }
        ConstraintEnforcement::AffineReduction => {
            !rows.is_empty() && rows.iter().any(|row| row.terms.len() > 1)
        }
        ConstraintEnforcement::LagrangeMultiplier | ConstraintEnforcement::NullSpaceReduction => {
            false
        }
    }
}
