//! Canonical sparse constraint-row construction, ordering, and rank classification.

use std::cmp::Ordering;

use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use super::contract::{
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, CanonicalStructuralDof,
    CanonicalStructuralDofOwner, ConstraintPlanError, ConstraintResidualQuantity, ReactionRecovery,
};
use super::{plan::canonical_digest, rank};

mod profile;
pub(super) use profile::from_term_count as row_workspace_from_term_count;
pub use profile::{constraint_row_workspace_profile, ConstraintRowWorkspaceProfile};

const ROW_DOMAIN: &str = "structural-constraint-row";

#[cfg(test)]
mod tests;

pub(super) fn new_singleton(
    constraint_id: &str,
    node_identity: StableDigest,
    dof: CanonicalStructuralDof,
    coefficient: f64,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    let mut terms = super::workspace::reserve(1)?;
    terms.push(CanonicalStructuralConstraintTerm {
        owner: CanonicalStructuralDofOwner::StructuralNode(node_identity),
        dof,
        coefficient,
    });
    new_sparse(constraint_id, terms, right_hand_side)
}

pub(super) fn new_direction(
    constraint_id: &str,
    node_identity: StableDigest,
    direction: [f64; 3],
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    let mut terms = super::workspace::reserve(
        direction
            .iter()
            .filter(|coefficient| **coefficient != 0.0)
            .count(),
    )?;
    terms.extend(
        CanonicalStructuralDof::TRANSLATIONS
            .into_iter()
            .zip(direction)
            .filter(|(_, coefficient)| *coefficient != 0.0)
            .map(|(dof, coefficient)| CanonicalStructuralConstraintTerm {
                owner: CanonicalStructuralDofOwner::StructuralNode(node_identity),
                dof,
                coefficient,
            }),
    );
    new_sparse(constraint_id, terms, right_hand_side)
}

pub(super) fn new_sparse(
    constraint_id: &str,
    mut terms: Vec<CanonicalStructuralConstraintTerm>,
    right_hand_side: f64,
) -> Result<CanonicalStructuralConstraintRow, ConstraintPlanError> {
    let _profile = constraint_row_workspace_profile(constraint_id, &terms, right_hand_side)?;
    super::workspace::sort_stable_by(&mut terms, |a, b| term_key(a).cmp(&term_key(b)))?;
    // Do not retain geometric growth slack from lowering across the whole plan. Boxing
    // removes excess capacity; exact-capacity inputs keep their allocation. A shrinking
    // allocator may move this buffer, so this is not a zero-allocation claim for spare capacity.
    let mut terms = terms.into_boxed_slice().into_vec();
    let mut written = 0;
    for read in 0..terms.len() {
        if written != 0 && term_key(&terms[written - 1]) == term_key(&terms[read]) {
            let coefficient = terms[read].coefficient;
            terms[written - 1].coefficient += coefficient;
        } else {
            terms.swap(written, read);
            written += 1;
        }
    }
    terms.truncate(written);
    terms.retain(|term| term.coefficient != 0.0);
    if terms.is_empty()
        || !right_hand_side.is_finite()
        || terms.iter().any(|term| !term.coefficient.is_finite())
    {
        return Err(ConstraintPlanError::Invalid(format!(
            "constraint `{constraint_id}` compiles to an empty or non-finite equation"
        )));
    }
    let reaction_recovery = if terms.len() == 1 {
        ReactionRecovery::UnconstrainedEquilibriumResidual
    } else {
        ReactionRecovery::ConstraintMultiplier
    };
    let residual_quantity = residual_quantity(&terms);
    let mut row = CanonicalStructuralConstraintRow {
        identity: Digest::from_bytes([0; 32]),
        constraint_id: super::lowering::owned_id(constraint_id)?,
        terms,
        right_hand_side,
        residual_quantity,
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
            row.residual_quantity,
            row.reaction_recovery,
        ),
    )
}

pub(super) fn residual_quantity(
    terms: &[CanonicalStructuralConstraintTerm],
) -> ConstraintResidualQuantity {
    if terms
        .iter()
        .all(|term| matches!(term.dof, CanonicalStructuralDof::Rotation(_)))
    {
        ConstraintResidualQuantity::Rotation
    } else {
        ConstraintResidualQuantity::Displacement
    }
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
    for (row, recovery) in rows.iter_mut().zip(result.classifications) {
        row.reaction_recovery = recovery;
        row.identity = row_identity(row)?;
    }
    Ok((result.active_row_count, result.redundant_row_count))
}

pub(super) fn validate_row_classification(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let result = rank::classify(rows)?;
    for (actual, expected) in rows.iter().zip(result.classifications) {
        if actual.reaction_recovery != expected {
            return Err(ConstraintPlanError::Invalid(
                "constraint row rank classification is inconsistent".into(),
            ));
        }
    }
    Ok((result.active_row_count, result.redundant_row_count))
}
