//! Checked requested Rust-slot envelope for graph derivation and borrowed rank classification.
use super::{
    graph::{Component, Components},
    invalid, BasisRow, CanonicalStructuralConstraintRow, ConstraintPlanError, DofKey,
    ReactionRecovery,
};
use std::mem::size_of;

/// Additional requested heap slots, excluding borrowed equations/string payloads, allocator
/// bookkeeping or size-class rounding, RSS, and error-string construction. This is a workspace
/// profile, not a policy ceiling or a bound for constraint compilation or canonical encoding.
/// Graph scratch and numeric storage are summed conservatively even when their lifetimes differ.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintRankWorkspaceProfile {
    pub row_count: usize,
    pub candidate_term_count: usize,
    pub dof_count: usize,
    pub component_count: usize,
    pub basis_row_capacity: usize,
    pub basis_coefficient_slots: usize,
    pub basis_source_slots: usize,
    pub graph_bytes: u64,
    pub basis_workspace_bytes: u64,
    pub scratch_workspace_bytes: u64,
    pub classification_workspace_bytes: u64,
    pub total_requested_bytes: u64,
}

/// Count-only requested-slot bound for constructing the connectivity graph/profile itself.
/// Scans row/term counts without allocating, validating equations, or running elimination.
/// Uses distinct DOFs <= candidate terms, so this can exceed the profile's tighter graph_bytes.
/// Callers can admit this reservation before calling `constraint_rank_workspace_profile`.
pub fn constraint_rank_graph_workspace_bytes(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<u64, ConstraintPlanError> {
    let terms = rows
        .iter()
        .try_fold(0usize, |total, row| count_add(total, row.terms.len()))?;
    graph_slots(terms, terms, rows.len())
}

/// Derive connectivity using counted candidate/union-find storage, without elimination or row
/// cloning. Numeric fill and source provenance cannot cross the resulting components.
pub fn constraint_rank_workspace_profile(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<ConstraintRankWorkspaceProfile, ConstraintPlanError> {
    ConstraintRankWorkspaceProfile::from_components(&Components::derive(rows)?)
}

impl ConstraintRankWorkspaceProfile {
    pub(super) fn from_components(graph: &Components) -> Result<Self, ConstraintPlanError> {
        let graph_workspace_bytes = graph_slots(
            graph.candidate_count,
            graph.dof_count,
            graph.row_components.len(),
        )?;
        let mut basis_row_capacity = 0usize;
        let mut basis_coefficient_slots = 0usize;
        let mut basis_source_slots = 0usize;
        let mut scratch_workspace_bytes = 0;
        for component in &graph.components {
            let basis = component.rows.min(component.dofs);
            basis_row_capacity = count_add(basis_row_capacity, basis)?;
            basis_coefficient_slots = count_add(
                basis_coefficient_slots,
                count_multiply(basis, component.dofs)?,
            )?;
            basis_source_slots =
                count_add(basis_source_slots, count_multiply(basis, component.rows)?)?;
            let coefficients = slots::<(DofKey, f64)>(component.dofs)?;
            let sources = slots::<&str>(component.rows)?;
            // One current row plus one replacement buffer while bounded fill/provenance
            // growth copies the old buffer. Retained bases never grow during elimination.
            scratch_workspace_bytes = scratch_workspace_bytes
                .max(add(add(coefficients, sources)?, coefficients.max(sources))?);
        }
        let basis_workspace_bytes = add(
            slots::<Vec<BasisRow<'_>>>(graph.components.len())?,
            add(
                slots::<BasisRow<'_>>(basis_row_capacity)?,
                add(
                    slots::<(DofKey, f64)>(basis_coefficient_slots)?,
                    slots::<&str>(basis_source_slots)?,
                )?,
            )?,
        )?;
        let classification_workspace_bytes = slots::<ReactionRecovery>(graph.row_components.len())?;
        let workspace_bytes = add(
            graph_workspace_bytes,
            add(
                basis_workspace_bytes,
                add(scratch_workspace_bytes, classification_workspace_bytes)?,
            )?,
        )?;
        Ok(Self {
            row_count: graph.row_components.len(),
            candidate_term_count: graph.candidate_count,
            dof_count: graph.dof_count,
            component_count: graph.components.len(),
            basis_row_capacity,
            basis_coefficient_slots,
            basis_source_slots,
            graph_bytes: graph_workspace_bytes,
            basis_workspace_bytes,
            scratch_workspace_bytes,
            classification_workspace_bytes,
            total_requested_bytes: workspace_bytes,
        })
    }
}

fn graph_slots(terms: usize, dofs: usize, rows: usize) -> Result<u64, ConstraintPlanError> {
    add(
        slots::<DofKey>(terms)?,
        add(
            multiply(slots::<usize>(dofs)?, 2)?,
            add(slots::<Component>(dofs)?, slots::<usize>(rows)?)?,
        )?,
    )
}

fn slots<T>(count: usize) -> Result<u64, ConstraintPlanError> {
    u64::try_from(count_multiply(count, size_of::<T>())?).map_err(|_| overflow())
}
fn count_add(left: usize, right: usize) -> Result<usize, ConstraintPlanError> {
    left.checked_add(right).ok_or_else(overflow)
}
fn count_multiply(left: usize, right: usize) -> Result<usize, ConstraintPlanError> {
    left.checked_mul(right).ok_or_else(overflow)
}
fn add(left: u64, right: u64) -> Result<u64, ConstraintPlanError> {
    left.checked_add(right).ok_or_else(overflow)
}
fn multiply(left: u64, right: u64) -> Result<u64, ConstraintPlanError> {
    left.checked_mul(right).ok_or_else(overflow)
}
fn overflow() -> ConstraintPlanError {
    invalid("constraint rank workspace size overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_products_and_aggregates_reject_overflow() {
        assert!(slots::<DofKey>(usize::MAX).is_err());
        assert!(count_add(usize::MAX, 1).is_err());
        assert!(add(u64::MAX, 1).is_err());
        assert!(multiply(u64::MAX, 2).is_err());
        assert!(graph_slots(usize::MAX, usize::MAX, usize::MAX).is_err());
    }
}
