//! Additional row finalization storage, excluding the incoming term allocation.
use super::*;
use crate::constraints::workspace::{add, check_cancelled, ordering_workspace_bytes, slots};

/// Requested additional slots plus the shared canonical materialization estimate.
/// The borrowed incoming term buffer/capacity is already owned by lowering. Compaction
/// reserves a possible replacement buffer; equal-term summation can only shrink it.
/// Excludes allocator bookkeeping/rounding, borrowed terms, error strings and RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintRowWorkspaceProfile {
    pub ordering_bytes: u64,
    pub compaction_bytes: u64,
    pub id_copy_bytes: u64,
    pub identity_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_row_workspace_profile(
    id: &str,
    terms: &[CanonicalStructuralConstraintTerm],
    rhs: f64,
) -> Result<ConstraintRowWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let ordering_bytes = ordering_workspace_bytes(terms.len())?;
    let compaction_bytes = slots::<CanonicalStructuralConstraintTerm>(terms.len())?;
    // The longest enum spellings bound every post-compaction identity shape, even
    // when coefficient combination removes all translations or leaves a singleton equation.
    let identity_materialization_bytes = super::super::plan::canonical_value_workspace_bytes(&(
        id,
        terms,
        rhs,
        ConstraintResidualQuantity::Displacement,
        ReactionRecovery::UnconstrainedEquilibriumResidual,
    ))?;
    let id_copy_bytes = id.len() as u64;
    let total_requested_bytes = add(
        add(ordering_bytes, compaction_bytes)?,
        add(id_copy_bytes, identity_materialization_bytes)?,
    )?;
    Ok(ConstraintRowWorkspaceProfile {
        ordering_bytes,
        compaction_bytes,
        id_copy_bytes,
        identity_materialization_bytes,
        total_requested_bytes,
    })
}
