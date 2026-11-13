//! Component inventory after lowering, before sorting/classification/validation/encoding.
use super::*;
use crate::constraints::{
    rank,
    workspace::{add, check_cancelled, ordering_workspace_bytes},
};

/// Requested finalization storage for an existing lowered plan. Connectivity profile
/// derivation itself allocates counted graph scratch; its public count-only bound must
/// be admitted first. Borrowed plan and lowering/topology storage remain separate.
/// Codec values are the shared estimate; allocator bookkeeping/rounding and RSS excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintPlanFinalizationWorkspaceProfile {
    pub ordering_bytes: u64,
    pub graph_preflight_bytes: u64,
    pub rank_bytes: u64,
    pub canonical_materialization_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_plan_finalization_workspace_profile(
    plan: &CanonicalStructuralConstraintPlan,
) -> Result<ConstraintPlanFinalizationWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let ordering_bytes = add(
        ordering_workspace_bytes(plan.groups.len())?,
        ordering_workspace_bytes(plan.rows.len())?,
    )?;
    let graph_preflight_bytes = rank::constraint_rank_graph_workspace_bytes(&plan.rows)?;
    let rank_bytes = rank::constraint_rank_workspace_profile(&plan.rows)?.total_requested_bytes;
    let canonical_materialization_bytes = canonical_value_workspace_bytes(plan)?;
    let total_requested_bytes = add(
        add(ordering_bytes, graph_preflight_bytes)?,
        add(rank_bytes, canonical_materialization_bytes)?,
    )?;
    Ok(ConstraintPlanFinalizationWorkspaceProfile {
        ordering_bytes,
        graph_preflight_bytes,
        rank_bytes,
        canonical_materialization_bytes,
        total_requested_bytes,
    })
}

#[cfg(test)]
mod tests;
