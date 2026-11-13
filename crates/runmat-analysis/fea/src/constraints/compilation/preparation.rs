//! Whole preparation inventory before authored/topology/reference/selection allocation.
//! Lowered geometry, rows, accumulated evidence and final rank remain additional phases.
use super::topology::CompilationInputs;
use super::*;
use crate::constraints::{
    compile, plan, prepared,
    workspace::{add, check_cancelled},
};

/// Requested preparation inventory composed with the topology owner's existing estimate.
/// Excludes borrowed inputs, lowering, allocator overhead and measured RSS.
/// Sums transient inventories with retained buffers
/// even where lifetimes are disjoint; this is not whole-compilation policy admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintPreparationWorkspaceProfile {
    pub authored_bytes: u64,
    pub topology_bytes: u64,
    pub reference_point_bytes: u64,
    pub largest_target_preparation_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_preparation_workspace_profile(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<ConstraintPreparationWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let inputs = CompilationInputs::derive(model, mesh)?;
    let authored_bytes = plan::constraint_authored_workspace_profile(model)?.total_requested_bytes;
    let topology_bytes = inputs.workspace_bytes()?;
    let reference_point_bytes =
        compile::constraint_reference_point_workspace_profile(model)?.total_requested_bytes;
    let mut largest_target_preparation_bytes = 0;
    for constraint in &model.constraints {
        check_cancelled()?;
        let Some(constraint) = constraint.structural() else {
            continue;
        };
        largest_target_preparation_bytes = largest_target_preparation_bytes.max(
            prepared::profile::derive(constraint, inputs.mesh(), inputs.structural())?
                .total_requested_bytes,
        );
    }
    let total_requested_bytes = [
        authored_bytes,
        topology_bytes,
        reference_point_bytes,
        largest_target_preparation_bytes,
    ]
    .into_iter()
    .try_fold(0, add)?;
    usize::try_from(total_requested_bytes)
        .map_err(|_| crate::constraints::workspace::overflow())?;
    check_cancelled()?;
    Ok(ConstraintPreparationWorkspaceProfile {
        authored_bytes,
        topology_bytes,
        reference_point_bytes,
        largest_target_preparation_bytes,
        total_requested_bytes,
    })
}

#[cfg(test)]
mod tests;
