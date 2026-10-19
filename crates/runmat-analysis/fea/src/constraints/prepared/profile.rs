//! Requested selection construction overlap before prepared occurrence buffers.
use super::super::workspace::{add, slots};
use super::*;
use crate::physics::structural::topology::{
    explicit_target_workspace_profile, target::mixed_target_workspace_profile,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::constraints) struct PreparedTargetWorkspaceProfile {
    pub target_occurrences: usize,
    pub occurrence_bytes: u64,
    pub selection_bytes: u64,
    pub total_requested_bytes: u64,
}

pub(in crate::constraints) fn derive(
    constraint: &StructuralConstraint,
    mesh: Option<&SolverMeshArtifact>,
    structural: Option<&StructuralModel>,
) -> Result<PreparedTargetWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let mut target_occurrences = 0;
    let mut selection_bytes = 0;
    visit_targets(&constraint.kind, |target| {
        target_occurrences = super::super::workspace::count_add(target_occurrences, 1)?;
        if let Some(structural) = structural {
            let (selection_workspace, explicit_candidates) = if let Some(mesh) = mesh {
                let profile = mixed_target_workspace_profile(target, mesh, structural)?;
                (
                    profile.total_requested_bytes,
                    profile.explicit_candidate_slots,
                )
            } else {
                let profile = explicit_target_workspace_profile(structural, &target.region_id)?;
                (profile.total_requested_bytes, profile.candidate_slots)
            };
            selection_bytes = add(selection_bytes, selection_workspace)?;
            selection_bytes = add(
                selection_bytes,
                super::super::target::identity::explicit_workspace(target, explicit_candidates)?,
            )?;
            selection_bytes = add(
                selection_bytes,
                slots::<runmat_meshing_core::StableDigest>(explicit_candidates)?,
            )?;
            selection_bytes = add(
                selection_bytes,
                super::super::target::profile::authored_target_metadata_bytes(target)?,
            )?;
        }
        if let Some(mesh) = mesh {
            // Only supported solid selector shapes can reach the solid resolver. Other
            // shapes may still select explicit connectivity; ownership decides that later.
            use runmat_analysis_core::{EntityDimension, ExpectedTopology};
            if matches!(
                (target.entity_dimension, target.expected_topology),
                (EntityDimension::Face, ExpectedTopology::BoundaryFace)
                    | (EntityDimension::Point, ExpectedTopology::MeshNode)
            ) {
                selection_bytes = add(
                    selection_bytes,
                    super::super::target::profile::derive(mesh, target, &constraint.constraint_id)?
                        .total_requested_bytes,
                )?;
            }
        }
        Ok(())
    })?;
    let occurrence_bytes = slots::<(&RegionTarget, PreparedTarget)>(target_occurrences)?;
    let total_requested_bytes = add(occurrence_bytes, selection_bytes)?;
    usize::try_from(total_requested_bytes).map_err(|_| super::super::workspace::overflow())?;
    Ok(PreparedTargetWorkspaceProfile {
        target_occurrences,
        occurrence_bytes,
        selection_bytes,
        total_requested_bytes,
    })
}
