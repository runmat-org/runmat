use crate::topology::storage::CanonicalNodeValues as NodeValues;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::vector::{add, magnitude, scale};
use super::{workspace::check_cancelled, ConstraintPlanError, ResolvedConstraintTarget};
use crate::topology::AnalysisProjection;
use runmat_analysis_core::AnalysisModel;
use runmat_simulation_usage::{LedgerError, OperationCount};

/// Upper bound for projection batches in one canonical constraint compilation.
/// Solid lowering derives one projection for each authored normal-dependent
/// constraint. Explicit lowering derives none; mixed explicit-owned targets may
/// use fewer than this bound, and their unused grant is never completed.
pub fn structural_constraint_normal_projection_work_upper_bound(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<OperationCount, ConstraintPlanError> {
    let Some(mesh) = mesh else {
        return Ok(OperationCount::ZERO);
    };
    let repetitions = model
        .constraints
        .iter()
        .filter_map(|constraint| constraint.structural())
        .filter(|constraint| super::compile::invokes_boundary_node_normals(constraint))
        .try_fold(0u64, |count, _| {
            count
                .checked_add(1)
                .ok_or(ConstraintPlanError::Accounting(LedgerError::CountOverflow))
        })?;
    if repetitions == 0 {
        return Ok(OperationCount::ZERO);
    }
    let per_projection =
        AnalysisProjection::derivation_work_upper_bound(mesh).map_err(|error| match error {
            crate::topology::AnalysisProjectionError::Accounting(cause) => {
                ConstraintPlanError::Accounting(cause)
            }
            crate::topology::AnalysisProjectionError::Cancelled => ConstraintPlanError::Cancelled,
            other => ConstraintPlanError::Invalid(other.to_string()),
        })?;
    per_projection
        .checked_mul(repetitions)
        .map_err(|overflow| ConstraintPlanError::Accounting(LedgerError::from(overflow)))
}

pub(super) fn boundary_node_normals(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<NodeValues<[f64; 3]>, ConstraintPlanError> {
    let [entity] = target.matched_entities.as_slice() else {
        return Err(ConstraintPlanError::AmbiguousNormal {
            constraint_id: constraint_id.into(),
            entity_count: target.matched_entities.len(),
        });
    };
    let unavailable = || ConstraintPlanError::NormalUnavailable {
        constraint_id: constraint_id.into(),
    };
    check_cancelled()?;
    let _profile = profile::constraint_normal_workspace_profile(mesh, target)?;
    let projection = AnalysisProjection::derive(mesh).map_err(|error| match error {
        crate::topology::AnalysisProjectionError::Cancelled => ConstraintPlanError::Cancelled,
        crate::topology::AnalysisProjectionError::Accounting(error) => {
            ConstraintPlanError::Accounting(error)
        }
        _ => unavailable(),
    })?;
    let mut accumulated = NodeValues::derive(
        target
            .matched_node_identities
            .iter()
            .map(|id| (*id, [0.0; 3])),
        target.matched_node_identities.len(),
    )?;
    for face in projection
        .boundary_faces
        .iter()
        .filter(|face| face.provenance.contains(entity))
    {
        check_cancelled()?;
        let vectors = projection
            .boundary_nodal_area_vectors(face)
            .map_err(|_| unavailable())?;
        for (&node_index, vector) in face.node_indices.iter().zip(vectors) {
            let identity = projection.nodes[node_index].stable_identity;
            let normal = accumulated.get_mut(&identity).ok_or_else(unavailable)?;
            *normal = add(*normal, vector);
        }
    }
    for identity in &target.matched_node_identities {
        check_cancelled()?;
        let vector = accumulated.get_mut(identity).ok_or_else(unavailable)?;
        let length = magnitude(*vector);
        if !length.is_finite() || length <= 1.0e-12 {
            return Err(unavailable());
        }
        *vector = scale(*vector, length.recip());
    }
    Ok(accumulated)
}
mod profile;
pub use profile::{constraint_normal_workspace_profile, ConstraintNormalWorkspaceProfile};
