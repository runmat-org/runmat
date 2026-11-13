//! Original projection estimate plus counted normal accumulation/output and facet scratch.
use super::super::workspace::{add, check_cancelled, slots};
use super::*;

/// Additional requested estimate; borrows original mesh/target. Projection construction
/// uses the shared topology owner's estimate. Normal slots are one buffer reused
/// for accumulation and normalized output; other fields count facet vector scratch.
/// Excludes canonical target/row encoding, allocator overhead and RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintNormalWorkspaceProfile {
    pub projection_bytes: u64,
    pub normal_slots: usize,
    pub facet_coordinate_slots: usize,
    pub facet_vector_slots: usize,
    pub total_requested_bytes: u64,
}
pub fn constraint_normal_workspace_profile(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
) -> Result<ConstraintNormalWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let projection_bytes = AnalysisProjection::derivation_resident_bytes_estimate(mesh)
        .map_err(|e| ConstraintPlanError::Invalid(e.to_string()))?;
    let width = mesh
        .topology
        .boundary_faces
        .iter()
        .map(|f| f.node_ids.len())
        .max()
        .unwrap_or(0);
    // Normalization mutates the accumulation buffer in place; only one retained vector.
    // Coordinate, nodal-vector and orientation-copy buffers are summed conservatively.
    let vector_slots = super::super::workspace::multiply(width, 2)?;
    let total = add(
        projection_bytes,
        add(
            slots::<(StableDigest, [f64; 3])>(target.matched_node_identities.len())?,
            add(slots::<[f64; 3]>(width)?, slots::<[f64; 3]>(vector_slots)?)?,
        )?,
    )?;
    Ok(ConstraintNormalWorkspaceProfile {
        projection_bytes,
        normal_slots: target.matched_node_identities.len(),
        facet_coordinate_slots: width,
        facet_vector_slots: vector_slots,
        total_requested_bytes: total,
    })
}

#[cfg(test)]
mod tests;
