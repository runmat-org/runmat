//! Pre-expansion requested-slot inventory for tie/periodic geometric pairing.
use super::super::workspace::{check_cancelled, multiply, overflow, slots};
use super::{
    BoundaryTriangleOrder, ConstraintPairingEvidence, ConstraintPairingWeight, ConstraintPlanError,
    PairingPolicy, ResolvedConstraintTarget, SolverMeshArtifact, StableDigest,
};

/// Additional Rust vector slots for constructing pairing evidence. Excludes borrowed
/// mesh/targets, allocator rounding/bookkeeping, strings, row construction/encoding,
/// transform stack scratch and RSS. Geometric best/replacement weight buffers overlap;
/// both are reserved in addition to retained pairing weights. No tree-node estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintPairingWorkspaceProfile {
    pub coordinate_index_slots: usize,
    pub identity_index_slots: usize,
    pub matching_used_slots: usize,
    pub pairing_slots: usize,
    pub maximum_weights_per_pairing: usize,
    pub retained_weight_slots: usize,
    pub candidate_weight_slots: usize,
    pub total_requested_bytes: u64,
}

/// Count before building indexes or pairing evidence. Does not validate or materialize
/// target resolution; consumers must separately admit those workspaces and row lowering.
pub fn constraint_pairing_workspace_profile(
    mesh: &SolverMeshArtifact,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
) -> Result<ConstraintPairingWorkspaceProfile, ConstraintPlanError> {
    derive(mesh, mesh.topology.nodes.len(), primary, secondary, policy)
}

pub(in crate::constraints) fn derive(
    mesh: &SolverMeshArtifact,
    node_count: usize,
    primary: &ResolvedConstraintTarget,
    secondary: &ResolvedConstraintTarget,
    policy: &PairingPolicy,
) -> Result<ConstraintPairingWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let matching = matches!(policy, PairingPolicy::Matching);
    let max_weights = if matching {
        1
    } else {
        let mut maximum = 0;
        for face in &mesh.topology.boundary_faces {
            check_cancelled()?;
            if face
                .provenance
                .iter()
                .any(|entity| primary.matched_entities.contains(entity))
            {
                maximum = maximum.max(facet_node_count(face.order));
            }
        }
        maximum
    };
    let profile = ConstraintPairingWorkspaceProfile {
        coordinate_index_slots: node_count,
        identity_index_slots: if matching {
            0
        } else {
            mesh.topology.nodes.len()
        },
        matching_used_slots: if matching {
            primary.matched_node_identities.len()
        } else {
            0
        },
        pairing_slots: secondary.matched_node_identities.len(),
        maximum_weights_per_pairing: max_weights,
        retained_weight_slots: multiply(secondary.matched_node_identities.len(), max_weights)?,
        candidate_weight_slots: if matching {
            0
        } else {
            multiply(max_weights, 2)?
        },
        total_requested_bytes: 0,
    };
    let total_requested_bytes = [
        slots::<(StableDigest, [f64; 3])>(profile.coordinate_index_slots)?,
        slots::<(u64, StableDigest)>(profile.identity_index_slots)?,
        slots::<StableDigest>(profile.matching_used_slots)?,
        slots::<ConstraintPairingEvidence>(profile.pairing_slots)?,
        slots::<ConstraintPairingWeight>(profile.retained_weight_slots)?,
        slots::<ConstraintPairingWeight>(profile.candidate_weight_slots)?,
    ]
    .into_iter()
    .try_fold(0u64, |sum, bytes| {
        sum.checked_add(bytes).ok_or_else(overflow)
    })?;
    Ok(ConstraintPairingWorkspaceProfile {
        total_requested_bytes,
        ..profile
    })
}

pub(super) fn facet_node_count(order: BoundaryTriangleOrder) -> usize {
    match order {
        BoundaryTriangleOrder::Tri3 => 3,
        BoundaryTriangleOrder::Tri6 => 6,
    }
}
pub(super) fn row_term_count(
    pairing: &ConstraintPairingEvidence,
    direction: [f64; 3],
) -> Result<usize, ConstraintPlanError> {
    multiply(
        pairing.primary_weights.len(),
        direction.iter().filter(|v| **v != 0.0).count(),
    )?
    .checked_add(1)
    .ok_or_else(overflow)
}

#[cfg(test)]
mod tests;
