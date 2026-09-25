use std::collections::BTreeMap;

use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use super::vector::{add, magnitude, scale};
use super::{ConstraintPlanError, ResolvedConstraintTarget};
use crate::topology::AnalysisProjection;

pub(super) fn boundary_node_normals(
    mesh: &SolverMeshArtifact,
    target: &ResolvedConstraintTarget,
    constraint_id: &str,
) -> Result<BTreeMap<StableDigest, [f64; 3]>, ConstraintPlanError> {
    let [entity] = target.matched_entities.as_slice() else {
        return Err(ConstraintPlanError::AmbiguousNormal {
            constraint_id: constraint_id.into(),
            entity_count: target.matched_entities.len(),
        });
    };
    let unavailable = || ConstraintPlanError::NormalUnavailable {
        constraint_id: constraint_id.into(),
    };
    let projection = AnalysisProjection::derive(mesh).map_err(|_| unavailable())?;
    let mut accumulated = BTreeMap::<StableDigest, [f64; 3]>::new();
    for face in projection
        .boundary_faces
        .iter()
        .filter(|face| face.provenance.contains(entity))
    {
        let vectors = projection
            .boundary_nodal_area_vectors(face)
            .map_err(|_| unavailable())?;
        for (&node_index, vector) in face.node_indices.iter().zip(vectors) {
            let identity = projection.nodes[node_index].stable_identity;
            let normal = accumulated.entry(identity).or_insert([0.0; 3]);
            *normal = add(*normal, vector);
        }
    }
    target
        .matched_node_identities
        .iter()
        .map(|identity| {
            let vector = accumulated.get(identity).copied().ok_or_else(unavailable)?;
            let length = magnitude(vector);
            if !length.is_finite() || length <= 1.0e-12 {
                return Err(unavailable());
            }
            Ok((*identity, scale(vector, length.recip())))
        })
        .collect()
}
