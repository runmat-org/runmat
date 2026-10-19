//! Exact exterior incidence shared by curl-conforming physical formulations.
use super::{AnalysisProjection, AnalysisProjectionError, TetrahedralEdgeSpace};
use runmat_meshing_core::BoundaryFaceRole;
use std::collections::{BTreeMap, BTreeSet};

const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

pub(super) fn exterior_edge_indices(
    space: &TetrahedralEdgeSpace,
    projection: &AnalysisProjection,
) -> Result<Vec<usize>, AnalysisProjectionError> {
    let invalid = AnalysisProjectionError::InvalidEdgeSpace;
    if space.projection_digest != projection.projection_digest
        || space.node_count != projection.nodes.len()
        || space.elements.len() != projection.elements.len()
        || space.elements.is_empty()
    {
        return Err(invalid("exterior projection identity mismatch"));
    }
    for edge in &space.edges {
        for (index, identity) in edge.node_indices.into_iter().zip(edge.node_identities) {
            if projection.nodes.get(index).map(|node| node.stable_identity) != Some(identity) {
                return Err(invalid("exterior edge node identity mismatch"));
            }
        }
    }
    let mut incidence = BTreeMap::<[usize; 3], Vec<usize>>::new();
    for (index, (stored, element)) in space.elements.iter().zip(&projection.elements).enumerate() {
        if stored.identity != element.stable_identity
            || element.node_indices.get(..4) != Some(stored.corners.as_slice())
        {
            return Err(invalid("exterior element identity mismatch"));
        }
        for local in FACES {
            let mut corners = local.map(|i| stored.corners[i]);
            corners.sort_unstable();
            let owners = incidence.entry(corners).or_default();
            owners.push(index);
            if owners.len() > 2 {
                return Err(invalid("nonmanifold tetrahedral face incidence"));
            }
        }
    }
    let exterior: BTreeSet<_> = incidence
        .iter()
        .filter(|(_, owners)| owners.len() == 1)
        .map(|(&corners, _)| corners)
        .collect();
    let mut declared = BTreeSet::new();
    let mut identities = BTreeSet::new();
    for face in projection
        .boundary_faces
        .iter()
        .filter(|face| face.role == BoundaryFaceRole::Exterior)
    {
        let corners: [usize; 3] = face
            .node_indices
            .get(..3)
            .and_then(|nodes| nodes.try_into().ok())
            .ok_or(invalid("invalid exterior face arity"))?;
        let mut key = corners;
        key.sort_unstable();
        if !declared.insert(key) || !identities.insert(face.stable_identity) {
            return Err(invalid("duplicate exterior face"));
        }
        let owners = incidence
            .get(&key)
            .ok_or(invalid("foreign exterior face"))?;
        if owners.len() != 1 || face.adjacent_element_indices.as_slice() != owners.as_slice() {
            return Err(invalid("invalid exterior face ownership"));
        }
    }
    if declared != exterior {
        return Err(invalid("incomplete exact exterior inventory"));
    }
    let mut exterior_edges = BTreeSet::new();
    for [a, b, c] in exterior {
        exterior_edges.extend([[a, b], [a, c], [b, c]]);
    }
    Ok(space
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            let mut key = edge.node_indices;
            key.sort_unstable();
            exterior_edges.contains(&key).then_some(index)
        })
        .collect())
}

#[cfg(test)]
mod tests;
