//! Explicit affine-geometry eligibility; no physical coefficient assembly or solver.
use super::{topology, MaxwellError};
use crate::topology::{AnalysisProjection, TetrahedralEdgeSpace};
use runmat_meshing_core::{BoundaryFaceRole, ElementOrder, StableDigest};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn spaces(
    projection: &AnalysisProjection,
) -> Result<TetrahedralEdgeSpace, MaxwellError> {
    let length = projection.characteristic_length_m;
    if !length.is_finite() || length <= 0.0 || projection.elements.is_empty() {
        return Err(topology("empty or invalid Maxwell geometry"));
    }
    for element in &projection.elements {
        if element.order != ElementOrder::Tet10 {
            continue;
        }
        for (midside, [a, b]) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .into_iter()
            .enumerate()
        {
            let middle = projection.nodes[element.node_indices[4 + midside]]
                .coordinates_m
                .canonical_value();
            let left = projection.nodes[element.node_indices[a]]
                .coordinates_m
                .canonical_value();
            let right = projection.nodes[element.node_indices[b]]
                .coordinates_m
                .canonical_value();
            if (0..3).any(|axis| {
                (middle[axis] - 0.5 * (left[axis] + right[axis])).abs()
                    > 64.0 * f64::EPSILON * length
            }) {
                return Err(MaxwellError::Unsupported(
                    "curved Maxwell form quadrature is not qualified; Tet10 must be affine".into(),
                ));
            }
        }
    }
    let edges = TetrahedralEdgeSpace::derive(projection).map_err(topology)?;
    // Validates the exact exterior inventory once for every Maxwell consumer.
    edges.exterior_edge_indices(projection).map_err(topology)?;
    Ok(edges)
}

/// Edges with tangential E = 0: the three corner edges of every grounded exterior face.
pub(super) fn grounded_edges(
    projection: &AnalysisProjection,
    edges: &TetrahedralEdgeSpace,
    grounded_faces: &[StableDigest],
) -> Result<BTreeSet<usize>, MaxwellError> {
    let by_nodes: BTreeMap<[usize; 2], usize> = edges
        .edges()
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            let [a, b] = edge.node_indices;
            ([a.min(b), a.max(b)], index)
        })
        .collect();
    let wanted: BTreeSet<_> = grounded_faces.iter().copied().collect();
    let mut found = BTreeSet::new();
    let mut grounded = BTreeSet::new();
    for face in &projection.boundary_faces {
        if !wanted.contains(&face.stable_identity) {
            continue;
        }
        if face.role != BoundaryFaceRole::Exterior {
            return Err(MaxwellError::Unsupported(
                "vector-potential ground on an internal face".into(),
            ));
        }
        found.insert(face.stable_identity);
        let corners: [usize; 3] = face
            .node_indices
            .get(..3)
            .and_then(|nodes| nodes.try_into().ok())
            .ok_or_else(|| topology("invalid grounded face arity"))?;
        for [a, b] in [[0, 1], [0, 2], [1, 2]] {
            let key = [corners[a].min(corners[b]), corners[a].max(corners[b])];
            grounded.insert(
                *by_nodes
                    .get(&key)
                    .ok_or_else(|| topology("grounded face edge absent from edge space"))?,
            );
        }
    }
    if found != wanted {
        return Err(topology("grounded face absent from canonical boundary"));
    }
    Ok(grounded)
}

/// Corner nodes lying on grounded faces (their gradients are not admissible test fields).
pub(super) fn grounded_nodes(
    projection: &AnalysisProjection,
    grounded_faces: &[StableDigest],
) -> BTreeSet<usize> {
    let wanted: BTreeSet<_> = grounded_faces.iter().copied().collect();
    projection
        .boundary_faces
        .iter()
        .filter(|face| wanted.contains(&face.stable_identity))
        .flat_map(|face| face.node_indices.iter().take(3).copied())
        .collect()
}
