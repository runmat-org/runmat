//! Borrowed canonical topology verification. Reconstruct only compact face uses,
//! never a second mesh, its region strings, or its full incidence inventory.
use super::*;
mod incidence;

pub(in crate::cdt) fn validate_canonical_volume_topology(
    topology: &DelaunayVolumeTopology,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayTopologyError> {
    validate_options(options)?;
    if topology.nodes.is_empty()
        || topology.nodes.len() as u64 > options.maximum_nodes
        || topology.tetrahedra.is_empty()
        || topology.tetrahedra.len() as u64 > options.maximum_tetrahedra
    {
        return Err(error(
            DelaunayTopologyErrorKind::ResourceLimit,
            "topology inventory is empty or exceeds its hard limit",
        ));
    }
    for (index, node) in topology.nodes.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        if node.identity == StableDigest::ZERO
            || node
                .coordinates_m
                .canonical_value()
                .iter()
                .any(|value| !value.is_finite())
            || index > 0 && topology.nodes[index - 1].identity >= node.identity
        {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidNode,
                "nodes must have finite coordinates and strictly ordered nonzero identities",
            ));
        }
    }
    let mut faces = Vec::with_capacity(topology.tetrahedra.len().saturating_mul(4));
    let mut previous = None;
    for (index, cell) in topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let mut key = cell.vertex_indices;
        key.sort_unstable();
        if key[3] as usize >= topology.nodes.len()
            || key.windows(2).any(|pair| pair[0] == pair[1])
            || previous.is_some_and(|old| old >= key)
        {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidTetrahedron,
                "tetrahedra must have distinct known vertices and unique canonical order",
            ));
        }
        previous = Some(key);
        let sign =
            orient_materialized_volume_nodes(key.map(|vertex| topology.nodes[vertex as usize]))
                .map_err(predicate_error)?;
        match sign {
            PredicateSign::Positive => {}
            PredicateSign::Negative => key.swap(0, 1),
            PredicateSign::Zero => {
                return Err(error(
                    DelaunayTopologyErrorKind::DegenerateTetrahedron,
                    "tetrahedron is coplanar at materialized solver coordinates",
                ))
            }
        }
        if key != cell.vertex_indices {
            return Err(error(
                DelaunayTopologyErrorKind::InvalidTetrahedron,
                "tetrahedron orientation is not canonical and positive",
            ));
        }
        for opposite in 0..4 {
            let mut face = [0; 3];
            let mut cursor = 0;
            for (slot, vertex) in cell.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            faces.push((face, index, opposite));
        }
    }
    faces.sort_unstable_by_key(|entry| entry.0);
    let mut start = 0;
    let mut boundary = 0;
    let mut groups = 0;
    while start < faces.len() {
        checkpoint(groups, options, cancellation)?;
        groups += 1;
        let end = start + faces[start..].partition_point(|entry| entry.0 == faces[start].0);
        match &faces[start..end] {
            [(face, cell, opposite)] => {
                if topology.tetrahedra[*cell].neighbors[*opposite].is_some() {
                    return Err(error(
                        DelaunayTopologyErrorKind::InvalidTetrahedron,
                        "boundary face advertises an absent neighbor",
                    ));
                }
                let expected = super::super::incidence::boundary_facet(
                    &topology.nodes,
                    &topology.tetrahedra[*cell],
                    *cell,
                    *opposite,
                )?;
                if topology.incidence.boundary_facets.get(boundary) != Some(&expected)
                    || expected.vertex_indices != *face
                {
                    return Err(error(
                        DelaunayTopologyErrorKind::InvalidTetrahedron,
                        "boundary incidence is incomplete or differs from canonical geometry",
                    ));
                }
                boundary += 1;
            }
            [(face, left, left_opposite), (_, right, right_opposite)] => {
                let left_cell = &topology.tetrahedra[*left];
                let right_cell = &topology.tetrahedra[*right];
                if left_cell.neighbors[*left_opposite] != Some(*right as u32)
                    || right_cell.neighbors[*right_opposite] != Some(*left as u32)
                {
                    return Err(error(
                        DelaunayTopologyErrorKind::InvalidTetrahedron,
                        "neighbor links differ from shared-face incidence",
                    ));
                }
                let face_nodes = face.map(|vertex| topology.nodes[vertex as usize]);
                let sign = |apex| {
                    orient_materialized_volume_nodes([
                        face_nodes[0],
                        face_nodes[1],
                        face_nodes[2],
                        topology.nodes[apex as usize],
                    ])
                    .map_err(predicate_error)
                };
                if sign(left_cell.vertex_indices[*left_opposite])?
                    == sign(right_cell.vertex_indices[*right_opposite])?
                {
                    return Err(error(
                        DelaunayTopologyErrorKind::InvalidTetrahedron,
                        "tetrahedra overlap across a shared face",
                    ));
                }
            }
            _ => {
                return Err(error(
                    DelaunayTopologyErrorKind::NonManifoldFace,
                    "more than two tetrahedra share a face",
                ))
            }
        }
        start = end;
    }
    if boundary != topology.incidence.boundary_facets.len() {
        return Err(error(
            DelaunayTopologyErrorKind::InvalidTetrahedron,
            "boundary incidence contains extra faces",
        ));
    }
    drop(faces);
    incidence::validate_incidence(topology, options, cancellation)
}

#[cfg(test)]
mod tests;
