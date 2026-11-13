use runmat_meshing_core::{quality::predicate::PredicateSign, MeshingCancellationSignal};

use super::{
    cavity::stable_face, error, predicate_error, topology_error, validate_options,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeNode, DelaunayVolumeTopology, InsertionWork,
};
use crate::cdt::topology::insphere_materialized_volume_nodes as insphere_volume_nodes;
use crate::cdt::topology::validate_canonical_volume_topology;

pub fn validate_delaunay_volume_topology(
    topology: &DelaunayVolumeTopology,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_topology(topology, &[], false, true, options, cancellation)
}

pub(in crate::cdt) fn validate_constrained_delaunay_volume_topology(
    topology: &DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_topology(topology, protected_faces, true, true, options, cancellation)
}

/// Validates canonical positive, manifold volume topology and every protected
/// boundary/interface face without requiring the optimized interior to remain
/// Delaunay. Quality reconnection intentionally leaves the Delaunay family.
pub(in crate::cdt) fn validate_conforming_volume_topology(
    topology: &DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_topology(
        topology,
        protected_faces,
        true,
        false,
        options,
        cancellation,
    )
}

pub(in crate::cdt) fn validate_locally_delaunay_volume_topology(
    topology: &DelaunayVolumeTopology,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_topology(topology, &[], true, true, options, cancellation)
}

fn validate_topology(
    topology: &DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    protect_region_boundaries: bool,
    enforce_delaunay: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    validate_canonical_volume_topology(topology, options.topology, cancellation)
        .map_err(topology_error)?;

    validate_boundary_and_spheres(
        topology,
        protected_faces,
        protect_region_boundaries,
        enforce_delaunay,
        options,
        cancellation,
    )
}

/// Recheck the current PLC and region contract against immutable construction authority.
/// The token does not authorize any protected face or metric policy by itself.
#[cfg(test)]
pub(in crate::cdt) fn validate_canonical_conforming_volume_topology(
    topology: &crate::cdt::topology::CanonicalVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    if cancellation.is_cancelled() {
        return Err(error(DelaunayInsertionErrorKind::Cancelled, "cancelled"));
    }

    if topology.nodes.len() as u64 > options.topology.maximum_nodes
        || topology.tetrahedra.len() as u64 > options.topology.maximum_tetrahedra
    {
        return Err(error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "canonical topology exceeds current inventory limits",
        ));
    }
    validate_boundary_and_spheres(
        topology,
        protected_faces,
        true,
        false,
        options,
        cancellation,
    )
}

fn validate_boundary_and_spheres(
    topology: &DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    protect_region_boundaries: bool,
    enforce_delaunay: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    if protected_faces.len() as u64 > options.maximum_protected_faces {
        return Err(error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "protected face inventory exceeds its hard limit",
        ));
    }
    let mut work = InsertionWork::new(options, cancellation);
    if protected_faces.iter().enumerate().any(|(index, face)| {
        face.contains(&runmat_meshing_core::StableDigest::ZERO)
            || face.windows(2).any(|pair| pair[0] >= pair[1])
            || index > 0 && protected_faces[index - 1] >= *face
    }) {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "protected faces must be canonical, unique, ordered, and present",
        ));
    }
    for face in protected_faces {
        work.checkpoint()?;
        if !face_exists(topology, *face, &mut work)? {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "protected face is absent from the admitted topology",
            ));
        }
    }
    let protected_faces = protected_faces
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();

    if enforce_delaunay && protected_faces.is_empty() && !protect_region_boundaries {
        validate_global_empty_spheres(topology, &mut work)?;
    }

    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for (opposite, neighbor_index) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor_index) = *neighbor_index else {
                continue;
            };
            if neighbor_index as usize <= tetrahedron_index {
                continue;
            }
            work.checkpoint()?;
            let neighbor = &topology.tetrahedra[neighbor_index as usize];
            let face = stable_face(topology, tetrahedron.vertex_indices, opposite);
            if protect_region_boundaries
                && neighbor.region_id != tetrahedron.region_id
                && !protected_faces.contains(&face)
            {
                return Err(error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "assigned region boundary is missing protected facet provenance",
                ));
            }
            if protected_faces.contains(&face)
                || protect_region_boundaries && neighbor.region_id != tetrahedron.region_id
            {
                continue;
            }
            let opposite = neighbor
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !tetrahedron.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "neighbors do not share exactly one triangular face",
                    )
                })?;
            // Empty-sphere admission is independent of the available flip.
            // A nonconvex two-tet cavity can require an edge-star (3-to-2)
            // repair; lack of a 2-to-3 edit does not establish Delaunay validity.
            if enforce_delaunay
                && in_circumsphere_exact(
                    topology,
                    tetrahedron.vertex_indices,
                    topology.nodes[opposite as usize],
                    &mut work,
                )?
            {
                let mut face_indices = [0_u32; 3];
                let mut cursor = 0;
                for vertex in tetrahedron.vertex_indices {
                    if neighbor.vertex_indices.contains(&vertex) {
                        face_indices[cursor] = vertex;
                        cursor += 1;
                    }
                }
                face_indices.sort_unstable();
                return Err(error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    format!(
                        "tetrahedron {tetrahedron_index} has neighbor {neighbor_index} across face \
                         {face_indices:?} strictly inside its circumsphere; tetrahedra are {:?} \
                         and {:?}; coordinates {:?} and opposite {:?}",
                        tetrahedron.vertex_indices,
                        neighbor.vertex_indices,
                        tetrahedron
                            .vertex_indices
                            .map(|index| topology.nodes[index as usize].coordinates_m.canonical_value()),
                        topology.nodes[opposite as usize].coordinates_m.canonical_value(),
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_global_empty_spheres(
    topology: &DelaunayVolumeTopology,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for (node_index, node) in topology.nodes.iter().copied().enumerate() {
            work.checkpoint()?;
            if tetrahedron.vertex_indices.contains(&(node_index as u32)) {
                continue;
            }
            if in_circumsphere_exact(topology, tetrahedron.vertex_indices, node, work)? {
                return Err(error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    format!(
                        "node {node_index} is strictly inside the circumsphere of tetrahedron \
                         {tetrahedron_index} {:?}",
                        tetrahedron.vertex_indices,
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn face_exists(
    topology: &DelaunayVolumeTopology,
    identities: [runmat_meshing_core::StableDigest; 3],
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    let indices = identities.map(|identity| {
        topology
            .nodes
            .binary_search_by_key(&identity, |node| node.identity)
            .ok()
    });
    let [Some(first), Some(second), Some(third)] = indices else {
        return Ok(false);
    };
    for tetrahedron_index in &topology.incidence.vertex_stars[first] {
        work.checkpoint()?;
        let tetrahedron = &topology.tetrahedra[*tetrahedron_index as usize];
        if tetrahedron.vertex_indices.contains(&(second as u32))
            && tetrahedron.vertex_indices.contains(&(third as u32))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn in_circumsphere_exact(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let points = vertices.map(|vertex| topology.nodes[vertex as usize]);
    insphere_volume_nodes([points[0], points[1], points[2], points[3], node])
        .map(|sign| sign == PredicateSign::Positive)
        .map_err(predicate_error)
}
