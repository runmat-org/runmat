use runmat_meshing_core::{
    quality::predicate::{insphere3d, PredicateSign},
    MeshingCancellationSignal,
};

use super::{
    cavity::stable_face, error, predicate_error, topology_error, validate_options,
    DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
    DelaunayVolumeNode, DelaunayVolumeTopology, InsertionWork,
};
use crate::cdt::topology::build_delaunay_volume_topology_with_regions;

pub fn validate_delaunay_volume_topology(
    topology: &DelaunayVolumeTopology,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_topology(topology, &[], false, options, cancellation)
}

pub(in crate::cdt) fn validate_constrained_delaunay_volume_topology(
    topology: &DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_topology(topology, protected_faces, true, options, cancellation)
}

fn validate_topology(
    topology: &DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    protect_region_boundaries: bool,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    let rebuilt = build_delaunay_volume_topology_with_regions(
        topology.nodes.clone(),
        topology
            .tetrahedra
            .iter()
            .map(|tetrahedron| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
            .collect(),
        options.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    if rebuilt != *topology {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "tetrahedra or neighbor links are not in canonical checked form",
        ));
    }

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
            if in_circumsphere_exact(
                topology,
                tetrahedron.vertex_indices,
                topology.nodes[opposite as usize],
                &mut work,
            )? {
                return Err(error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    format!(
                        "tetrahedron {tetrahedron_index} has neighbor {neighbor_index} strictly inside its circumsphere"
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
    let points = vertices.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
    insphere3d([
        points[0],
        points[1],
        points[2],
        points[3],
        node.coordinates_m,
    ])
    .map(|sign| sign == PredicateSign::Positive)
    .map_err(predicate_error)
}
