use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    StableDigest,
};

use super::{
    edge_exists, error, node_index, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DelaunayVolumeTopology, RecoveryWork,
};
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, validate_delaunay_volume_topology,
    DelaunayInsertionErrorKind, DelaunayTopologyErrorKind,
};

pub(super) fn try_recover_edge_with_face_flip(
    topology: &DelaunayVolumeTopology,
    left: StableDigest,
    right: StableDigest,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    let left_index = node_index(topology, left).ok_or_else(|| {
        invalid_topology(
            constraint_index,
            "face-flip endpoint is missing from topology",
        )
    })? as u32;
    let right_index = node_index(topology, right).ok_or_else(|| {
        invalid_topology(
            constraint_index,
            "face-flip endpoint is missing from topology",
        )
    })? as u32;
    let mut candidates = Vec::new();
    for tetrahedron_index in &topology.incidence.vertex_stars[left_index as usize] {
        work.search_step(constraint_index)?;
        let tetrahedron = &topology.tetrahedra[*tetrahedron_index as usize];
        let left_slot = tetrahedron
            .vertex_indices
            .iter()
            .position(|vertex| *vertex == left_index)
            .ok_or_else(|| {
                invalid_topology(
                    constraint_index,
                    "vertex-star incidence does not contain its declared node",
                )
            })?;
        let Some(neighbor_index) = tetrahedron.neighbors[left_slot] else {
            continue;
        };
        let neighbor = &topology.tetrahedra[neighbor_index as usize];
        if !neighbor.vertex_indices.contains(&right_index) {
            continue;
        }
        let mut face = tetrahedron
            .vertex_indices
            .into_iter()
            .filter(|vertex| *vertex != left_index)
            .collect::<Vec<_>>();
        face.sort_unstable();
        candidates.push((
            [face[0], face[1], face[2]],
            *tetrahedron_index,
            neighbor_index,
        ));
    }
    candidates.sort_unstable();
    candidates.dedup();

    for (face, left_tetrahedron, right_tetrahedron) in candidates {
        work.flip_attempt(constraint_index)?;
        let left_region = &topology.tetrahedra[left_tetrahedron as usize].region_id;
        let right_region = &topology.tetrahedra[right_tetrahedron as usize].region_id;
        if left_region != right_region
            || !convex_flip_cavity(topology, left_index, right_index, face, constraint_index)?
        {
            continue;
        }
        let mut tetrahedra = topology
            .tetrahedra
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                *index != left_tetrahedron as usize && *index != right_tetrahedron as usize
            })
            .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
            .collect::<Vec<_>>();
        for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            tetrahedra.push((
                [left_index, right_index, edge[0], edge[1]],
                left_region.clone(),
            ));
        }
        let candidate = match build_delaunay_volume_topology_with_regions(
            topology.nodes.clone(),
            tetrahedra,
            work.options.insertion.topology,
            work.cancellation,
        ) {
            Ok(candidate) => candidate,
            Err(candidate_error) => match candidate_error.kind {
                DelaunayTopologyErrorKind::ResourceLimit => {
                    return Err(error(
                        DelaunaySegmentRecoveryErrorKind::ResourceLimit,
                        Some(constraint_index),
                        candidate_error.to_string(),
                    ));
                }
                DelaunayTopologyErrorKind::Cancelled => {
                    return Err(error(
                        DelaunaySegmentRecoveryErrorKind::Cancelled,
                        Some(constraint_index),
                        candidate_error.to_string(),
                    ));
                }
                _ => continue,
            },
        };
        match validate_delaunay_volume_topology(
            &candidate,
            work.options.insertion,
            work.cancellation,
        ) {
            Ok(()) => {}
            Err(validation) => match validation.kind {
                DelaunayInsertionErrorKind::ResourceLimit => {
                    return Err(error(
                        DelaunaySegmentRecoveryErrorKind::ResourceLimit,
                        Some(constraint_index),
                        validation.to_string(),
                    ));
                }
                DelaunayInsertionErrorKind::Cancelled => {
                    return Err(error(
                        DelaunaySegmentRecoveryErrorKind::Cancelled,
                        Some(constraint_index),
                        validation.to_string(),
                    ));
                }
                _ => continue,
            },
        }
        if edge_exists(&candidate, left, right, constraint_index, work)? {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn convex_flip_cavity(
    topology: &DelaunayVolumeTopology,
    left: u32,
    right: u32,
    face: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let coordinates = |index: u32| topology.nodes[index as usize].coordinates_m;
    let signs = [
        [left, right, face[0], face[1]],
        [left, right, face[1], face[2]],
        [left, right, face[2], face[0]],
    ]
    .map(|vertices| orient3d(vertices.map(coordinates)))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(|predicate| {
        error(
            DelaunaySegmentRecoveryErrorKind::InvalidTopology,
            Some(constraint_index),
            format!("face-flip orientation predicate failed: {predicate:?}"),
        )
    })?;
    Ok(signs[0] != PredicateSign::Zero && signs.iter().all(|sign| *sign == signs[0]))
}

fn invalid_topology(constraint_index: u32, reason: &'static str) -> DelaunaySegmentRecoveryError {
    error(
        DelaunaySegmentRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}
