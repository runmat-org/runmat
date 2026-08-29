use runmat_meshing_core::{
    quality::predicate::{insphere3d, orient3d, tetrahedron_volume, PredicateSign},
    StableDigest,
};

use super::{
    edge_exists, error, node_index, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DelaunayVolumeTopology, RecoveryWork,
};
use crate::{
    cdt::{
        insertion::{
            topology_access::TetrahedralTopology, validate_constrained_delaunay_volume_topology,
        },
        mutation::TopologyMutation,
        DelaunayInsertionErrorKind,
    },
    reconnect::two_to_three_geometry_is_admissible,
};

pub(super) fn try_recover_edge_with_face_flip(
    topology: &DelaunayVolumeTopology,
    left: StableDigest,
    right: StableDigest,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    let mut mutation = TopologyMutation::from_canonical(topology.clone());
    if !try_recover_edge_with_face_flip_mutation(
        &mut mutation,
        left,
        right,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(None);
    }
    let candidate = mutation
        .freeze(work.options.insertion.topology, work.cancellation)
        .map_err(|failure| invalid_topology(constraint_index, failure.to_string()))?;
    validate_constrained_delaunay_volume_topology(
        &candidate,
        protected_faces,
        work.options.insertion,
        work.cancellation,
    )
    .map_err(|failure| match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => error(
            DelaunaySegmentRecoveryErrorKind::ResourceLimit,
            Some(constraint_index),
            failure.to_string(),
        ),
        DelaunayInsertionErrorKind::Cancelled => error(
            DelaunaySegmentRecoveryErrorKind::Cancelled,
            Some(constraint_index),
            failure.to_string(),
        ),
        _ => invalid_topology(constraint_index, failure.to_string()),
    })?;
    Ok(Some(candidate))
}

pub(super) fn try_recover_edge_with_face_flip_mutation(
    topology: &mut TopologyMutation,
    left: StableDigest,
    right: StableDigest,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
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
    for tetrahedron_index in topology.vertex_star(left_index) {
        work.search_step(constraint_index)?;
        let tetrahedron = topology.tetrahedron(tetrahedron_index).ok_or_else(|| {
            invalid_topology(
                constraint_index,
                "vertex-star incidence references an inactive tetrahedron",
            )
        })?;
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
        let neighbor = topology
            .tetrahedron(neighbor_index)
            .ok_or_else(|| invalid_topology(constraint_index, "face-flip neighbor is inactive"))?;
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
            tetrahedron_index,
            neighbor_index,
        ));
    }
    candidates.sort_unstable();
    candidates.dedup();

    for (face, left_tetrahedron, right_tetrahedron) in candidates {
        work.flip_attempt(constraint_index)?;
        let left_region = topology
            .tetrahedron(left_tetrahedron)
            .expect("candidate tetrahedron remains active")
            .region_id
            .clone();
        let right_region = topology
            .tetrahedron(right_tetrahedron)
            .expect("candidate tetrahedron remains active")
            .region_id
            .clone();
        if left_region != right_region
            || protected_faces.contains(&stable_face(topology, face))
            || !convex_flip_cavity(topology, left_index, right_index, face, constraint_index)?
            || !flip_preserves_local_volume(
                topology,
                left_tetrahedron,
                right_tetrahedron,
                left_index,
                right_index,
                face,
            )
        {
            continue;
        }
        let mut candidate = topology.clone();
        let removed = [left_tetrahedron, right_tetrahedron]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        let mut created = Vec::with_capacity(3);
        for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            created.push((
                [left_index, right_index, edge[0], edge[1]],
                left_region.clone(),
            ));
        }
        let created = match candidate.replace_cavity(&removed, created) {
            Ok(created) => created,
            Err(_) => continue,
        };
        if !created_faces_are_constrained_delaunay(
            &candidate,
            &created,
            protected_faces,
            constraint_index,
            work,
        )? {
            continue;
        }
        if edge_exists(&candidate, left, right, constraint_index, work)? {
            *topology = candidate;
            return Ok(true);
        }
    }
    Ok(false)
}

fn flip_preserves_local_volume(
    topology: &impl TetrahedralTopology,
    left_tetrahedron: u32,
    right_tetrahedron: u32,
    left: u32,
    right: u32,
    face: [u32; 3],
) -> bool {
    let coordinates = |index: u32| topology.nodes()[index as usize].coordinates_m;
    let initial = [left_tetrahedron, right_tetrahedron]
        .into_iter()
        .map(|index| {
            tetrahedron_volume(
                topology
                    .tetrahedron(index)
                    .expect("face-flip candidate is active")
                    .vertex_indices
                    .map(coordinates),
            )
        })
        .sum::<f64>();
    let replacement = [
        [left, right, face[0], face[1]],
        [left, right, face[1], face[2]],
        [left, right, face[2], face[0]],
    ]
    .into_iter()
    .map(|vertices| tetrahedron_volume(vertices.map(coordinates)))
    .sum::<f64>();
    let scale = initial.abs().max(replacement.abs()).max(f64::MIN_POSITIVE);
    initial.is_finite()
        && replacement.is_finite()
        && (initial - replacement).abs() <= scale * 64.0 * f64::EPSILON
}

fn convex_flip_cavity(
    topology: &impl TetrahedralTopology,
    left: u32,
    right: u32,
    face: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let coordinates = |index: u32| topology.nodes()[index as usize].coordinates_m;
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

fn created_faces_are_constrained_delaunay(
    topology: &TopologyMutation,
    created: &[u32],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let created = created
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut examined = std::collections::BTreeSet::new();
    for tetrahedron_index in &created {
        let tetrahedron = topology
            .tetrahedron(*tetrahedron_index)
            .expect("created face-flip tetrahedron remains active");
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor_index) = *neighbor else {
                continue;
            };
            let pair = [*tetrahedron_index, neighbor_index];
            let mut pair_key = pair;
            pair_key.sort_unstable();
            if !examined.insert(pair_key) {
                continue;
            }
            work.search_step(constraint_index)?;
            let neighbor = topology.tetrahedron(neighbor_index).ok_or_else(|| {
                invalid_topology(constraint_index, "created face-flip neighbor is inactive")
            })?;
            let face = face_opposite(tetrahedron.vertex_indices, opposite);
            if protected_faces.contains(&stable_face(topology, face))
                || tetrahedron.region_id != neighbor.region_id
            {
                continue;
            }
            let right_apex = neighbor
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !tetrahedron.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    invalid_topology(
                        constraint_index,
                        "created face-flip neighbors do not share exactly one face",
                    )
                })?;
            let left_apex = tetrahedron
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !neighbor.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    invalid_topology(
                        constraint_index,
                        "created face-flip neighbors have no distinct apex",
                    )
                })?;
            let points = tetrahedron
                .vertex_indices
                .map(|vertex| topology.nodes()[vertex as usize].coordinates_m);
            let relation = insphere3d([
                points[0],
                points[1],
                points[2],
                points[3],
                topology.nodes()[right_apex as usize].coordinates_m,
            ])
            .map_err(|failure| {
                invalid_topology(
                    constraint_index,
                    format!("face-flip circumsphere predicate failed: {failure:?}"),
                )
            })?;
            if relation == PredicateSign::Positive
                && two_to_three_topology_is_admissible(
                    topology, pair[0], pair[1], left_apex, right_apex,
                )
                && two_to_three_geometry_is_admissible(
                    tetrahedron.vertex_indices,
                    neighbor.vertex_indices,
                    |vertex| topology.nodes()[vertex as usize].coordinates_m,
                )
                .map_err(|failure| {
                    invalid_topology(
                        constraint_index,
                        format!("face-flip admissibility predicate failed: {failure:?}"),
                    )
                })?
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn two_to_three_topology_is_admissible(
    topology: &impl TetrahedralTopology,
    left_tetrahedron: u32,
    right_tetrahedron: u32,
    left_apex: u32,
    right_apex: u32,
) -> bool {
    !topology.vertex_star(left_apex).into_iter().any(|index| {
        index != left_tetrahedron
            && index != right_tetrahedron
            && topology
                .tetrahedron(index)
                .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&right_apex))
    })
}

fn face_opposite(vertices: [u32; 4], opposite: usize) -> [u32; 3] {
    let mut face = [0; 3];
    let mut cursor = 0;
    for (slot, vertex) in vertices.into_iter().enumerate() {
        if slot != opposite {
            face[cursor] = vertex;
            cursor += 1;
        }
    }
    face.sort_unstable();
    face
}

fn stable_face(topology: &impl TetrahedralTopology, face: [u32; 3]) -> [StableDigest; 3] {
    let mut stable = face.map(|vertex| topology.nodes()[vertex as usize].identity);
    stable.sort_unstable();
    stable
}

fn invalid_topology(
    constraint_index: u32,
    reason: impl Into<String>,
) -> DelaunaySegmentRecoveryError {
    error(
        DelaunaySegmentRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}
