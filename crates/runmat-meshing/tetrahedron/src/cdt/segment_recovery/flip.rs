mod admission;
pub(super) use admission::try_recover_edge_with_face_flip;
mod boundary_corridor;
mod crossing;
mod diagonal;
mod edge_star;
use crossing::try_recover_edge_by_crossing_face_flips;
pub(super) use diagonal::try_recover_admitted_planar_diagonal;
use diagonal::{try_recover_boundary_diagonal, try_recover_coplanar_patch_diagonal};
use edge_star::{
    try_recover_edge_with_edge_star_rotation, try_recover_edge_with_three_to_two_mutation,
};
mod search_state;
mod spatial_candidates;
use runmat_meshing_core::{quality::predicate::PredicateSign, StableDigest};
use search_state::topology_state;
use std::collections::BTreeSet;

use super::{
    edge_exists, error, node_index, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DelaunayVolumeTopology, RecoveryWork,
};
use crate::{
    cdt::{
        insertion::topology_access::TetrahedralTopology, mutation::TopologyMutation,
        topology::orient_materialized_volume_nodes as orient_volume_nodes,
    },
    reconnect::{
        local_tetrahedron_boundary_faces, three_to_two_edge_flip_candidate, LocalTetrahedron,
    },
};

pub(in crate::cdt) fn try_recover_edge_with_face_flip_mutation(
    topology: &mut TopologyMutation,
    left: StableDigest,
    right: StableDigest,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    try_recover_edge_with_flip_mutation(
        topology,
        FlipRecoveryRequest {
            endpoints: [left, right],
            protected_edges,
            protected_faces,
            constraint_index,
            search_crossing_faces: true,
        },
        work,
    )
}

pub(in crate::cdt) fn try_recover_edge_with_local_flip_mutation(
    topology: &mut TopologyMutation,
    left: StableDigest,
    right: StableDigest,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    try_recover_edge_with_flip_mutation(
        topology,
        FlipRecoveryRequest {
            endpoints: [left, right],
            protected_edges,
            protected_faces,
            constraint_index,
            search_crossing_faces: false,
        },
        work,
    )
}

struct FlipRecoveryRequest<'a> {
    endpoints: [StableDigest; 2],
    protected_edges: &'a [[StableDigest; 2]],
    protected_faces: &'a [[StableDigest; 3]],
    constraint_index: u32,
    search_crossing_faces: bool,
}

fn try_recover_edge_with_flip_mutation(
    topology: &mut TopologyMutation,
    request: FlipRecoveryRequest<'_>,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let FlipRecoveryRequest {
        endpoints: [left, right],
        protected_edges,
        protected_faces,
        constraint_index,
        search_crossing_faces,
    } = request;
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
    let protected_edges = protected_edge_indices(topology, protected_edges, constraint_index)?;
    if try_recover_boundary_diagonal(
        topology,
        left_index,
        right_index,
        &protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(true);
    }
    if search_crossing_faces
        && boundary_corridor::recover(
            topology,
            left_index,
            right_index,
            &protected_edges,
            protected_faces,
            constraint_index,
            work,
        )?
    {
        return Ok(true);
    }
    if try_recover_coplanar_patch_diagonal(
        topology,
        left_index,
        right_index,
        &protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(true);
    }
    if try_recover_edge_with_edge_star_rotation(
        topology,
        left_index,
        right_index,
        &protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(true);
    }
    if try_recover_edge_with_three_to_two_mutation(
        topology,
        left_index,
        right_index,
        &protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(true);
    }
    if search_crossing_faces
        && try_recover_edge_by_crossing_face_flips(
            topology,
            left_index,
            right_index,
            &protected_edges,
            protected_faces,
            constraint_index,
            work,
        )?
    {
        return Ok(true);
    }
    let mut candidates = Vec::new();
    for tetrahedron_index in topology.vertex_star(left_index) {
        work.search_flip_step(constraint_index)?;
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
        {
            continue;
        }
        let mut candidate = topology.cavity_trial();
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
        match candidate.replace_cavity(&removed, created) {
            Ok(_) => {}
            Err(_) => continue,
        }
        if edge_exists(&*candidate, left, right, constraint_index, work)? {
            candidate.commit();
            return Ok(true);
        }
    }
    Ok(false)
}

fn removed_internal_face_is_protected(
    topology: &TopologyMutation,
    removed: &[u32],
    protected_faces: &[[StableDigest; 3]],
) -> bool {
    for tetrahedron_index in removed {
        let tetrahedron = topology
            .tetrahedron(*tetrahedron_index)
            .expect("removed tetrahedron remains active");
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if neighbor.is_some_and(|neighbor| removed.contains(&neighbor))
                && protected_faces.contains(&stable_face(
                    topology,
                    face_opposite(tetrahedron.vertex_indices, opposite),
                ))
            {
                return true;
            }
        }
    }
    false
}

fn protected_edge_indices(
    topology: &impl TetrahedralTopology,
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
) -> Result<BTreeSet<[u32; 2]>, DelaunaySegmentRecoveryError> {
    protected_edges
        .iter()
        .map(|edge| {
            let resolve = |identity| {
                node_index(topology, identity)
                    .map(|index| index as u32)
                    .ok_or_else(|| {
                        invalid_topology(
                            constraint_index,
                            "protected segment edge endpoint is missing from topology",
                        )
                    })
            };
            let mut edge = [resolve(edge[0])?, resolve(edge[1])?];
            edge.sort_unstable();
            Ok(edge)
        })
        .collect()
}

fn convex_flip_cavity(
    topology: &impl TetrahedralTopology,
    left: u32,
    right: u32,
    face: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    if [left, right, face[0], face[1], face[2]]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != 5
    {
        return Ok(false);
    }
    let signs = [
        [left, right, face[0], face[1]],
        [left, right, face[1], face[2]],
        [left, right, face[2], face[0]],
    ]
    .map(|vertices| orient_volume_nodes(vertices.map(|index| topology.nodes()[index as usize])))
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
