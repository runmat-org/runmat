use runmat_meshing_core::{quality::predicate::PredicateSign, StableDigest};
use std::collections::BTreeSet;

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
        topology::orient_materialized_volume_nodes as orient_volume_nodes,
        DelaunayInsertionErrorKind,
    },
    reconnect::{
        local_tetrahedron_boundary_faces, three_to_two_edge_flip_candidate, LocalTetrahedron,
    },
};

pub(super) fn try_recover_edge_with_face_flip(
    topology: &DelaunayVolumeTopology,
    left: StableDigest,
    right: StableDigest,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    let mut mutation = TopologyMutation::from_canonical(topology.clone());
    if !try_recover_edge_with_face_flip_mutation(
        &mut mutation,
        left,
        right,
        protected_edges,
        protected_faces,
        constraint_index,
        work,
    )? {
        return Ok(None);
    }
    let candidate = mutation
        .freeze(work.options.insertion.topology, work.cancellation)
        .map_err(|failure| invalid_topology(constraint_index, failure.to_string()))?;
    let validation = validate_constrained_delaunay_volume_topology(
        &candidate,
        protected_faces,
        work.options.insertion,
        work.cancellation,
    );
    match validation {
        Ok(()) => Ok(Some(candidate)),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::ResourceLimit => Err(error(
            DelaunaySegmentRecoveryErrorKind::ResourceLimit,
            Some(constraint_index),
            failure.to_string(),
        )),
        Err(failure) if failure.kind == DelaunayInsertionErrorKind::Cancelled => Err(error(
            DelaunaySegmentRecoveryErrorKind::Cancelled,
            Some(constraint_index),
            failure.to_string(),
        )),
        Err(_) => Ok(None),
    }
}

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
        match candidate.replace_cavity(&removed, created) {
            Ok(_) => {}
            Err(_) => continue,
        }
        if edge_exists(&candidate, left, right, constraint_index, work)? {
            *topology = candidate;
            return Ok(true);
        }
    }
    Ok(false)
}

fn try_recover_boundary_diagonal(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let snapshot = topology
        .snapshot(work.options.insertion.topology, work.cancellation)
        .map_err(|failure| invalid_topology(constraint_index, failure.to_string()))?;
    let left_faces = snapshot.incidence.boundary_facets.iter().filter(|facet| {
        facet.vertex_indices.contains(&left) && !facet.vertex_indices.contains(&right)
    });
    for left_face in left_faces {
        for right_face in snapshot.incidence.boundary_facets.iter().filter(|facet| {
            facet.vertex_indices.contains(&right) && !facet.vertex_indices.contains(&left)
        }) {
            work.flip_attempt(constraint_index)?;
            let shared = left_face
                .vertex_indices
                .into_iter()
                .filter(|vertex| right_face.vertex_indices.contains(vertex))
                .collect::<Vec<_>>();
            if shared.len() != 2 {
                continue;
            }
            if orient_volume_nodes([
                snapshot.nodes[left as usize],
                snapshot.nodes[right as usize],
                snapshot.nodes[shared[0] as usize],
                snapshot.nodes[shared[1] as usize],
            ])
            .map_err(|failure| {
                invalid_topology(
                    constraint_index,
                    format!("boundary-diagonal predicate failed: {failure:?}"),
                )
            })? != PredicateSign::Zero
            {
                continue;
            }
            let mut removed_edge = [shared[0], shared[1]];
            removed_edge.sort_unstable();
            if protected_edges.contains(&removed_edge)
                || protected_faces.contains(&stable_face(&snapshot, left_face.vertex_indices))
                || protected_faces.contains(&stable_face(&snapshot, right_face.vertex_indices))
            {
                continue;
            }
            let left_tetrahedron = &snapshot.tetrahedra[left_face.tetrahedron_index as usize];
            let right_tetrahedron = &snapshot.tetrahedra[right_face.tetrahedron_index as usize];
            let left_apex =
                left_tetrahedron.vertex_indices[left_face.opposite_vertex_slot as usize];
            let right_apex =
                right_tetrahedron.vertex_indices[right_face.opposite_vertex_slot as usize];
            if left_apex != right_apex || left_tetrahedron.region_id != right_tetrahedron.region_id
            {
                continue;
            }
            let removed =
                BTreeSet::from([left_face.tetrahedron_index, right_face.tetrahedron_index]);
            let created = vec![
                (
                    [left, right, shared[0], left_apex],
                    left_tetrahedron.region_id.clone(),
                ),
                (
                    [left, right, shared[1], left_apex],
                    left_tetrahedron.region_id.clone(),
                ),
            ];
            let mut candidate = topology.clone();
            if candidate.replace_cavity(&removed, created).is_err() {
                continue;
            }
            let checked =
                match candidate.snapshot(work.options.insertion.topology, work.cancellation) {
                    Ok(checked) => checked,
                    Err(failure)
                        if failure.kind
                            == crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology =>
                    {
                        continue;
                    }
                    Err(failure) => {
                        return Err(error(
                        match failure.kind {
                            crate::cdt::mutation::TopologyMutationErrorKind::ResourceLimit => {
                                DelaunaySegmentRecoveryErrorKind::ResourceLimit
                            }
                            crate::cdt::mutation::TopologyMutationErrorKind::Cancelled => {
                                DelaunaySegmentRecoveryErrorKind::Cancelled
                            }
                            crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology => {
                                unreachable!()
                            }
                        },
                        Some(constraint_index),
                        failure.to_string(),
                    ));
                    }
                };
            if edge_exists(
                &checked,
                snapshot.nodes[left as usize].identity,
                snapshot.nodes[right as usize].identity,
                constraint_index,
                work,
            )? {
                *topology = TopologyMutation::from_canonical(checked);
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn try_recover_coplanar_patch_diagonal(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let snapshot = topology
        .snapshot(work.options.insertion.topology, work.cancellation)
        .map_err(|failure| invalid_topology(constraint_index, failure.to_string()))?;
    let mut uses = std::collections::BTreeMap::<[u32; 3], Vec<(u32, u32)>>::new();
    for (tetrahedron_index, tetrahedron) in snapshot.tetrahedra.iter().enumerate() {
        for opposite in 0..4 {
            let mut face = tetrahedron
                .vertex_indices
                .into_iter()
                .enumerate()
                .filter_map(|(slot, vertex)| (slot != opposite).then_some(vertex))
                .collect::<Vec<_>>();
            face.sort_unstable();
            uses.entry([face[0], face[1], face[2]]).or_default().push((
                tetrahedron_index as u32,
                tetrahedron.vertex_indices[opposite],
            ));
        }
    }
    let left_faces = uses.iter().filter(|(face, face_uses)| {
        face.contains(&left) && !face.contains(&right) && face_uses.len() == 2
    });
    for (left_face, left_uses) in left_faces {
        for (right_face, right_uses) in uses.iter().filter(|(face, face_uses)| {
            face.contains(&right) && !face.contains(&left) && face_uses.len() == 2
        }) {
            work.flip_attempt(constraint_index)?;
            let shared = left_face
                .iter()
                .copied()
                .filter(|vertex| right_face.contains(vertex))
                .collect::<Vec<_>>();
            if shared.len() != 2
                || orient_volume_nodes([
                    snapshot.nodes[left as usize],
                    snapshot.nodes[right as usize],
                    snapshot.nodes[shared[0] as usize],
                    snapshot.nodes[shared[1] as usize],
                ])
                .map_err(|failure| {
                    invalid_topology(
                        constraint_index,
                        format!("coplanar-patch predicate failed: {failure:?}"),
                    )
                })? != PredicateSign::Zero
            {
                continue;
            }
            let mut removed_edge = [shared[0], shared[1]];
            removed_edge.sort_unstable();
            if protected_edges.contains(&removed_edge)
                || protected_faces.contains(&stable_face(&snapshot, *left_face))
                || protected_faces.contains(&stable_face(&snapshot, *right_face))
            {
                continue;
            }
            let left_apexes = left_uses
                .iter()
                .map(|(_, apex)| *apex)
                .collect::<BTreeSet<_>>();
            let right_apexes = right_uses
                .iter()
                .map(|(_, apex)| *apex)
                .collect::<BTreeSet<_>>();
            if left_apexes != right_apexes {
                continue;
            }
            let removed = left_uses
                .iter()
                .chain(right_uses)
                .map(|(tetrahedron, _)| *tetrahedron)
                .collect::<BTreeSet<_>>();
            if removed.len() != 4 {
                continue;
            }
            let mut created = Vec::with_capacity(4);
            let mut regions_match = true;
            for apex in left_apexes {
                let left_tetrahedron = left_uses
                    .iter()
                    .find(|(_, candidate)| *candidate == apex)
                    .map(|(index, _)| &snapshot.tetrahedra[*index as usize])
                    .expect("left patch apex has a tetrahedron");
                let right_tetrahedron = right_uses
                    .iter()
                    .find(|(_, candidate)| *candidate == apex)
                    .map(|(index, _)| &snapshot.tetrahedra[*index as usize])
                    .expect("right patch apex has a tetrahedron");
                if left_tetrahedron.region_id != right_tetrahedron.region_id {
                    regions_match = false;
                    break;
                }
                created.push((
                    [left, right, shared[0], apex],
                    left_tetrahedron.region_id.clone(),
                ));
                created.push((
                    [left, right, shared[1], apex],
                    left_tetrahedron.region_id.clone(),
                ));
            }
            if !regions_match {
                continue;
            }
            let mut candidate = topology.clone();
            if candidate.replace_cavity(&removed, created).is_err() {
                continue;
            }
            let Ok(checked) =
                candidate.snapshot(work.options.insertion.topology, work.cancellation)
            else {
                continue;
            };
            if edge_exists(
                &checked,
                snapshot.nodes[left as usize].identity,
                snapshot.nodes[right as usize].identity,
                constraint_index,
                work,
            )? {
                *topology = TopologyMutation::from_canonical(checked);
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn try_recover_edge_with_edge_star_rotation(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let edge_stars =
        candidate_edge_stars_linking_endpoints(topology, left, right, constraint_index, work)?;
    for (removed_edge, star) in edge_stars {
        if star.len() < 3
            || removed_edge.contains(&left)
            || removed_edge.contains(&right)
            || protected_edges.contains(&removed_edge)
        {
            continue;
        }
        let removed_tetrahedra = star
            .iter()
            .map(|index| {
                topology
                    .tetrahedron(*index)
                    .expect("edge-star tetrahedron is active")
                    .vertex_indices
            })
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let cavity_nodes = removed_tetrahedra
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let other = cavity_nodes
            .iter()
            .copied()
            .filter(|node| *node != left && *node != right)
            .collect::<Vec<_>>();
        let mut created = Vec::new();
        for first in 0..other.len() {
            for second in (first + 1)..other.len() {
                let mut left_face = [left, other[first], other[second]];
                let mut right_face = [right, other[first], other[second]];
                left_face.sort_unstable();
                right_face.sort_unstable();
                if boundary.contains(&left_face) && boundary.contains(&right_face) {
                    created.push([left, right, other[first], other[second]]);
                }
            }
        }
        let boundary_matches = local_tetrahedron_boundary_faces(&created) == boundary;
        if created.len() < 2 || !boundary_matches {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let region = topology
            .tetrahedron(star[0])
            .expect("edge-star tetrahedron is active")
            .region_id
            .clone();
        let regions_match = !star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .is_none_or(|tetrahedron| tetrahedron.region_id != region)
        });
        let crosses_protected =
            removed_internal_face_is_protected(topology, &star, protected_faces);
        if !regions_match || crosses_protected {
            continue;
        }
        let mut trial = topology.clone();
        let removed = star
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        match trial.replace_cavity(
            &removed,
            created
                .into_iter()
                .map(|tetrahedron| (tetrahedron, region.clone()))
                .collect(),
        ) {
            Ok(_) => {}
            Err(_) => continue,
        }
        *topology = trial;
        return Ok(true);
    }
    Ok(false)
}

fn try_recover_edge_with_three_to_two_mutation(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let edge_stars =
        candidate_edge_stars_linking_endpoints(topology, left, right, constraint_index, work)?;
    for (removed_edge, star) in edge_stars {
        if star.len() != 3
            || removed_edge.contains(&left)
            || removed_edge.contains(&right)
            || protected_edges.contains(&removed_edge)
        {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let local = std::array::from_fn(|index| {
            let tetrahedron = topology
                .tetrahedron(star[index])
                .expect("edge-star tetrahedron remains active");
            LocalTetrahedron {
                tetrahedron_id: star[index],
                node_ids: tetrahedron.vertex_indices,
            }
        });
        let Ok(candidate) = three_to_two_edge_flip_candidate(local, removed_edge) else {
            continue;
        };
        if !candidate
            .created_tetrahedra
            .iter()
            .any(|tetrahedron| tetrahedron.contains(&left) && tetrahedron.contains(&right))
        {
            continue;
        }
        let region = topology
            .tetrahedron(star[0])
            .expect("edge-star tetrahedron remains active")
            .region_id
            .clone();
        if star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .is_none_or(|tetrahedron| tetrahedron.region_id != region)
        }) || removed_internal_face_is_protected(topology, &star, protected_faces)
        {
            continue;
        }
        let mut trial = topology.clone();
        let removed = star
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        match trial.replace_cavity(
            &removed,
            candidate
                .created_tetrahedra
                .iter()
                .copied()
                .map(|tetrahedron| (tetrahedron, region.clone()))
                .collect(),
        ) {
            Ok(_) => {}
            Err(_) => continue,
        }
        let left_identity = topology.nodes()[left as usize].identity;
        let right_identity = topology.nodes()[right as usize].identity;
        if edge_exists(
            &trial,
            left_identity,
            right_identity,
            constraint_index,
            work,
        )? {
            *topology = trial;
            return Ok(true);
        }
    }
    Ok(false)
}

fn candidate_edge_stars_linking_endpoints(
    topology: &TopologyMutation,
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<std::collections::BTreeMap<[u32; 2], Vec<u32>>, DelaunaySegmentRecoveryError> {
    let incident_edges = |endpoint: u32,
                          work: &mut RecoveryWork<'_>|
     -> Result<BTreeSet<[u32; 2]>, DelaunaySegmentRecoveryError> {
        let mut edges = BTreeSet::new();
        for tetrahedron_index in topology.vertex_star(endpoint) {
            work.search_step(constraint_index)?;
            let tetrahedron = topology.tetrahedron(tetrahedron_index).ok_or_else(|| {
                invalid_topology(
                    constraint_index,
                    "vertex-star incidence references an inactive tetrahedron",
                )
            })?;
            for first in 0..4 {
                for second in (first + 1)..4 {
                    let mut edge = [
                        tetrahedron.vertex_indices[first],
                        tetrahedron.vertex_indices[second],
                    ];
                    edge.sort_unstable();
                    if !edge.contains(&left) && !edge.contains(&right) {
                        edges.insert(edge);
                    }
                }
            }
        }
        Ok(edges)
    };
    let left_edges = incident_edges(left, work)?;
    let right_edges = incident_edges(right, work)?;
    let mut stars = std::collections::BTreeMap::new();
    for edge in left_edges.intersection(&right_edges).copied() {
        let mut star = Vec::new();
        for tetrahedron_index in topology.vertex_star(edge[0]) {
            work.search_step(constraint_index)?;
            if topology
                .tetrahedron(tetrahedron_index)
                .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&edge[1]))
            {
                star.push(tetrahedron_index);
            }
        }
        star.sort_unstable();
        stars.insert(edge, star);
    }
    Ok(stars)
}

fn try_recover_edge_by_crossing_face_flips(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let mut visited = std::collections::BTreeSet::new();
    let constraints = CrossingFaceConstraints {
        target: [left, right],
        protected_edges,
        protected_faces,
        constraint_index,
    };
    search_crossing_face_flips(topology, &constraints, work, &mut visited)
}

struct CrossingFaceConstraints<'a> {
    target: [u32; 2],
    protected_edges: &'a BTreeSet<[u32; 2]>,
    protected_faces: &'a [[StableDigest; 3]],
    constraint_index: u32,
}

fn search_crossing_face_flips(
    topology: &mut TopologyMutation,
    constraints: &CrossingFaceConstraints<'_>,
    work: &mut RecoveryWork<'_>,
    visited: &mut std::collections::BTreeSet<Vec<[u32; 4]>>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let [left, right] = constraints.target;
    let constraint_index = constraints.constraint_index;
    loop {
        if !visited.insert(topology_state(topology)) {
            return Ok(false);
        }
        let initial_crossings = crossing_face_count(topology, left, right, constraint_index, work)?;
        if initial_crossings == 0 {
            if let Some(face) =
                first_edge_crossing_face(topology, left, right, constraint_index, work)?
            {
                if let Some(updated) =
                    try_reduce_crossing_face_edge(topology, face, 0, constraints, work, visited)?
                {
                    *topology = updated;
                    continue;
                }
            }
            return Ok(false);
        }
        let mut accepted = None;
        'tetrahedra: for left_tetrahedron in 0..topology.tetrahedron_slot_count() {
            let Some(tetrahedron) = topology.tetrahedron(left_tetrahedron as u32) else {
                continue;
            };
            for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
                let Some(right_tetrahedron) = *neighbor else {
                    continue;
                };
                if right_tetrahedron <= left_tetrahedron as u32 {
                    continue;
                }
                work.search_step(constraint_index)?;
                let face = face_opposite(tetrahedron.vertex_indices, opposite);
                if !segment_strictly_crosses_face(topology, [left, right], face, constraint_index)?
                    || constraints
                        .protected_faces
                        .contains(&stable_face(topology, face))
                {
                    continue;
                }
                let neighbor = topology
                    .tetrahedron(right_tetrahedron)
                    .expect("crossing-face neighbor is active");
                if tetrahedron.region_id != neighbor.region_id {
                    continue;
                }
                let left_apex = tetrahedron.vertex_indices[opposite];
                let Some(right_apex) = neighbor
                    .vertex_indices
                    .iter()
                    .copied()
                    .find(|vertex| !face.contains(vertex))
                else {
                    continue;
                };
                work.flip_attempt(constraint_index)?;
                let topology_admissible = two_to_three_topology_is_admissible(
                    topology,
                    left_tetrahedron as u32,
                    right_tetrahedron,
                    left_apex,
                    right_apex,
                );
                let geometry_admissible =
                    convex_flip_cavity(topology, left_apex, right_apex, face, constraint_index)?;
                if !topology_admissible {
                    continue;
                }
                if !geometry_admissible {
                    if let Some(reduced) = try_reduce_crossing_face_edge(
                        topology,
                        face,
                        initial_crossings,
                        constraints,
                        work,
                        visited,
                    )? {
                        accepted = Some((reduced, false));
                        break 'tetrahedra;
                    }
                    continue;
                }
                let mut trial = topology.clone();
                let removed = [left_tetrahedron as u32, right_tetrahedron]
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>();
                let tetrahedron_region = tetrahedron.region_id.clone();
                let created = [
                    [left_apex, right_apex, face[0], face[1]],
                    [left_apex, right_apex, face[1], face[2]],
                    [left_apex, right_apex, face[2], face[0]],
                ]
                .into_iter()
                .map(|tetrahedron| (tetrahedron, tetrahedron_region.clone()))
                .collect();
                match trial.replace_cavity(&removed, created) {
                    Ok(_) => {}
                    Err(_) => continue,
                }
                let recovered = trial.vertex_star(left).into_iter().any(|index| {
                    trial
                        .tetrahedron(index)
                        .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&right))
                });
                let remaining = crossing_face_count(&trial, left, right, constraint_index, work)?;
                if !recovered && remaining >= initial_crossings {
                    continue;
                }
                accepted = Some((trial, recovered));
                break 'tetrahedra;
            }
        }
        let Some((updated, recovered)) = accepted else {
            return Ok(false);
        };
        *topology = updated;
        if recovered {
            return Ok(true);
        }
    }
}

fn topology_state(topology: &impl TetrahedralTopology) -> Vec<[u32; 4]> {
    let mut state = (0..topology.tetrahedron_slot_count())
        .filter_map(|index| topology.tetrahedron(index as u32))
        .map(|tetrahedron| {
            let mut vertices = tetrahedron.vertex_indices;
            vertices.sort_unstable();
            vertices
        })
        .collect::<Vec<_>>();
    state.sort_unstable();
    state
}

fn try_reduce_crossing_face_edge(
    topology: &TopologyMutation,
    face: [u32; 3],
    initial_crossings: usize,
    constraints: &CrossingFaceConstraints<'_>,
    work: &mut RecoveryWork<'_>,
    visited: &mut std::collections::BTreeSet<Vec<[u32; 4]>>,
) -> Result<Option<TopologyMutation>, DelaunaySegmentRecoveryError> {
    let [target_left, target_right] = constraints.target;
    let constraint_index = constraints.constraint_index;
    let initial_obstructions = crossing_face_obstruction_count(
        topology,
        target_left,
        target_right,
        constraint_index,
        work,
    )?;
    let mut best = None::<(usize, usize, TopologyMutation)>;
    for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        edge.sort_unstable();
        if constraints.protected_edges.contains(&edge) {
            continue;
        }
        let star = topology
            .vertex_star(edge[0])
            .into_iter()
            .filter(|index| {
                topology
                    .tetrahedron(*index)
                    .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&edge[1]))
            })
            .collect::<Vec<_>>();
        if star.len() < 3 {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let removed_tetrahedra = star
            .iter()
            .map(|index| {
                topology
                    .tetrahedron(*index)
                    .expect("crossing edge-star tetrahedron is active")
                    .vertex_indices
            })
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let cavity_nodes = removed_tetrahedra
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let region = topology
            .tetrahedron(star[0])
            .expect("crossing edge-star tetrahedron is active")
            .region_id
            .clone();
        if star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .is_none_or(|tetrahedron| tetrahedron.region_id != region)
        }) || removed_internal_face_is_protected(topology, &star, constraints.protected_faces)
        {
            continue;
        }
        let nodes = cavity_nodes.into_iter().collect::<Vec<_>>();
        for first in 0..nodes.len() {
            for second in (first + 1)..nodes.len() {
                let mut diagonal = [nodes[first], nodes[second]];
                diagonal.sort_unstable();
                if diagonal == edge {
                    continue;
                }
                let other = nodes
                    .iter()
                    .copied()
                    .filter(|node| !diagonal.contains(node))
                    .collect::<Vec<_>>();
                let mut replacement = Vec::new();
                for left in 0..other.len() {
                    for right in (left + 1)..other.len() {
                        let mut first_face = [diagonal[0], other[left], other[right]];
                        let mut second_face = [diagonal[1], other[left], other[right]];
                        first_face.sort_unstable();
                        second_face.sort_unstable();
                        if boundary.contains(&first_face) && boundary.contains(&second_face) {
                            replacement.push([diagonal[0], diagonal[1], other[left], other[right]]);
                        }
                    }
                }
                if replacement.len() < 2
                    || local_tetrahedron_boundary_faces(&replacement) != boundary
                {
                    continue;
                }
                let mut trial = topology.clone();
                let removed = star
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>();
                match trial.replace_cavity(
                    &removed,
                    replacement
                        .into_iter()
                        .map(|tetrahedron| (tetrahedron, region.clone()))
                        .collect(),
                ) {
                    Ok(_) => {}
                    Err(_) => continue,
                }
                let remaining =
                    crossing_face_count(&trial, target_left, target_right, constraint_index, work)?;
                let obstructions = crossing_face_obstruction_count(
                    &trial,
                    target_left,
                    target_right,
                    constraint_index,
                    work,
                )?;
                if (remaining, obstructions) > (initial_crossings, initial_obstructions) {
                    continue;
                }
                if (remaining, obstructions) == (initial_crossings, initial_obstructions) {
                    if initial_crossings == 0 {
                        return Ok(Some(trial));
                    }
                    let mut branch = trial;
                    if search_crossing_face_flips(&mut branch, constraints, work, visited)? {
                        return Ok(Some(branch));
                    }
                    continue;
                }
                if best
                    .as_ref()
                    .is_none_or(|current| (remaining, obstructions) < (current.0, current.1))
                {
                    best = Some((remaining, obstructions, trial));
                }
            }
        }
    }
    Ok(best.map(|(_, _, topology)| topology))
}

fn crossing_face_obstruction_count(
    topology: &impl TetrahedralTopology,
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<usize, DelaunaySegmentRecoveryError> {
    let mut count = 0;
    for tetrahedron_index in 0..topology.tetrahedron_slot_count() {
        let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index as u32) else {
            continue;
        };
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor_index) = *neighbor else {
                continue;
            };
            if neighbor_index <= tetrahedron_index as u32 {
                continue;
            }
            work.search_step(constraint_index)?;
            let face = face_opposite(tetrahedron.vertex_indices, opposite);
            if !segment_strictly_crosses_face(topology, [left, right], face, constraint_index)? {
                continue;
            }
            let neighbor = topology
                .tetrahedron(neighbor_index)
                .expect("crossing neighbor is active");
            let left_apex = tetrahedron.vertex_indices[opposite];
            let Some(right_apex) = neighbor
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !face.contains(vertex))
            else {
                count += 1;
                continue;
            };
            if !two_to_three_topology_is_admissible(
                topology,
                tetrahedron_index as u32,
                neighbor_index,
                left_apex,
                right_apex,
            ) || !convex_flip_cavity(topology, left_apex, right_apex, face, constraint_index)?
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn first_edge_crossing_face(
    topology: &impl TetrahedralTopology,
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<[u32; 3]>, DelaunaySegmentRecoveryError> {
    let mut candidates = std::collections::BTreeSet::new();
    for tetrahedron_index in 0..topology.tetrahedron_slot_count() {
        let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index as u32) else {
            continue;
        };
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if neighbor.is_none_or(|neighbor| neighbor <= tetrahedron_index as u32) {
                continue;
            }
            work.search_step(constraint_index)?;
            let face = face_opposite(tetrahedron.vertex_indices, opposite);
            if segment_crosses_face_edge(topology, [left, right], face, constraint_index)? {
                candidates.insert(face);
            }
        }
    }
    Ok(candidates.into_iter().next())
}

fn segment_crosses_face_edge(
    topology: &impl TetrahedralTopology,
    segment: [u32; 2],
    face: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    if face.contains(&segment[0]) || face.contains(&segment[1]) {
        return Ok(false);
    }
    let orient = |indices: [u32; 4]| {
        orient_volume_nodes(indices.map(|index| topology.nodes()[index as usize])).map_err(
            |failure| {
                invalid_topology(
                    constraint_index,
                    format!("crossing-edge predicate failed: {failure:?}"),
                )
            },
        )
    };
    let side = [
        orient([face[0], face[1], face[2], segment[0]])?,
        orient([face[0], face[1], face[2], segment[1]])?,
    ];
    if side.contains(&PredicateSign::Zero) || side[0] == side[1] {
        return Ok(false);
    }
    let around = [
        orient([segment[0], segment[1], face[0], face[1]])?,
        orient([segment[0], segment[1], face[1], face[2]])?,
        orient([segment[0], segment[1], face[2], face[0]])?,
    ];
    let nonzero = around
        .iter()
        .copied()
        .filter(|sign| *sign != PredicateSign::Zero)
        .collect::<Vec<_>>();
    Ok(around
        .iter()
        .filter(|sign| **sign == PredicateSign::Zero)
        .count()
        == 1
        && nonzero.len() == 2
        && nonzero[0] == nonzero[1])
}

fn crossing_face_count(
    topology: &impl TetrahedralTopology,
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<usize, DelaunaySegmentRecoveryError> {
    let mut count = 0;
    for tetrahedron_index in 0..topology.tetrahedron_slot_count() {
        let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index as u32) else {
            continue;
        };
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if neighbor.is_none_or(|neighbor| neighbor <= tetrahedron_index as u32) {
                continue;
            }
            work.search_step(constraint_index)?;
            if segment_strictly_crosses_face(
                topology,
                [left, right],
                face_opposite(tetrahedron.vertex_indices, opposite),
                constraint_index,
            )? {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn segment_strictly_crosses_face(
    topology: &impl TetrahedralTopology,
    segment: [u32; 2],
    face: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    if face.contains(&segment[0]) || face.contains(&segment[1]) {
        return Ok(false);
    }
    let orient = |indices: [u32; 4]| {
        orient_volume_nodes(indices.map(|index| topology.nodes()[index as usize])).map_err(
            |failure| {
                invalid_topology(
                    constraint_index,
                    format!("crossing-face predicate failed: {failure:?}"),
                )
            },
        )
    };
    let side = [
        orient([face[0], face[1], face[2], segment[0]])?,
        orient([face[0], face[1], face[2], segment[1]])?,
    ];
    if side.contains(&PredicateSign::Zero) || side[0] == side[1] {
        return Ok(false);
    }
    let around = [
        orient([segment[0], segment[1], face[0], face[1]])?,
        orient([segment[0], segment[1], face[1], face[2]])?,
        orient([segment[0], segment[1], face[2], face[0]])?,
    ];
    Ok(around[0] != PredicateSign::Zero && around.iter().all(|sign| *sign == around[0]))
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
