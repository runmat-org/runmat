use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    StableDigest,
};

use super::{
    cavity::{circumsphere_relation, stable_face},
    error, topology_error, two_to_three_topology_is_admissible, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayVolumeTopology, InsertionWork,
};
use crate::{
    cdt::topology::build_delaunay_volume_topology_with_regions,
    reconnect::{
        three_to_two_edge_flip_candidate, two_to_three_geometry_is_admissible, LocalTetrahedron,
    },
};

pub(super) fn legalize_delaunay_volume_topology(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut flip_count = 0_u64;
    loop {
        let Some((left, right, face)) =
            first_illegal_face(&topology, protected_faces, protect_region_boundaries, work)?
        else {
            return Ok(topology);
        };
        record_flip(&mut flip_count, work)?;
        topology = apply_two_to_three_flip(topology, left, right, face, work)?;
    }
}

pub(super) fn legalize_delaunay_volume_topology_complete(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut flip_count = 0_u64;
    loop {
        if let Some((removed, created)) =
            first_reducible_edge(&topology, protected_faces, protect_region_boundaries, work)?
        {
            record_flip(&mut flip_count, work)?;
            topology = apply_reconnection(topology, &removed, created, work)?;
            continue;
        }
        let Some((left, right, face)) =
            first_illegal_face(&topology, protected_faces, protect_region_boundaries, work)?
        else {
            return Ok(topology);
        };
        record_flip(&mut flip_count, work)?;
        topology = apply_two_to_three_flip(topology, left, right, face, work)?;
    }
}

fn record_flip(
    flip_count: &mut u64,
    work: &InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    *flip_count = flip_count.checked_add(1).ok_or_else(|| {
        error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "Delaunay legalization flip count overflow",
        )
    })?;
    if *flip_count > work.options.maximum_cavity_tetrahedra {
        return Err(error(
            DelaunayInsertionErrorKind::ResourceLimit,
            "Delaunay legalization flip limit exceeded",
        ));
    }
    Ok(())
}

pub(super) fn first_reducible_edge(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    _protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(Vec<usize>, Vec<[u32; 4]>)>, DelaunayInsertionError> {
    let mut edge_stars = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for left in 0..4 {
            for right in (left + 1)..4 {
                let mut edge = [
                    tetrahedron.vertex_indices[left],
                    tetrahedron.vertex_indices[right],
                ];
                edge.sort_unstable();
                edge_stars.entry(edge).or_default().push(tetrahedron_index);
            }
        }
    }
    for (edge, star) in edge_stars {
        if star.len() != 3 {
            continue;
        }
        work.checkpoint()?;
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
        {
            continue;
        }
        let local = std::array::from_fn(|slot| LocalTetrahedron {
            tetrahedron_id: star[slot] as u32,
            node_ids: topology.tetrahedra[star[slot]].vertex_indices,
        });
        let Ok(candidate) = three_to_two_edge_flip_candidate(local, edge) else {
            continue;
        };
        let mut removed_faces = BTreeSet::new();
        for index in &star {
            let tetrahedron = &topology.tetrahedra[*index];
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| star.contains(&(neighbor as usize)))
                {
                    removed_faces.insert(stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ));
                }
            }
        }
        if removed_faces
            .iter()
            .any(|face| protected_faces.contains(face))
        {
            continue;
        }
        let mut created: [[u32; 4]; 2] = candidate
            .created_tetrahedra
            .try_into()
            .expect("three-to-two reconnection creates two tetrahedra");
        let mut nondegenerate = true;
        for tetrahedron in &mut created {
            match orient3d(tetrahedron.map(|node| topology.nodes[node as usize].coordinates_m))
                .map_err(super::predicate_error)?
            {
                PredicateSign::Positive => {}
                PredicateSign::Negative => tetrahedron.swap(0, 1),
                PredicateSign::Zero => nondegenerate = false,
            }
        }
        if !nondegenerate {
            continue;
        }
        let created_keys = created.map(|mut tetrahedron| {
            tetrahedron.sort_unstable();
            tetrahedron
        });
        if created_keys[0] == created_keys[1]
            || topology
                .tetrahedra
                .iter()
                .enumerate()
                .filter(|(index, _)| !star.contains(index))
                .any(|(_, tetrahedron)| {
                    let mut key = tetrahedron.vertex_indices;
                    key.sort_unstable();
                    created_keys.contains(&key)
                })
        {
            continue;
        }
        if !prospective_faces_are_manifold(topology, &star, &created) {
            continue;
        }
        if !two_to_three_geometry_is_admissible(created[0], created[1], |node| {
            topology.nodes[node as usize].coordinates_m
        })
        .map_err(super::predicate_error)?
        {
            continue;
        }
        let opposite = created[1]
            .iter()
            .copied()
            .find(|node| !created[0].contains(node))
            .expect("inverse face flip has an opposite node");
        if circumsphere_relation(
            topology,
            created[0],
            topology.nodes[opposite as usize],
            work,
        )? == PredicateSign::Positive
        {
            continue;
        }
        return Ok(Some((star, created.into_iter().collect())));
    }
    Ok(None)
}

fn prospective_faces_are_manifold(
    topology: &DelaunayVolumeTopology,
    removed: &[usize],
    created: &[[u32; 4]],
) -> bool {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let retained = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(_, tetrahedron)| tetrahedron.vertex_indices);
    for tetrahedron in retained.chain(created.iter().copied()) {
        for opposite in 0..4 {
            let mut face = [0u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            let count = face_counts.entry(face).or_default();
            *count += 1;
            if *count > 2 {
                return false;
            }
        }
    }
    true
}

fn apply_reconnection(
    topology: DelaunayVolumeTopology,
    removed: &[usize],
    created: Vec<[u32; 4]>,
    work: &InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let region = topology.tetrahedra[removed[0]].region_id.clone();
    let removed = removed.iter().copied().collect::<BTreeSet<_>>();
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    tetrahedra.extend(
        created
            .into_iter()
            .map(|tetrahedron| (tetrahedron, region.clone())),
    );
    build_delaunay_volume_topology_with_regions(
        topology.nodes,
        tetrahedra,
        work.options.topology,
        work.cancellation(),
    )
    .map_err(topology_error)
}

fn first_illegal_face(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(usize, usize, [u32; 3])>, DelaunayInsertionError> {
    for (left_index, left) in topology.tetrahedra.iter().enumerate() {
        for (opposite, neighbor) in left.neighbors.iter().enumerate() {
            let Some(right_index) = neighbor.map(|index| index as usize) else {
                continue;
            };
            if right_index <= left_index {
                continue;
            }
            work.checkpoint()?;
            let right = &topology.tetrahedra[right_index];
            let stable = stable_face(topology, left.vertex_indices, opposite);
            if protected_faces.contains(&stable)
                || protect_region_boundaries && left.region_id != right.region_id
            {
                continue;
            }
            let right_apex = right
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !left.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "legalization neighbors do not share exactly one triangular face",
                    )
                })?;
            let left_apex = left
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !right.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "legalization neighbors do not have distinct opposite vertices",
                    )
                })?;
            if circumsphere_relation(
                topology,
                left.vertex_indices,
                topology.nodes[right_apex as usize],
                work,
            )? != PredicateSign::Positive
                || !two_to_three_topology_is_admissible(
                    topology,
                    left_index,
                    right_index,
                    left_apex,
                    right_apex,
                )
                || !two_to_three_geometry_is_admissible(
                    left.vertex_indices,
                    right.vertex_indices,
                    |node| topology.nodes[node as usize].coordinates_m,
                )
                .map_err(super::predicate_error)?
            {
                continue;
            }
            let mut face = [0; 3];
            let mut cursor = 0;
            for vertex in left.vertex_indices {
                if right.vertex_indices.contains(&vertex) {
                    face[cursor] = vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            return Ok(Some((left_index, right_index, face)));
        }
    }
    Ok(None)
}

fn apply_two_to_three_flip(
    topology: DelaunayVolumeTopology,
    left_index: usize,
    right_index: usize,
    face: [u32; 3],
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let left = &topology.tetrahedra[left_index];
    let right = &topology.tetrahedra[right_index];
    if left.region_id != right.region_id {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "Delaunay legalization cannot cross a region boundary",
        ));
    }
    let left_apex = left
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !face.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "missing left apex",
            )
        })?;
    let right_apex = right
        .vertex_indices
        .iter()
        .copied()
        .find(|vertex| !face.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "missing right apex",
            )
        })?;
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != left_index && *index != right_index)
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        tetrahedra.push((
            [left_apex, right_apex, edge[0], edge[1]],
            left.region_id.clone(),
        ));
    }
    build_delaunay_volume_topology_with_regions(
        topology.nodes,
        tetrahedra,
        work.options.topology,
        work.cancellation(),
    )
    .map_err(topology_error)
}
