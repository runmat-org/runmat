use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use super::*;

pub(super) fn anchor_trimmed_constrained_cavity(
    tetrahedra: &[CavityTetrahedron],
    selected: BTreeSet<usize>,
    anchor_tetrahedron_index: usize,
    protected_node_ids: Vec<u32>,
) -> Result<Option<ConstrainedCavity>, ConstrainedCavityExtractionError> {
    let Some(selected) =
        anchor_connected_tetrahedron_subset(tetrahedra, &selected, anchor_tetrahedron_index)
    else {
        return Ok(None);
    };
    let selected_score = boundary_edge_defect_score(tetrahedra, &selected);
    let mut pending = vec![(selected.clone(), selected_score)];
    let mut visited = BTreeSet::<BTreeSet<usize>>::from([selected]);
    let mut evaluated = 0_usize;

    while !pending.is_empty() && evaluated < MAX_ANCHOR_TRIM_STATES {
        let best_index = pending
            .iter()
            .enumerate()
            .min_by_key(|(_, (candidate, score))| (*score, Reverse(candidate.len())))
            .map(|(index, _)| index)
            .expect("pending should be non-empty");
        let (selected, _) = pending.swap_remove(best_index);
        evaluated += 1;
        let cavity = build_constrained_cavity_from_index_set(
            tetrahedra,
            &selected,
            protected_node_ids.clone(),
        );
        match validate_constrained_cavity(&cavity) {
            Ok(_) => return Ok(Some(cavity)),
            Err(ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. }) => {
                for edge in non_manifold_boundary_edges(tetrahedra, &selected) {
                    for owner in boundary_face_owner_indices_for_edge(tetrahedra, &selected, edge) {
                        if owner == anchor_tetrahedron_index {
                            continue;
                        }
                        let mut candidate = selected.clone();
                        candidate.remove(&owner);
                        let Some(connected) = anchor_connected_tetrahedron_subset(
                            tetrahedra,
                            &candidate,
                            anchor_tetrahedron_index,
                        ) else {
                            continue;
                        };
                        if visited.insert(connected.clone()) {
                            let score = boundary_edge_defect_score(tetrahedra, &connected);
                            pending.push((connected, score));
                        }
                    }
                }
            }
            Err(ConstrainedCavityValidationError::TooFewBoundaryFaces { .. }) => continue,
            Err(err) => return Err(ConstrainedCavityExtractionError::Validation(err)),
        }
    }
    Ok(None)
}

fn boundary_face_owner_indices_for_edge(
    tetrahedra: &[CavityTetrahedron],
    selected: &BTreeSet<usize>,
    edge: [u32; 2],
) -> Vec<usize> {
    let target_edge = sorted_edge(edge);
    boundary_face_owners(tetrahedra, selected)
        .into_iter()
        .filter_map(|(_, owners)| (owners.len() == 1).then_some(owners[0]))
        .filter_map(|(tetrahedron_index, face)| {
            face_edges(face)
                .into_iter()
                .any(|face_edge| sorted_edge(face_edge) == target_edge)
                .then_some(tetrahedron_index)
        })
        .collect()
}

fn non_manifold_boundary_edges(
    tetrahedra: &[CavityTetrahedron],
    selected: &BTreeSet<usize>,
) -> Vec<[u32; 2]> {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for (_, owners) in boundary_face_owners(tetrahedra, selected) {
        if owners.len() != 1 {
            continue;
        }
        for edge in face_edges(owners[0].1) {
            *edge_counts.entry(sorted_edge(edge)).or_default() += 1;
        }
    }
    edge_counts
        .into_iter()
        .filter_map(|(edge, count)| (count != 2).then_some(edge))
        .collect()
}

fn boundary_edge_defect_score(
    tetrahedra: &[CavityTetrahedron],
    selected: &BTreeSet<usize>,
) -> usize {
    let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
    for (_, owners) in boundary_face_owners(tetrahedra, selected) {
        if owners.len() != 1 {
            continue;
        }
        for edge in face_edges(owners[0].1) {
            *edge_counts.entry(sorted_edge(edge)).or_default() += 1;
        }
    }
    edge_counts
        .values()
        .map(|count| count.abs_diff(2))
        .sum::<usize>()
}

fn boundary_face_owners(
    tetrahedra: &[CavityTetrahedron],
    selected: &BTreeSet<usize>,
) -> BTreeMap<[u32; 3], Vec<(usize, [u32; 3])>> {
    let mut face_owners = BTreeMap::<[u32; 3], Vec<(usize, [u32; 3])>>::new();
    for tetrahedron_index in selected {
        for face in tetrahedron_faces(tetrahedra[*tetrahedron_index].node_ids) {
            face_owners
                .entry(sorted_face(face))
                .or_default()
                .push((*tetrahedron_index, face));
        }
    }
    face_owners
}

fn anchor_connected_tetrahedron_subset(
    tetrahedra: &[CavityTetrahedron],
    selected: &BTreeSet<usize>,
    anchor_tetrahedron_index: usize,
) -> Option<BTreeSet<usize>> {
    if !selected.contains(&anchor_tetrahedron_index) {
        return None;
    }
    let mut face_to_tetrahedra = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for tetrahedron_index in selected {
        for face in tetrahedron_faces(tetrahedra[*tetrahedron_index].node_ids) {
            face_to_tetrahedra
                .entry(sorted_face(face))
                .or_default()
                .push(*tetrahedron_index);
        }
    }
    let mut connected = BTreeSet::<usize>::new();
    let mut pending = vec![anchor_tetrahedron_index];
    while let Some(tetrahedron_index) = pending.pop() {
        if !connected.insert(tetrahedron_index) {
            continue;
        }
        for face in tetrahedron_faces(tetrahedra[tetrahedron_index].node_ids) {
            if let Some(neighbors) = face_to_tetrahedra.get(&sorted_face(face)) {
                for neighbor in neighbors {
                    if selected.contains(neighbor) && !connected.contains(neighbor) {
                        pending.push(*neighbor);
                    }
                }
            }
        }
    }
    Some(connected)
}
