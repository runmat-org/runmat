use super::*;

pub(super) fn on_demand_interior_mate_faces_for_trace(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
    boundary_faces: &BTreeSet<[u32; 3]>,
    trace: &BoundaryExactCoverTrace,
) -> Option<BTreeSet<[u32; 3]>> {
    let future_mate_dead_ends = trace
        .dead_ends
        .iter()
        .filter(|dead_end| dead_end.reason == "forced_interior_mate_future_mate_conflict")
        .collect::<Vec<_>>();
    let no_candidate_dead_end_faces = trace
        .dead_ends
        .iter()
        .filter_map(|dead_end| {
            (dead_end.reason == "forced_interior_mate_no_candidate_contains_face")
                .then_some(dead_end.face)
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    let open_interior_dead_end_faces = trace
        .dead_ends
        .iter()
        .filter(|dead_end| {
            matches!(
                dead_end.reason,
                "interior_face_no_raw_candidate"
                    | "interior_face_no_addable_candidate"
                    | "interior_face_candidates_exhausted"
                    | "interior_incomplete"
            )
        })
        .flat_map(|dead_end| {
            open_interior_faces_from_tetrahedron_node_ids(&dead_end.selected_tetrahedra)
        })
        .filter(|face| !boundary_faces.contains(face))
        .collect::<BTreeSet<_>>();
    let root_blocked_boundary_mate_faces = trace
        .dead_ends
        .iter()
        .any(|dead_end| dead_end.reason == "boundary_face_no_addable_candidate")
        .then(|| {
            root_boundary_future_mate_faces(cavity, candidates, options.volume_relative_tolerance)
        })
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
    if future_mate_dead_ends.is_empty()
        && no_candidate_dead_end_faces.is_empty()
        && open_interior_dead_end_faces.is_empty()
        && root_blocked_boundary_mate_faces.is_empty()
    {
        return None;
    }

    let search =
        BoundaryExactCoverSearch::new(cavity, candidates, options.volume_relative_tolerance);
    let mut mate_faces = BTreeSet::<[u32; 3]>::new();
    mate_faces.extend(no_candidate_dead_end_faces);
    mate_faces.extend(open_interior_dead_end_faces);
    mate_faces.extend(root_blocked_boundary_mate_faces);
    for dead_end in future_mate_dead_ends {
        add_future_mate_faces(
            &search,
            candidates,
            boundary_faces,
            dead_end,
            &mut mate_faces,
        );
    }
    Some(mate_faces)
}

fn add_future_mate_faces(
    search: &BoundaryExactCoverSearch<'_>,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    boundary_faces: &BTreeSet<[u32; 3]>,
    dead_end: &BoundaryExactCoverDeadEnd,
    mate_faces: &mut BTreeSet<[u32; 3]>,
) {
    let Some(face) = dead_end.face else {
        return;
    };
    let selected_indices = dead_end
        .selected_tetrahedra
        .iter()
        .filter_map(|selected_tetrahedron| {
            candidates.iter().position(|candidate| {
                sorted_tetrahedron_nodes(candidate.node_ids)
                    == sorted_tetrahedron_nodes(*selected_tetrahedron)
            })
        })
        .collect::<Vec<_>>();
    let face_counts = selected_face_counts(candidates, &selected_indices);
    for candidate_index in (0..candidates.len()).filter(|candidate_index| {
        !selected_indices.contains(candidate_index)
            && search.candidate_faces[*candidate_index].contains(&face)
    }) {
        for candidate_face in search.candidate_faces[candidate_index] {
            if !boundary_faces.contains(&candidate_face)
                && face_counts.get(&candidate_face).copied().unwrap_or(0) == 0
                && !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected_indices,
                )
            {
                mate_faces.insert(candidate_face);
            }
        }
    }
}

fn selected_face_counts(
    candidates: &[ConstrainedCavityRefillTetrahedron],
    selected_indices: &[usize],
) -> BTreeMap<[u32; 3], usize> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for selected_index in selected_indices {
        for selected_face in
            tetrahedron_faces(candidates[*selected_index].node_ids).map(sorted_face)
        {
            *face_counts.entry(selected_face).or_default() += 1;
        }
    }
    face_counts
}

fn open_interior_faces_from_tetrahedron_node_ids(tetrahedra: &[[u32; 4]]) -> Vec<[u32; 3]> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(*tetrahedron).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

fn root_boundary_future_mate_faces(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    volume_relative_tolerance: f64,
) -> Vec<[u32; 3]> {
    let search = BoundaryExactCoverSearch::new(cavity, candidates, volume_relative_tolerance);
    let face_counts = BTreeMap::<[u32; 3], usize>::new();
    let selected = Vec::<usize>::new();
    let mut mate_faces = BTreeSet::<[u32; 3]>::new();
    for boundary_face in &search.boundary_faces {
        for candidate_index in 0..candidates.len() {
            if !search.candidate_faces[candidate_index].contains(boundary_face) {
                continue;
            }
            for candidate_face in search.candidate_faces[candidate_index] {
                if search.boundary_faces.contains(&candidate_face) {
                    continue;
                }
                if !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                ) {
                    mate_faces.insert(candidate_face);
                }
            }
        }
    }
    mate_faces.into_iter().collect()
}
