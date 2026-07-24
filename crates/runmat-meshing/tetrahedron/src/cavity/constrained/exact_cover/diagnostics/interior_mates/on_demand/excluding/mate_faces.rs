use super::*;

pub(super) fn on_demand_interior_mate_faces_for_trace(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
    boundary_faces: &BTreeSet<[u32; 3]>,
    no_candidate_dead_end_faces: BTreeSet<[u32; 3]>,
    future_mate_dead_ends: &[BoundaryExactCoverDeadEnd],
) -> BTreeSet<[u32; 3]> {
    let search =
        BoundaryExactCoverSearch::new(cavity, candidates, options.volume_relative_tolerance);
    let mut mate_faces = no_candidate_dead_end_faces;
    for future_dead_end in future_mate_dead_ends {
        let Some(face) = future_dead_end.face else {
            continue;
        };
        let selected_indices = future_dead_end
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
    mate_faces
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
