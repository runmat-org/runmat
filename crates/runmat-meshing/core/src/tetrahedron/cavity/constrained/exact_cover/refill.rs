use super::*;

pub(in super::super) fn boundary_node_exact_cover_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() < 4
        || node_ids.len() > MAX_BOUNDARY_EXACT_COVER_NODES
        || cavity.boundary_faces.len() > MAX_BOUNDARY_EXACT_COVER_FACES
    {
        return Ok(None);
    }
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    let touches_boundary = tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    let points = tetrahedron_node_ids.map(|node_id| boundary_nodes[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        if touches_boundary
                            && candidate_keys.insert(sorted_tetrahedron_nodes(tetrahedron.node_ids))
                        {
                            candidates.push(tetrahedron.clone());
                        }
                        all_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    if candidates.is_empty() || candidates.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
        return Ok(None);
    }
    if let Some(refill) =
        exact_cover_refill_from_candidate_tetrahedra(cavity, &candidates, options)?
    {
        return Ok(Some(refill));
    }
    exact_cover_refill_from_on_demand_interior_mates(cavity, candidates, all_candidates, options)
}

pub(in super::super) fn exact_cover_refill_from_candidate_tetrahedra(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    if candidates.is_empty() {
        return Ok(None);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, candidates, options.volume_relative_tolerance);
    let Some(selected_indices) = search.search_best() else {
        return Ok(None);
    };
    let selected_tetrahedra = selected_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect::<Vec<_>>();
    refill_from_tetrahedra(
        cavity,
        selected_tetrahedra,
        options.volume_relative_tolerance,
    )
    .map(Some)
}

pub(in super::super) fn exact_cover_refill_from_on_demand_interior_mates(
    cavity: &ConstrainedCavity,
    mut candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    all_candidates: Vec<ConstrainedCavityRefillTetrahedron>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    if candidates.is_empty() || candidates.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
        return Ok(None);
    }
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidate_keys = candidates
        .iter()
        .map(|candidate| sorted_tetrahedron_nodes(candidate.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }

    for _ in 0..64 {
        let (selected, trace) = {
            let mut search = BoundaryExactCoverSearch::new(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            search.search_with_trace()
        };
        if let Some(selected) = selected {
            let selected_tetrahedra = selected
                .into_iter()
                .map(|index| candidates[index].clone())
                .collect::<Vec<_>>();
            return refill_from_tetrahedra(
                cavity,
                selected_tetrahedra,
                options.volume_relative_tolerance,
            )
            .map(Some);
        }

        let future_mate_dead_ends = trace
            .dead_ends
            .iter()
            .filter(|dead_end| dead_end.reason == "forced_interior_mate_future_mate_conflict")
            .cloned()
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
                root_boundary_future_mate_faces(
                    cavity,
                    &candidates,
                    options.volume_relative_tolerance,
                )
            })
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
        if future_mate_dead_ends.is_empty()
            && no_candidate_dead_end_faces.is_empty()
            && open_interior_dead_end_faces.is_empty()
            && root_blocked_boundary_mate_faces.is_empty()
        {
            return Ok(None);
        }

        let search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let mut mate_faces = BTreeSet::<[u32; 3]>::new();
        mate_faces.extend(no_candidate_dead_end_faces);
        mate_faces.extend(open_interior_dead_end_faces);
        mate_faces.extend(root_blocked_boundary_mate_faces);
        for dead_end in &future_mate_dead_ends {
            let Some(face) = dead_end.face else {
                continue;
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
            let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
            for selected_index in &selected_indices {
                for selected_face in
                    tetrahedron_faces(candidates[*selected_index].node_ids).map(sorted_face)
                {
                    *face_counts.entry(selected_face).or_default() += 1;
                }
            }
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

        let mut added = false;
        for mate_face in mate_faces {
            let Some(indices) = all_candidates_by_face.get(&mate_face) else {
                continue;
            };
            let mut indices = indices.clone();
            indices.sort_by(|left, right| {
                all_candidates[*right]
                    .exact_scaled_jacobian
                    .total_cmp(&all_candidates[*left].exact_scaled_jacobian)
            });
            for index in indices {
                if candidates.len() >= MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                    break;
                }
                let candidate = &all_candidates[index];
                if candidate_keys.insert(sorted_tetrahedron_nodes(candidate.node_ids)) {
                    candidates.push(candidate.clone());
                    added = true;
                    break;
                }
            }
        }
        if !added {
            return Ok(None);
        }
    }

    Ok(None)
}

pub(super) fn open_interior_faces_from_tetrahedron_node_ids(
    tetrahedra: &[[u32; 4]],
) -> Vec<[u32; 3]> {
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

pub(super) fn root_boundary_future_mate_faces(
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
