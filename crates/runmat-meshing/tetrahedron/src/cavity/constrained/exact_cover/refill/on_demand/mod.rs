use super::*;

mod mate_faces;
pub(in crate::cavity::constrained) use mate_faces::on_demand_interior_mate_faces_for_trace;

pub(in super::super::super) fn exact_cover_refill_from_on_demand_interior_mates(
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
            let Ok(result) =
                search.search_with_trace_controlled(&runmat_meshing_core::NeverCancelled, u64::MAX)
            else {
                return Ok(None);
            };
            result
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

        let Some(mate_faces) = mate_faces::on_demand_interior_mate_faces_for_trace(
            cavity,
            &candidates,
            options,
            &boundary_faces,
            &trace,
        ) else {
            return Ok(None);
        };

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
