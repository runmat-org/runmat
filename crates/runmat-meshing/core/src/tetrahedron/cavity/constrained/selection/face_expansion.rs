use super::super::topology::{face_edges, sorted_edge, sorted_face, tetrahedron_edges};
use super::*;

pub fn constrained_cavity_expanded_across_boundary_face(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    boundary_face: [u32; 3],
) -> Result<ConstrainedCavity, ConstrainedCavityExpansionError> {
    constrained_cavity_expanded_across_boundary_faces(cavity, source_tetrahedra, &[boundary_face])
}

pub fn constrained_cavity_expanded_across_boundary_faces(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    boundary_faces: &[[u32; 3]],
) -> Result<ConstrainedCavity, ConstrainedCavityExpansionError> {
    let target_faces = boundary_faces
        .iter()
        .copied()
        .map(sorted_face)
        .collect::<Vec<_>>();

    let mut selected_tetrahedron_ids = cavity
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for target in &target_faces {
        let face = cavity
            .boundary_faces
            .iter()
            .find(|face| sorted_face(face.node_ids) == *target)
            .ok_or(ConstrainedCavityExpansionError::BoundaryFaceNotFound { node_ids: *target })?;
        if face.outside_tetrahedron_ids.is_empty() {
            return Err(
                ConstrainedCavityExpansionError::BoundaryFaceHasNoOutsideTetrahedron {
                    node_ids: *target,
                },
            );
        }
        selected_tetrahedron_ids.extend(face.outside_tetrahedron_ids.iter().copied());
    }
    let tetrahedron_id_to_index = source_tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| (tetrahedron.tetrahedron_id, index))
        .collect::<BTreeMap<_, _>>();

    for step in 0..MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS {
        let selected_indices = selected_tetrahedron_ids
            .iter()
            .map(|tetrahedron_id| {
                tetrahedron_id_to_index.get(tetrahedron_id).copied().ok_or(
                    ConstrainedCavityExpansionError::SourceTetrahedronIdNotFound {
                        tetrahedron_id: *tetrahedron_id,
                    },
                )
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let expanded = build_constrained_cavity_from_index_set(
            source_tetrahedra,
            &selected_indices,
            cavity.protected_node_ids.clone(),
        );
        match validate_constrained_cavity(&expanded) {
            Ok(_) => return Ok(expanded),
            Err(ConstrainedCavityValidationError::NonManifoldBoundaryEdge { node_ids, .. }) => {
                let mut added = false;
                for boundary in &expanded.boundary_faces {
                    let touches_edge = face_edges(boundary.node_ids)
                        .into_iter()
                        .any(|edge| sorted_edge(edge) == node_ids);
                    if !touches_edge {
                        continue;
                    }
                    for tetrahedron_id in &boundary.outside_tetrahedron_ids {
                        added |= selected_tetrahedron_ids.insert(*tetrahedron_id);
                    }
                }
                if !added {
                    for tetrahedron in source_tetrahedra {
                        if selected_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id) {
                            continue;
                        }
                        let touches_edge = tetrahedron_edges(tetrahedron.node_ids)
                            .into_iter()
                            .any(|edge| sorted_edge(edge) == node_ids);
                        if touches_edge {
                            added |= selected_tetrahedron_ids.insert(tetrahedron.tetrahedron_id);
                        }
                    }
                }
                if !added {
                    return Err(
                        ConstrainedCavityExpansionError::BoundaryEdgeHasNoOutsideTetrahedron {
                            node_ids,
                        },
                    );
                }
            }
            Err(err) => {
                return Err(ConstrainedCavityExpansionError::Extraction(
                    ConstrainedCavityExtractionError::Validation(err),
                ));
            }
        }
        if step + 1 == MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS {
            return Err(ConstrainedCavityExpansionError::ExpansionDidNotConverge {
                step_count: MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS,
            });
        }
    }

    Err(ConstrainedCavityExpansionError::ExpansionDidNotConverge {
        step_count: MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS,
    })
}

pub fn constrained_cavity_expanded_across_first_valid_boundary_face(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    boundary_faces: &[[u32; 3]],
) -> Result<Option<ConstrainedCavity>, ConstrainedCavityExpansionError> {
    for boundary_face in boundary_faces {
        match constrained_cavity_expanded_across_boundary_face(
            cavity,
            source_tetrahedra,
            *boundary_face,
        ) {
            Ok(expanded) => return Ok(Some(expanded)),
            Err(ConstrainedCavityExpansionError::SourceTetrahedronIdNotFound {
                tetrahedron_id,
            }) => {
                return Err(
                    ConstrainedCavityExpansionError::SourceTetrahedronIdNotFound { tetrahedron_id },
                );
            }
            Err(
                ConstrainedCavityExpansionError::BoundaryFaceNotFound { .. }
                | ConstrainedCavityExpansionError::BoundaryFaceHasNoOutsideTetrahedron { .. }
                | ConstrainedCavityExpansionError::BoundaryEdgeHasNoOutsideTetrahedron { .. }
                | ConstrainedCavityExpansionError::ExpansionDidNotConverge { .. }
                | ConstrainedCavityExpansionError::Extraction(_),
            ) => continue,
        }
    }
    Ok(None)
}
