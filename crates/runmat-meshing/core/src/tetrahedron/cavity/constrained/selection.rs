use std::collections::{BTreeMap, BTreeSet};

use super::{
    topology::{
        boundary_face_map, face_edges, sorted_edge, sorted_face, tetrahedron_edges,
        tetrahedron_faces,
    },
    validate_constrained_cavity, CavityTetrahedron, ConstrainedCavity,
    ConstrainedCavityBoundaryEdgeRecovery, ConstrainedCavityBoundaryEdgeRecoveryQueue,
    ConstrainedCavityBoundaryEdgeRecoveryStep, ConstrainedCavityBoundaryFace,
    ConstrainedCavityExpansionError, ConstrainedCavityExtractionError,
    ConstrainedCavityRefillTetrahedron, ConstrainedCavityValidationError, MAX_ANCHOR_TRIM_STATES,
    MAX_CONSTRAINED_CAVITY_EXPANSION_STEPS,
};

mod anchor_trim;
use anchor_trim::anchor_trimmed_constrained_cavity;
mod edge_recovery;
pub use edge_recovery::{
    constrained_cavity_expanded_across_boundary_edge_star_excluding_nodes,
    constrained_cavity_expanded_across_boundary_faces_or_recovered_edge_star,
    constrained_cavity_recovered_boundary_edge_star_excluding_nodes,
    constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes,
};

pub fn constrained_cavity_from_selected_tetrahedra(
    tetrahedra: &[CavityTetrahedron],
    selected_tetrahedron_indices: &[usize],
    protected_node_ids: Vec<u32>,
) -> Result<ConstrainedCavity, ConstrainedCavityExtractionError> {
    let selected = selected_tetrahedron_index_set(tetrahedra, selected_tetrahedron_indices)?;
    let cavity = build_constrained_cavity_from_index_set(tetrahedra, &selected, protected_node_ids);
    validate_constrained_cavity(&cavity).map_err(ConstrainedCavityExtractionError::Validation)?;
    Ok(cavity)
}

pub fn constrained_cavity_from_refill_tetrahedron_component(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    inherited_boundary_faces: &[ConstrainedCavityBoundaryFace],
    protected_node_ids: Vec<u32>,
) -> Result<ConstrainedCavity, ConstrainedCavityValidationError> {
    let inherited_faces = boundary_face_map(inherited_boundary_faces)?;
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for face in tetrahedron_faces(tetrahedron.node_ids) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    let boundary_faces = face_counts
        .into_iter()
        .filter_map(|(face, count)| {
            (count == 1).then(|| {
                inherited_faces
                    .get(&face)
                    .map(|source| (*source).clone())
                    .unwrap_or(ConstrainedCavityBoundaryFace {
                        node_ids: face,
                        outside_tetrahedron_ids: Vec::new(),
                        source_face_id: None,
                        source_edge_ids: [None, None, None],
                        region_ids: Vec::new(),
                    })
            })
        })
        .collect::<Vec<_>>();
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: (0..tetrahedra.len()).map(|index| index as u32).collect(),
        boundary_faces,
        protected_node_ids,
        target_volume_m3: tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
    };
    validate_constrained_cavity(&cavity)?;
    Ok(cavity)
}

pub fn constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
    tetrahedra: &[CavityTetrahedron],
    selected_tetrahedron_indices: &[usize],
    anchor_tetrahedron_index: usize,
    protected_node_ids: Vec<u32>,
) -> Result<Option<ConstrainedCavity>, ConstrainedCavityExtractionError> {
    if anchor_tetrahedron_index >= tetrahedra.len() {
        return Err(
            ConstrainedCavityExtractionError::SelectedTetrahedronIndexOutOfBounds {
                tetrahedron_index: anchor_tetrahedron_index,
                tetrahedron_count: tetrahedra.len(),
            },
        );
    }
    let selected = selected_tetrahedron_index_set(tetrahedra, selected_tetrahedron_indices)?;
    if !selected.contains(&anchor_tetrahedron_index) {
        return Ok(None);
    }

    anchor_trimmed_constrained_cavity(
        tetrahedra,
        selected,
        anchor_tetrahedron_index,
        protected_node_ids,
    )
}

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

fn selected_tetrahedron_index_set(
    tetrahedra: &[CavityTetrahedron],
    selected_tetrahedron_indices: &[usize],
) -> Result<BTreeSet<usize>, ConstrainedCavityExtractionError> {
    if selected_tetrahedron_indices.is_empty() {
        return Err(ConstrainedCavityExtractionError::EmptySelection);
    }

    let mut selected = BTreeSet::<usize>::new();
    for tetrahedron_index in selected_tetrahedron_indices {
        if *tetrahedron_index >= tetrahedra.len() {
            return Err(
                ConstrainedCavityExtractionError::SelectedTetrahedronIndexOutOfBounds {
                    tetrahedron_index: *tetrahedron_index,
                    tetrahedron_count: tetrahedra.len(),
                },
            );
        }
        if !selected.insert(*tetrahedron_index) {
            return Err(
                ConstrainedCavityExtractionError::DuplicateSelectedTetrahedronIndex {
                    tetrahedron_index: *tetrahedron_index,
                },
            );
        }
    }
    Ok(selected)
}

pub(super) fn build_constrained_cavity_from_index_set(
    tetrahedra: &[CavityTetrahedron],
    selected: &BTreeSet<usize>,
    protected_node_ids: Vec<u32>,
) -> ConstrainedCavity {
    let mut target_volume_m3 = 0.0_f64;
    let mut face_owners = BTreeMap::<[u32; 3], Vec<(usize, [u32; 3])>>::new();
    let mut all_face_owners = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (tetrahedron_index, tetrahedron) in tetrahedra.iter().enumerate() {
        for face in tetrahedron_faces(tetrahedron.node_ids) {
            all_face_owners
                .entry(sorted_face(face))
                .or_default()
                .push(tetrahedron_index);
        }
    }
    for tetrahedron_index in selected {
        let tetrahedron = &tetrahedra[*tetrahedron_index];
        target_volume_m3 += tetrahedron.volume_m3;
        for face in tetrahedron_faces(tetrahedron.node_ids) {
            face_owners
                .entry(sorted_face(face))
                .or_default()
                .push((*tetrahedron_index, face));
        }
    }

    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    for owners in face_owners.values() {
        if owners.len() != 1 {
            continue;
        }
        let (tetrahedron_index, oriented_face) = owners[0];
        let mut outside_tetrahedron_ids = all_face_owners
            .get(&sorted_face(oriented_face))
            .into_iter()
            .flat_map(|owners| owners.iter())
            .filter_map(|owner_index| {
                (!selected.contains(owner_index)).then_some(tetrahedra[*owner_index].tetrahedron_id)
            })
            .collect::<Vec<_>>();
        outside_tetrahedron_ids.sort_unstable();
        outside_tetrahedron_ids.dedup();
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: oriented_face,
            outside_tetrahedron_ids,
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: tetrahedra[tetrahedron_index].region_ids.clone(),
        });
    }

    ConstrainedCavity {
        removed_tetrahedron_ids: selected
            .iter()
            .map(|tetrahedron_index| tetrahedra[*tetrahedron_index].tetrahedron_id)
            .collect(),
        boundary_faces,
        protected_node_ids,
        target_volume_m3,
    }
}
