use super::*;

pub fn constrained_cavity_expanded_across_boundary_faces_or_recovered_edge_star(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    boundary_faces: &[[u32; 3]],
    excluded_node_ids: &[u32],
) -> Result<ConstrainedCavityBoundaryEdgeRecovery, ConstrainedCavityExpansionError> {
    let attempted_boundary_faces = boundary_faces
        .iter()
        .copied()
        .map(sorted_face)
        .collect::<Vec<_>>();
    match constrained_cavity_expanded_across_boundary_faces(
        cavity,
        source_tetrahedra,
        boundary_faces,
    ) {
        Ok(expanded) => Ok(ConstrainedCavityBoundaryEdgeRecovery {
            cavity: expanded,
            attempted_boundary_faces,
            recovered_edge: None,
        }),
        Err(ConstrainedCavityExpansionError::BoundaryEdgeHasNoOutsideTetrahedron { node_ids }) => {
            let expanded = constrained_cavity_expanded_across_boundary_edge_star_excluding_nodes(
                cavity,
                source_tetrahedra,
                node_ids,
                excluded_node_ids,
            )?;
            let before = cavity
                .removed_tetrahedron_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let added_tetrahedron_ids = expanded
                .removed_tetrahedron_ids
                .iter()
                .copied()
                .filter(|tetrahedron_id| !before.contains(tetrahedron_id))
                .collect::<Vec<_>>();
            Ok(ConstrainedCavityBoundaryEdgeRecovery {
                recovered_edge: Some(ConstrainedCavityBoundaryEdgeRecoveryStep {
                    node_ids,
                    added_tetrahedron_ids,
                    removed_tetrahedron_count_before: cavity.removed_tetrahedron_ids.len(),
                    removed_tetrahedron_count_after: expanded.removed_tetrahedron_ids.len(),
                }),
                cavity: expanded,
                attempted_boundary_faces,
            })
        }
        Err(err) => Err(err),
    }
}

pub fn constrained_cavity_recovered_boundary_edge_star_excluding_nodes(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    edge: [u32; 2],
    excluded_node_ids: &[u32],
) -> Result<ConstrainedCavityBoundaryEdgeRecovery, ConstrainedCavityExpansionError> {
    let target_edge = sorted_edge(edge);
    let expanded = constrained_cavity_expanded_across_boundary_edge_star_excluding_nodes(
        cavity,
        source_tetrahedra,
        target_edge,
        excluded_node_ids,
    )?;
    let before = cavity
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let added_tetrahedron_ids = expanded
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .filter(|tetrahedron_id| !before.contains(tetrahedron_id))
        .collect::<Vec<_>>();
    let removed_tetrahedron_count_after = expanded.removed_tetrahedron_ids.len();
    Ok(ConstrainedCavityBoundaryEdgeRecovery {
        cavity: expanded,
        attempted_boundary_faces: Vec::new(),
        recovered_edge: Some(ConstrainedCavityBoundaryEdgeRecoveryStep {
            node_ids: target_edge,
            added_tetrahedron_ids,
            removed_tetrahedron_count_before: cavity.removed_tetrahedron_ids.len(),
            removed_tetrahedron_count_after,
        }),
    })
}

pub fn constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    edges: &[[u32; 2]],
    excluded_node_ids: &[u32],
) -> Result<ConstrainedCavityBoundaryEdgeRecoveryQueue, ConstrainedCavityExpansionError> {
    let mut current = cavity.clone();
    let mut steps = Vec::<ConstrainedCavityBoundaryEdgeRecoveryStep>::new();
    for edge in edges {
        let recovery = constrained_cavity_recovered_boundary_edge_star_excluding_nodes(
            &current,
            source_tetrahedra,
            *edge,
            excluded_node_ids,
        )?;
        if let Some(step) = recovery.recovered_edge {
            steps.push(step);
        }
        current = recovery.cavity;
    }
    Ok(ConstrainedCavityBoundaryEdgeRecoveryQueue {
        cavity: current,
        steps,
    })
}

pub fn constrained_cavity_expanded_across_boundary_edge_star_excluding_nodes(
    cavity: &ConstrainedCavity,
    source_tetrahedra: &[CavityTetrahedron],
    edge: [u32; 2],
    excluded_node_ids: &[u32],
) -> Result<ConstrainedCavity, ConstrainedCavityExpansionError> {
    let target_edge = sorted_edge(edge);
    let excluded_node_ids = excluded_node_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut selected_tetrahedron_ids = cavity
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut added = false;
    for tetrahedron in source_tetrahedra {
        if tetrahedron
            .node_ids
            .into_iter()
            .any(|node_id| excluded_node_ids.contains(&node_id))
        {
            continue;
        }
        let touches_edge = tetrahedron_edges(tetrahedron.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == target_edge);
        if touches_edge {
            added |= selected_tetrahedron_ids.insert(tetrahedron.tetrahedron_id);
        }
    }
    if !added {
        return Err(
            ConstrainedCavityExpansionError::BoundaryEdgeHasNoOutsideTetrahedron {
                node_ids: target_edge,
            },
        );
    }

    let tetrahedron_id_to_index = source_tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| (tetrahedron.tetrahedron_id, index))
        .collect::<BTreeMap<_, _>>();
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
    validate_constrained_cavity(&expanded).map_err(|err| {
        ConstrainedCavityExpansionError::Extraction(ConstrainedCavityExtractionError::Validation(
            err,
        ))
    })?;
    Ok(expanded)
}
