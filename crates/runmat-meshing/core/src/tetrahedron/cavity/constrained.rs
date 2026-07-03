#![cfg_attr(test, allow(dead_code))]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    predicate::{
        distance_squared, orient_tetrahedron_node_ids, point_in_closed_triangle_surface,
        tetrahedron_centroid, tetrahedron_circumsphere, tetrahedron_edge_aspect_ratio,
        tetrahedron_scaled_jacobian, tetrahedron_signed_volume, triangle_area, Point3,
        PointInClosedSurface, Triangle3,
    },
    tetrahedron::reconnect::{
        evaluate_local_tetrahedron_flip_quality, three_to_two_edge_flip_candidate,
        two_to_three_face_flip_candidate, LocalTetrahedron, LocalTetrahedronFlipCandidate,
        LocalTetrahedronFlipError, LocalTetrahedronFlipQualityThresholds,
    },
    tolerance::MeshingTolerance,
};

mod boundary_completion;
mod boundary_splits;
mod cap_connectors;
mod caps;
mod connectivity;
#[cfg(test)]
mod diagnostic_metrics;
mod geometry;
mod missing_faces;
mod refill_faces;
mod topology;
mod types;

use boundary_completion::*;
use boundary_splits::*;
use cap_connectors::*;
use caps::*;
use connectivity::*;
#[cfg(test)]
use diagnostic_metrics::*;
use geometry::*;
use missing_faces::*;
use refill_faces::*;
use topology::*;
pub use types::*;

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

pub fn split_refill_tetrahedra_across_shared_face_at_barycentric(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
    face: [u32; 3],
    barycentric: [f64; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    (
        Vec<ConstrainedCavityRefillTetrahedron>,
        ConstrainedCavityNode,
    ),
    ConstrainedCavityRefillTetrahedronSplitError,
> {
    let barycentric_sum = barycentric.iter().sum::<f64>();
    if barycentric
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || (barycentric_sum - 1.0).abs() > 1.0e-12
    {
        return Err(
            ConstrainedCavityRefillTetrahedronSplitError::InvalidBarycentricCoordinates {
                barycentric,
            },
        );
    }
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !node_map.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillTetrahedronSplitError::MissingNode { node_id });
            }
        }
    }
    let target_face = sorted_face(face);
    for node_id in target_face {
        if !node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityRefillTetrahedronSplitError::MissingNode { node_id });
        }
    }
    let incident_tetrahedron_indices = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            tetrahedron_faces(tetrahedron.node_ids)
                .map(sorted_face)
                .contains(&target_face)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if incident_tetrahedron_indices.len() != 2 {
        return Err(
            ConstrainedCavityRefillTetrahedronSplitError::FaceIncidenceNotTwo {
                node_ids: target_face,
                incident_tetrahedron_count: incident_tetrahedron_indices.len(),
            },
        );
    }
    let mut split_node_id = node_map
        .keys()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    while node_map.contains_key(&split_node_id) {
        split_node_id = split_node_id.saturating_add(1);
    }
    let face_points = target_face.map(|node_id| node_map[&node_id]);
    let split_node = ConstrainedCavityNode {
        node_id: split_node_id,
        coordinates_m: [
            face_points[0][0] * barycentric[0]
                + face_points[1][0] * barycentric[1]
                + face_points[2][0] * barycentric[2],
            face_points[0][1] * barycentric[0]
                + face_points[1][1] * barycentric[1]
                + face_points[2][1] * barycentric[2],
            face_points[0][2] * barycentric[0]
                + face_points[1][2] * barycentric[1]
                + face_points[2][2] * barycentric[2],
        ],
    };
    let mut split_node_map = node_map;
    split_node_map.insert(split_node.node_id, split_node.coordinates_m);
    let incident_tetrahedron_indices = incident_tetrahedron_indices
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut split_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for (index, tetrahedron) in tetrahedra.iter().enumerate() {
        if !incident_tetrahedron_indices.contains(&index) {
            split_tetrahedra.push(tetrahedron.clone());
            continue;
        }
        let opposite_node = tetrahedron
            .node_ids
            .into_iter()
            .find(|node_id| !target_face.contains(node_id))
            .expect("incident tetrahedron should have an opposite node");
        for child_node_ids in [
            [
                target_face[0],
                target_face[1],
                split_node.node_id,
                opposite_node,
            ],
            [
                target_face[1],
                target_face[2],
                split_node.node_id,
                opposite_node,
            ],
            [
                target_face[2],
                target_face[0],
                split_node.node_id,
                opposite_node,
            ],
        ] {
            let points = child_node_ids.map(|node_id| split_node_map[&node_id]);
            match raw_refill_tetrahedron_with_rejection_reason(child_node_ids, points, options) {
                Ok(child) => split_tetrahedra.push(child),
                Err(reason) => {
                    return Err(
                        ConstrainedCavityRefillTetrahedronSplitError::RejectedChildTetrahedron {
                            node_ids: child_node_ids,
                            reason,
                        },
                    );
                }
            }
        }
    }
    Ok((split_tetrahedra, split_node))
}

pub fn flip_refill_tetrahedra_across_shared_face(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
    face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityRefillTetrahedronFlipError> {
    let node_map = refill_component_node_map(tetrahedra, nodes)?;
    let target_face = sorted_face(face);
    for node_id in target_face {
        if !node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id });
        }
    }
    let incident_tetrahedron_indices = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            tetrahedron_faces(tetrahedron.node_ids)
                .map(sorted_face)
                .contains(&target_face)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if incident_tetrahedron_indices.len() != 2 {
        return Err(
            ConstrainedCavityRefillTetrahedronFlipError::FaceIncidenceNotTwo {
                node_ids: target_face,
                incident_tetrahedron_count: incident_tetrahedron_indices.len(),
            },
        );
    }
    let left_index = incident_tetrahedron_indices[0];
    let right_index = incident_tetrahedron_indices[1];
    let flip = two_to_three_face_flip_candidate(
        LocalTetrahedron {
            tetrahedron_id: left_index as u32,
            node_ids: tetrahedra[left_index].node_ids,
        },
        LocalTetrahedron {
            tetrahedron_id: right_index as u32,
            node_ids: tetrahedra[right_index].node_ids,
        },
    )
    .map_err(
        |err| ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
            reason: local_tetrahedron_flip_error_reason(&err),
        },
    )?;
    refill_tetrahedra_from_flip_candidate(tetrahedra, &node_map, &flip, options)
}

pub fn flip_refill_tetrahedra_around_shared_edge(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityRefillTetrahedronFlipError> {
    let node_map = refill_component_node_map(tetrahedra, nodes)?;
    let target_edge = sorted_edge(edge);
    for node_id in target_edge {
        if !node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id });
        }
    }
    let incident_tetrahedron_indices = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            (tetrahedron.node_ids.contains(&target_edge[0])
                && tetrahedron.node_ids.contains(&target_edge[1]))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    if incident_tetrahedron_indices.len() != 3 {
        return Err(
            ConstrainedCavityRefillTetrahedronFlipError::EdgeIncidenceNotThree {
                node_ids: target_edge,
                incident_tetrahedron_count: incident_tetrahedron_indices.len(),
            },
        );
    }
    let flip = three_to_two_edge_flip_candidate(
        [
            LocalTetrahedron {
                tetrahedron_id: incident_tetrahedron_indices[0] as u32,
                node_ids: tetrahedra[incident_tetrahedron_indices[0]].node_ids,
            },
            LocalTetrahedron {
                tetrahedron_id: incident_tetrahedron_indices[1] as u32,
                node_ids: tetrahedra[incident_tetrahedron_indices[1]].node_ids,
            },
            LocalTetrahedron {
                tetrahedron_id: incident_tetrahedron_indices[2] as u32,
                node_ids: tetrahedra[incident_tetrahedron_indices[2]].node_ids,
            },
        ],
        target_edge,
    )
    .map_err(
        |err| ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
            reason: local_tetrahedron_flip_error_reason(&err),
        },
    )?;
    refill_tetrahedra_from_flip_candidate(tetrahedra, &node_map, &flip, options)
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

pub fn constrained_cavity_refill_pressure_boundary_faces(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<[u32; 3]>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    if node_ids.len() < 4 || boundary_faces.is_empty() {
        return Ok(Vec::new());
    }

    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut face_candidate_counts = boundary_faces
        .iter()
        .map(|face| (*face, 0_usize))
        .collect::<BTreeMap<_, _>>();
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
                    let tetrahedron_boundary_faces =
                        tetrahedron_faces(tetrahedron_node_ids).map(sorted_face);
                    if !tetrahedron_boundary_faces
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    for face in tetrahedron_boundary_faces {
                        if let Some(count) = face_candidate_counts.get_mut(&face) {
                            *count += 1;
                        }
                    }
                }
            }
        }
    }
    let min_count = face_candidate_counts
        .values()
        .copied()
        .min()
        .unwrap_or_default();
    Ok(face_candidate_counts
        .into_iter()
        .filter_map(|(face, count)| (count == min_count).then_some(face))
        .collect())
}

fn anchor_trimmed_constrained_cavity(
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

fn build_constrained_cavity_from_index_set(
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

pub fn generate_constrained_cavity_refill_candidates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidate_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefill, ConstrainedCavityRefillError> {
    let evaluation = evaluate_constrained_cavity_refill_candidates(
        cavity,
        boundary_nodes,
        interior_candidate_nodes,
        options,
    )?;
    evaluation
        .refill
        .ok_or(ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: evaluation.rejected_by_reason,
        })
}

pub fn evaluate_constrained_cavity_refill_candidates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidate_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefillEvaluation, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut rejected_by_reason = BTreeMap::<String, usize>::new();

    if interior_candidate_nodes.is_empty() {
        if boundary_node_ids.len() == 4 {
            let Some(refill) =
                single_tetrahedron_refill_candidate(cavity, &boundary_node_map, options)
                    .map_err(ConstrainedCavityRefillError::Validation)?
            else {
                record_refill_rejection(
                    &mut rejected_by_reason,
                    "single_tetrahedron_refill_rejected",
                );
                return Ok(ConstrainedCavityRefillEvaluation {
                    refill: None,
                    rejected_by_reason,
                });
            };
            return Ok(ConstrainedCavityRefillEvaluation {
                refill: Some(refill),
                rejected_by_reason,
            });
        };
        match boundary_node_refill_candidate(cavity, &boundary_node_map, options) {
            Ok(Ok(refill)) => {
                return Ok(ConstrainedCavityRefillEvaluation {
                    refill: Some(refill),
                    rejected_by_reason,
                });
            }
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
        match centroid_interior_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) {
            Ok(Ok(refill)) => {
                return Ok(ConstrainedCavityRefillEvaluation {
                    refill: Some(refill),
                    rejected_by_reason,
                });
            }
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
        return Ok(ConstrainedCavityRefillEvaluation {
            refill: None,
            rejected_by_reason,
        });
    }

    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let tolerance = MeshingTolerance::default();
    let mut best = None::<ConstrainedCavityRefill>;
    let mut valid_interior_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in interior_candidate_nodes {
        if !seen_interior_nodes.insert(node.node_id) {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
        if boundary_node_ids.contains(&node.node_id) {
            return Err(
                ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode {
                    node_id: node.node_id,
                },
            );
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            record_refill_rejection(&mut rejected_by_reason, "protected_boundary_distance");
            continue;
        }
        if point_in_closed_triangle_surface(node.coordinates_m, &boundary_triangles, tolerance)
            != PointInClosedSurface::Inside
        {
            record_refill_rejection(&mut rejected_by_reason, "interior_point_outside_cavity");
            continue;
        }
        valid_interior_nodes.push(node.clone());
        let refill = match star_refill_candidate_with_rejection_reason(
            cavity,
            &boundary_node_map,
            node.clone(),
            options,
        ) {
            Ok(Ok(refill)) => refill,
            Ok(Err(reason)) => {
                record_refill_rejection(&mut rejected_by_reason, reason);
                continue;
            }
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err));
                continue;
            }
        };
        if best
            .as_ref()
            .is_none_or(|candidate| refill_is_better(&refill, candidate))
        {
            best = Some(refill);
        }
    }
    if best.is_none() && valid_interior_nodes.len() >= 2 {
        match two_interior_node_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            &valid_interior_nodes,
            options,
        ) {
            Ok(Ok(refill)) => best = Some(refill),
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }
    if best.is_none() && valid_interior_nodes.len() >= 3 {
        match multi_interior_node_refill_candidate(
            cavity,
            &boundary_node_map,
            &boundary_triangles,
            &valid_interior_nodes,
            options,
        ) {
            Ok(Ok(refill)) => best = Some(refill),
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }
    if best.is_none() && boundary_node_ids.len() > 4 {
        match boundary_node_refill_candidate(cavity, &boundary_node_map, options) {
            Ok(Ok(refill)) => best = Some(refill),
            Ok(Err(reason)) => record_refill_rejection(&mut rejected_by_reason, reason),
            Err(err) => {
                record_refill_rejection(&mut rejected_by_reason, refill_validation_reason(&err))
            }
        }
    }

    Ok(ConstrainedCavityRefillEvaluation {
        refill: best,
        rejected_by_reason,
    })
}

pub fn retriangulate_constrained_cavity_from_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let tolerance = MeshingTolerance::default();
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(node.coordinates_m, &boundary_triangles, tolerance)
            == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    if candidate_nodes.len() < 4
        || candidate_nodes.len() > MAX_BOUNDARY_EXACT_COVER_NODES
        || cavity.boundary_faces.len() > MAX_BOUNDARY_EXACT_COVER_FACES
    {
        return Ok(None);
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let mut candidate_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        tolerance,
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidate_tetrahedra.push(tetrahedron);
                    }
                    if candidate_tetrahedra.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                        return Ok(None);
                    }
                }
            }
        }
    }
    let inserted_node_ids = candidate_nodes
        .iter()
        .filter_map(|node| (!boundary_node_ids.contains(&node.node_id)).then_some(node.node_id))
        .collect::<BTreeSet<_>>();
    if !inserted_node_ids.is_empty() {
        append_cap_side_connector_chain_tetrahedra(
            &mut candidate_tetrahedra,
            &mut seen_tetrahedra,
            &node_map,
            &inserted_node_ids,
            &boundary_triangles,
            options,
        );
        if candidate_tetrahedra.len() > MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
            return Ok(None);
        }
    }
    let Some(mut refill) =
        exact_cover_refill_from_candidate_tetrahedra(cavity, &candidate_tetrahedra, options)
            .map_err(ConstrainedCavityRefillError::Validation)?
    else {
        return Ok(None);
    };
    let used_node_ids = refill
        .tetrahedra
        .iter()
        .flat_map(|tetrahedron| tetrahedron.node_ids)
        .collect::<BTreeSet<_>>();
    refill.inserted_nodes = candidate_nodes
        .into_iter()
        .filter(|node| !boundary_node_ids.contains(&node.node_id))
        .filter(|node| used_node_ids.contains(&node.node_id))
        .collect();
    Ok(Some(refill))
}

pub fn generate_constrained_cavity_component_steiner_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    component_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
    max_nodes: usize,
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    if max_nodes == 0 || component_tetrahedra.is_empty() {
        return Ok(Vec::new());
    }
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let cavity_centroid = cavity_boundary_node_centroid(cavity, &boundary_node_map).ok_or(
        ConstrainedCavityRefillError::Validation(
            ConstrainedCavityValidationError::TooFewBoundaryFaces {
                boundary_face_count: cavity.boundary_faces.len(),
            },
        ),
    )?;
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in component_tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !node_map.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillError::MissingBoundaryNode { node_id });
            }
        }
    }
    let mut scored_candidates = Vec::<(f64, Point3)>::new();
    let mut sorted_tetrahedra = component_tetrahedra.to_vec();
    sorted_tetrahedra.sort_by(|left, right| {
        left.exact_scaled_jacobian
            .total_cmp(&right.exact_scaled_jacobian)
            .then_with(|| left.aspect_ratio.total_cmp(&right.aspect_ratio).reverse())
    });
    for tetrahedron in sorted_tetrahedra {
        let tetrahedron_points = tetrahedron.node_ids.map(|node_id| node_map[&node_id]);
        let tetrahedron_centroid = tetrahedron_centroid(tetrahedron_points);
        component_steiner_candidate_points(
            tetrahedron.node_ids,
            tetrahedron_points,
            tetrahedron_centroid,
            cavity_centroid,
        )
        .into_iter()
        .filter(|point| {
            candidate_respects_protected_boundary_distance(
                cavity,
                &boundary_node_map,
                *point,
                options,
            )
        })
        .filter(|point| {
            point_in_closed_triangle_surface(
                *point,
                &boundary_triangles,
                MeshingTolerance::default(),
            ) == PointInClosedSurface::Inside
        })
        .for_each(|point| {
            let score = component_steiner_candidate_quality_score(
                tetrahedron.node_ids,
                tetrahedron_points,
                point,
            );
            if score.is_finite() && score > 0.0 {
                scored_candidates.push((score, point));
            }
        });
    }
    scored_candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1[0].total_cmp(&right.1[0]))
            .then_with(|| left.1[1].total_cmp(&right.1[1]))
            .then_with(|| left.1[2].total_cmp(&right.1[2]))
    });
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut selected_points = Vec::<Point3>::new();
    for (_, point) in scored_candidates {
        if existing_points
            .iter()
            .chain(selected_points.iter())
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        selected_points.push(point);
        if selected_points.len() >= max_nodes {
            break;
        }
    }
    let mut next_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    Ok(selected_points
        .into_iter()
        .map(|coordinates_m| {
            while node_ids.contains(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            let node = ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m,
            };
            next_node_id = next_node_id.saturating_add(1);
            node
        })
        .collect())
}

pub fn generate_constrained_cavity_patch_steiner_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    max_nodes: usize,
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    if max_nodes == 0 {
        return Ok(Vec::new());
    }
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if boundary_node_ids.len() < 4 {
        return Ok(Vec::new());
    }
    let boundary_points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&boundary_points) {
        let node_ids = tetrahedron
            .vertices
            .map(|index| boundary_points[index].node_id);
        let points = tetrahedron
            .vertices
            .map(|index| boundary_points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    if missing_faces.is_empty() {
        return Ok(Vec::new());
    }
    let cavity_centroid = cavity_boundary_node_centroid(cavity, &boundary_node_map).ok_or(
        ConstrainedCavityRefillError::Validation(
            ConstrainedCavityValidationError::TooFewBoundaryFaces {
                boundary_face_count: cavity.boundary_faces.len(),
            },
        ),
    )?;
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut scored_candidates = Vec::<(usize, f64, Point3)>::new();
    for component in missing_face_components(&missing_faces, MissingFaceLink::Node) {
        let mut patch_node_ids = BTreeSet::<u32>::new();
        for face_index in &component {
            patch_node_ids.extend(missing_faces[*face_index]);
        }
        let mut surface_points = Vec::<Point3>::new();
        if let Some(point) = centroid_of_node_set(&patch_node_ids, &boundary_node_map) {
            surface_points.push(point);
        }
        for face_index in &component {
            if let Some(point) = face_centroid(missing_faces[*face_index], &boundary_node_map) {
                surface_points.push(point);
            }
        }
        for surface_point in surface_points {
            for point in
                patch_steiner_candidate_points(surface_point, cavity_centroid, &boundary_triangles)
            {
                if !candidate_respects_protected_boundary_distance(
                    cavity,
                    &boundary_node_map,
                    point,
                    options,
                ) {
                    continue;
                }
                let nearest_distance = existing_points
                    .iter()
                    .map(|existing| distance_squared(*existing, point))
                    .fold(f64::INFINITY, f64::min);
                if nearest_distance.is_finite() && nearest_distance > 1.0e-24 {
                    scored_candidates.push((component.len(), nearest_distance, point));
                }
            }
        }
    }
    scored_candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.total_cmp(&left.1))
            .then_with(|| left.2[0].total_cmp(&right.2[0]))
            .then_with(|| left.2[1].total_cmp(&right.2[1]))
            .then_with(|| left.2[2].total_cmp(&right.2[2]))
    });
    let mut selected_points = Vec::<Point3>::new();
    for (_, _, point) in scored_candidates {
        if existing_points
            .iter()
            .chain(selected_points.iter())
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        selected_points.push(point);
        if selected_points.len() >= max_nodes {
            break;
        }
    }
    let mut next_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    Ok(selected_points
        .into_iter()
        .map(|coordinates_m| {
            while node_ids.contains(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            let node = ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m,
            };
            next_node_id = next_node_id.saturating_add(1);
            node
        })
        .collect())
}

pub fn generate_constrained_cavity_boundary_cap_nodes(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    max_nodes: usize,
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    if max_nodes == 0 {
        return Ok(Vec::new());
    }
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        return Ok(Vec::new());
    };
    let solid_empty_faces =
        solid_empty_boundary_faces(cavity, &boundary_node_map, &boundary_triangles, options);
    if solid_empty_faces.is_empty() {
        return Ok(Vec::new());
    }
    let cap_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let existing_points = nodes
        .iter()
        .map(|node| node.coordinates_m)
        .collect::<Vec<_>>();
    let mut scored_candidates = Vec::<(f64, [u32; 3], Point3, &'static str)>::new();
    for face in solid_empty_faces {
        let Some(surface_point) = face_centroid(face, &boundary_node_map) else {
            continue;
        };
        for candidate in
            local_cap_apex_candidates(face, surface_point, cavity_centroid, &boundary_node_map)
        {
            if existing_points
                .iter()
                .any(|existing| distance_squared(*existing, candidate.coordinates_m) <= 1.0e-24)
            {
                continue;
            }
            if !candidate_respects_protected_boundary_distance(
                cavity,
                &boundary_node_map,
                candidate.coordinates_m,
                options,
            ) {
                continue;
            }
            if point_in_closed_triangle_surface(
                candidate.coordinates_m,
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                continue;
            }
            let tetrahedron_points = [
                boundary_node_map[&face[0]],
                boundary_node_map[&face[1]],
                boundary_node_map[&face[2]],
                candidate.coordinates_m,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                continue;
            }
            let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], cap_node_id],
                tetrahedron_points,
                options,
            ) else {
                continue;
            };
            scored_candidates.push((
                tetrahedron.exact_scaled_jacobian,
                face,
                candidate.coordinates_m,
                candidate.source,
            ));
        }
    }
    scored_candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2[0].total_cmp(&right.2[0]))
            .then_with(|| left.2[1].total_cmp(&right.2[1]))
            .then_with(|| left.2[2].total_cmp(&right.2[2]))
            .then_with(|| left.3.cmp(right.3))
    });
    let mut selected_faces = BTreeSet::<[u32; 3]>::new();
    let mut selected_points = Vec::<Point3>::new();
    for (_, face, point, _) in scored_candidates {
        if selected_faces.contains(&face) {
            continue;
        }
        if existing_points
            .iter()
            .chain(selected_points.iter())
            .any(|existing| distance_squared(*existing, point) <= 1.0e-24)
        {
            continue;
        }
        selected_faces.insert(face);
        selected_points.push(point);
        if selected_points.len() >= max_nodes {
            break;
        }
    }
    let mut next_node_id = nodes
        .iter()
        .map(|node| node.node_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    Ok(selected_points
        .into_iter()
        .map(|coordinates_m| {
            while node_ids.contains(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            let node = ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m,
            };
            next_node_id = next_node_id.saturating_add(1);
            node
        })
        .collect())
}

pub fn constrained_cavity_solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<[u32; 3]>, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    Ok(solid_empty_boundary_faces(
        cavity,
        &boundary_node_map,
        &boundary_triangles,
        options,
    ))
}

pub fn constrained_cavity_classified_solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavitySolidEmptyBoundaryFaces, ConstrainedCavityRefillError> {
    let faces = constrained_cavity_solid_empty_boundary_faces(cavity, nodes, options)?;
    let boundary_faces = boundary_face_map(&cavity.boundary_faces)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let mut true_exterior_faces = Vec::<[u32; 3]>::new();
    let mut expandable_faces = Vec::<[u32; 3]>::new();
    for face in &faces {
        let Some(boundary_face) = boundary_faces.get(face) else {
            continue;
        };
        if boundary_face.outside_tetrahedron_ids.is_empty() {
            true_exterior_faces.push(*face);
        } else {
            expandable_faces.push(*face);
        }
    }
    Ok(ConstrainedCavitySolidEmptyBoundaryFaces {
        faces,
        true_exterior_faces,
        expandable_faces,
    })
}

pub fn recover_constrained_cavity_solid_empty_boundaries(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    source_tetrahedra: &[CavityTetrahedron],
    source_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<
    ConstrainedCavitySolidEmptyBoundaryRecovery,
    ConstrainedCavitySolidEmptyBoundaryRecoveryError,
> {
    let classification =
        constrained_cavity_classified_solid_empty_boundary_faces(cavity, nodes, options)
            .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
    let mut current_cavity = cavity.clone();
    let mut current_nodes = nodes.to_vec();
    let mut split_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut split_steps = Vec::<ConstrainedCavityBoundaryPatchSplitStep>::new();
    let mut rejected_splits = Vec::<ConstrainedCavitySolidEmptyBoundaryRejectedSplit>::new();
    let mut expanded_removed_tetrahedron_ids = Vec::<u32>::new();
    if !classification.expandable_faces.is_empty() {
        let before = current_cavity
            .removed_tetrahedron_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        current_cavity = constrained_cavity_expanded_across_boundary_faces(
            &current_cavity,
            source_tetrahedra,
            &classification.expandable_faces,
        )
        .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Expansion)?;
        expanded_removed_tetrahedron_ids = current_cavity
            .removed_tetrahedron_ids
            .iter()
            .copied()
            .filter(|tetrahedron_id| !before.contains(tetrahedron_id))
            .collect();
        current_nodes = constrained_cavity_boundary_nodes_from_sources(
            &current_cavity,
            &current_nodes,
            source_nodes,
        )
        .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
    }

    let split_classification = constrained_cavity_classified_solid_empty_boundary_faces(
        &current_cavity,
        &current_nodes,
        options,
    )
    .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
    if !split_classification.true_exterior_faces.is_empty() {
        let patch_split = split_constrained_cavity_boundary_patch_at_centroids(
            &current_cavity,
            &current_nodes,
            &[],
            &split_classification.true_exterior_faces,
        )
        .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Split)?;
        let mut split_candidate_nodes = current_nodes.clone();
        split_candidate_nodes.extend(patch_split.split_nodes.clone());
        let split_candidate_classification =
            constrained_cavity_classified_solid_empty_boundary_faces(
                &patch_split.cavity,
                &split_candidate_nodes,
                options,
            )
            .map_err(ConstrainedCavitySolidEmptyBoundaryRecoveryError::Refill)?;
        if split_candidate_classification.faces.len() <= split_classification.faces.len() {
            current_cavity = patch_split.cavity;
            split_nodes = patch_split.split_nodes;
            split_steps = patch_split.steps;
        } else {
            rejected_splits.push(ConstrainedCavitySolidEmptyBoundaryRejectedSplit {
                input_faces: split_classification.true_exterior_faces,
                output_faces: split_candidate_classification.faces,
                split_node_count: patch_split.split_nodes.len(),
                split_step_count: patch_split.steps.len(),
            });
        }
    }

    Ok(ConstrainedCavitySolidEmptyBoundaryRecovery {
        cavity: current_cavity,
        split_nodes,
        classification,
        split_steps,
        rejected_splits,
        expanded_removed_tetrahedron_ids,
    })
}

fn constrained_cavity_boundary_nodes_from_sources(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    source_nodes: &[ConstrainedCavityNode],
) -> Result<Vec<ConstrainedCavityNode>, ConstrainedCavityRefillError> {
    let mut coordinates = source_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    coordinates.extend(nodes.iter().map(|node| (node.node_id, node.coordinates_m)));
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| {
            coordinates
                .get(&node_id)
                .copied()
                .map(|coordinates_m| ConstrainedCavityNode {
                    node_id,
                    coordinates_m,
                })
                .ok_or(ConstrainedCavityRefillError::MissingBoundaryNode { node_id })
        })
        .collect()
}

fn solid_empty_boundary_faces(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Vec<[u32; 3]> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut solid_faces = BTreeSet::<[u32; 3]>::new();
    for first in 0..boundary_node_ids.len() {
        for second in (first + 1)..boundary_node_ids.len() {
            for third in (second + 1)..boundary_node_ids.len() {
                for fourth in (third + 1)..boundary_node_ids.len() {
                    let tetrahedron_node_ids = [
                        boundary_node_ids[first],
                        boundary_node_ids[second],
                        boundary_node_ids[third],
                        boundary_node_ids[fourth],
                    ];
                    let candidate_faces = tetrahedron_faces(tetrahedron_node_ids).map(sorted_face);
                    if !candidate_faces
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_nodes[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    )
                    .is_ok()
                    {
                        for face in candidate_faces {
                            if boundary_faces.contains(&face) {
                                solid_faces.insert(face);
                            }
                        }
                    }
                }
            }
        }
    }
    boundary_faces
        .into_iter()
        .filter(|face| !solid_faces.contains(face))
        .collect()
}

fn component_steiner_candidate_points(
    node_ids: [u32; 4],
    points: [Point3; 4],
    tetrahedron_centroid: Point3,
    cavity_centroid: Point3,
) -> Vec<Point3> {
    let mut candidates = Vec::<Point3>::new();
    candidates.push(tetrahedron_centroid);
    candidates.push(tetrahedron_incenter(points));
    if let Some((center, _)) = tetrahedron_circumsphere(points, MeshingTolerance::default()) {
        candidates.push(center);
        for fraction in [0.25, 0.5, 0.75] {
            candidates.push(interpolate(tetrahedron_centroid, center, fraction));
        }
    }
    for face in tetrahedron_faces(node_ids) {
        let face_points = face.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate| *candidate == node_id)
                .expect("tetrahedron face node should be in tetrahedron");
            points[index]
        });
        let face_centroid = [
            (face_points[0][0] + face_points[1][0] + face_points[2][0]) / 3.0,
            (face_points[0][1] + face_points[1][1] + face_points[2][1]) / 3.0,
            (face_points[0][2] + face_points[1][2] + face_points[2][2]) / 3.0,
        ];
        for fraction in [0.18, 0.33, 0.5, 0.67] {
            candidates.push(interpolate(face_centroid, tetrahedron_centroid, fraction));
        }
        for fraction in [0.12, 0.25, 0.4] {
            candidates.push(interpolate(face_centroid, cavity_centroid, fraction));
        }
    }
    for edge in tetrahedron_edges(node_ids) {
        let edge_points = edge.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate| *candidate == node_id)
                .expect("tetrahedron edge node should be in tetrahedron");
            points[index]
        });
        let midpoint = [
            (edge_points[0][0] + edge_points[1][0]) * 0.5,
            (edge_points[0][1] + edge_points[1][1]) * 0.5,
            (edge_points[0][2] + edge_points[1][2]) * 0.5,
        ];
        for fraction in [0.2, 0.4, 0.6] {
            candidates.push(interpolate(midpoint, tetrahedron_centroid, fraction));
        }
    }
    candidates
}

fn component_steiner_candidate_quality_score(
    node_ids: [u32; 4],
    points: [Point3; 4],
    candidate: Point3,
) -> f64 {
    let mut min_quality = f64::INFINITY;
    for face in tetrahedron_faces(node_ids) {
        let face_points = face.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate_id| *candidate_id == node_id)
                .expect("tetrahedron face node should be in tetrahedron");
            points[index]
        });
        let quality = tetrahedron_scaled_jacobian([
            face_points[0],
            face_points[1],
            face_points[2],
            candidate,
        ]);
        min_quality = min_quality.min(quality);
    }
    min_quality
}

fn tetrahedron_incenter(points: [Point3; 4]) -> Point3 {
    let weights = [
        triangle_area([points[1], points[2], points[3]]),
        triangle_area([points[0], points[3], points[2]]),
        triangle_area([points[0], points[1], points[3]]),
        triangle_area([points[0], points[2], points[1]]),
    ];
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= f64::EPSILON {
        return tetrahedron_centroid(points);
    }
    [
        (points[0][0] * weights[0]
            + points[1][0] * weights[1]
            + points[2][0] * weights[2]
            + points[3][0] * weights[3])
            / total,
        (points[0][1] * weights[0]
            + points[1][1] * weights[1]
            + points[2][1] * weights[2]
            + points[3][1] * weights[3])
            / total,
        (points[0][2] * weights[0]
            + points[1][2] * weights[1]
            + points[2][2] * weights[2]
            + points[3][2] * weights[3])
            / total,
    ]
}

fn interpolate(left: Point3, right: Point3, fraction: f64) -> Point3 {
    [
        left[0] * (1.0 - fraction) + right[0] * fraction,
        left[1] * (1.0 - fraction) + right[1] * fraction,
        left[2] * (1.0 - fraction) + right[2] * fraction,
    ]
}

pub fn validate_constrained_cavity(
    cavity: &ConstrainedCavity,
) -> Result<ConstrainedCavityValidationReport, ConstrainedCavityValidationError> {
    if cavity.removed_tetrahedron_ids.is_empty() {
        return Err(ConstrainedCavityValidationError::EmptyRemovedTetrahedronSet);
    }
    if !cavity.target_volume_m3.is_finite() || cavity.target_volume_m3 <= 0.0 {
        return Err(ConstrainedCavityValidationError::InvalidTargetVolume {
            target_volume_m3: cavity.target_volume_m3,
        });
    }
    if cavity.boundary_faces.len() < 4 {
        return Err(ConstrainedCavityValidationError::TooFewBoundaryFaces {
            boundary_face_count: cavity.boundary_faces.len(),
        });
    }

    let mut boundary_faces = BTreeSet::<[u32; 3]>::new();
    let mut boundary_edges = BTreeMap::<[u32; 2], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for (face_index, face) in cavity.boundary_faces.iter().enumerate() {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(ConstrainedCavityValidationError::DegenerateBoundaryFace {
                face_index,
                node_ids: face.node_ids,
            });
        }
        let sorted_face = sorted_face(face.node_ids);
        if !boundary_faces.insert(sorted_face) {
            return Err(ConstrainedCavityValidationError::DuplicateBoundaryFace {
                node_ids: sorted_face,
            });
        }
        for node_id in face.node_ids {
            boundary_nodes.insert(node_id);
        }
        for edge in face_edges(face.node_ids) {
            *boundary_edges.entry(sorted_edge(edge)).or_default() += 1;
        }
    }

    for (edge, face_count) in &boundary_edges {
        if *face_count != 2 {
            return Err(ConstrainedCavityValidationError::NonManifoldBoundaryEdge {
                node_ids: *edge,
                face_count: *face_count,
            });
        }
    }

    for node_id in &cavity.protected_node_ids {
        if !boundary_nodes.contains(node_id) {
            return Err(
                ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary {
                    node_id: *node_id,
                },
            );
        }
    }

    Ok(ConstrainedCavityValidationReport {
        boundary_face_count: cavity.boundary_faces.len(),
        boundary_edge_count: boundary_edges.len(),
        boundary_node_count: boundary_nodes.len(),
        protected_node_count: cavity.protected_node_ids.len(),
        target_volume_m3: cavity.target_volume_m3,
    })
}

pub fn validate_constrained_cavity_refill_volume(
    target_volume_m3: f64,
    candidate_volume_m3: f64,
    relative_tolerance: f64,
) -> Result<(), ConstrainedCavityValidationError> {
    if !target_volume_m3.is_finite() || target_volume_m3 <= 0.0 {
        return Err(ConstrainedCavityValidationError::InvalidTargetVolume { target_volume_m3 });
    }
    let tolerance_m3 = target_volume_m3.max(1.0e-18) * relative_tolerance.max(0.0);
    if !candidate_volume_m3.is_finite()
        || candidate_volume_m3 <= 0.0
        || (candidate_volume_m3 - target_volume_m3).abs() > tolerance_m3
    {
        return Err(ConstrainedCavityValidationError::InvalidRefillVolume {
            target_volume_m3,
            candidate_volume_m3,
            tolerance_m3,
        });
    }
    Ok(())
}

pub fn validate_constrained_cavity_boundary_preserved(
    cavity: &ConstrainedCavity,
    candidate_boundary_faces: &[ConstrainedCavityBoundaryFace],
) -> Result<(), ConstrainedCavityValidationError> {
    if cavity.boundary_faces.len() != candidate_boundary_faces.len() {
        return Err(
            ConstrainedCavityValidationError::BoundaryFaceCountMismatch {
                expected_count: cavity.boundary_faces.len(),
                candidate_count: candidate_boundary_faces.len(),
            },
        );
    }

    let expected_faces = boundary_face_map(&cavity.boundary_faces)?;
    let candidate_faces = boundary_face_map(candidate_boundary_faces)?;

    for expected_face in expected_faces.keys() {
        if !candidate_faces.contains_key(expected_face) {
            return Err(ConstrainedCavityValidationError::MissingBoundaryFace {
                node_ids: *expected_face,
            });
        }
    }
    for candidate_face in candidate_faces.keys() {
        if !expected_faces.contains_key(candidate_face) {
            return Err(ConstrainedCavityValidationError::UnexpectedBoundaryFace {
                node_ids: *candidate_face,
            });
        }
    }

    for (face_key, expected) in &expected_faces {
        let candidate = candidate_faces
            .get(face_key)
            .expect("candidate face should exist after key comparison");
        let expected_outside_tetrahedron_ids = sorted_u32_ids(&expected.outside_tetrahedron_ids);
        let candidate_outside_tetrahedron_ids = sorted_u32_ids(&candidate.outside_tetrahedron_ids);
        if expected_outside_tetrahedron_ids != candidate_outside_tetrahedron_ids {
            return Err(
                ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch {
                    node_ids: *face_key,
                    expected_outside_tetrahedron_ids,
                    candidate_outside_tetrahedron_ids,
                },
            );
        }
        if expected.source_face_id != candidate.source_face_id {
            return Err(
                ConstrainedCavityValidationError::BoundarySourceFaceMismatch {
                    node_ids: *face_key,
                    expected_source_face_id: expected.source_face_id,
                    candidate_source_face_id: candidate.source_face_id,
                },
            );
        }
        let expected_edges = boundary_face_source_edges(expected)?;
        let candidate_edges = boundary_face_source_edges(candidate)?;
        for (edge_key, expected_source_edge_id) in expected_edges {
            let candidate_source_edge_id = candidate_edges.get(&edge_key).copied().flatten();
            if expected_source_edge_id != candidate_source_edge_id {
                return Err(
                    ConstrainedCavityValidationError::BoundarySourceEdgeMismatch {
                        node_ids: edge_key,
                        expected_source_edge_id,
                        candidate_source_edge_id,
                    },
                );
            }
        }
        let expected_regions = sorted_region_ids(&expected.region_ids);
        let candidate_regions = sorted_region_ids(&candidate.region_ids);
        if expected_regions != candidate_regions {
            return Err(ConstrainedCavityValidationError::BoundaryRegionMismatch {
                node_ids: *face_key,
                expected_region_ids: expected_regions,
                candidate_region_ids: candidate_regions,
            });
        }
    }

    Ok(())
}

pub fn split_constrained_cavity_boundary_face(
    face: &ConstrainedCavityBoundaryFace,
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 3], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    let [a, b, c] = face.node_ids;
    Ok([
        split_child_boundary_face(face, [a, b, split_node_id], &perimeter_source_edges),
        split_child_boundary_face(face, [b, c, split_node_id], &perimeter_source_edges),
        split_child_boundary_face(face, [c, a, split_node_id], &perimeter_source_edges),
    ])
}

pub fn split_constrained_cavity_boundary_faces(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 2);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
            split_faces.extend(split_constrained_cavity_boundary_face(face, split_node_id)?);
        } else {
            split_faces.push(face.clone());
        }
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

pub fn split_constrained_cavity_boundary_edge(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    let target_edge = sorted_edge(edge);
    let boundary_node_map = boundary_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node_id in target_edge {
        if !boundary_node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryNode { node_id });
        }
    }
    let Some(target_face) = cavity.boundary_faces.iter().find(|face| {
        face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == target_edge)
    }) else {
        return Err(
            ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryEdge {
                node_ids: target_edge,
            },
        );
    };
    let split_node = boundary_edge_split_node(target_edge, &boundary_node_map, 0.5);
    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        target_face.node_ids,
        target_edge,
        split_node.node_id,
    )
    .map_err(ConstrainedCavityBoundaryEdgeSplitError::Split)?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)
        .map_err(ConstrainedCavityBoundaryEdgeSplitError::Validation)?;
    Ok((split_cavity, split_node))
}

pub fn split_constrained_cavity_boundary_edge_patch_at_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    split_constrained_cavity_boundary_edge_patch_with_weights_impl(
        cavity,
        boundary_nodes,
        edge,
        [0.25, 0.25, 0.25, 0.25],
    )
}

#[cfg(test)]
pub(crate) fn split_constrained_cavity_boundary_edge_patch_with_weights(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
    weights: [f64; 4],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    split_constrained_cavity_boundary_edge_patch_with_weights_impl(
        cavity,
        boundary_nodes,
        edge,
        weights,
    )
}

fn split_constrained_cavity_boundary_edge_patch_with_weights_impl(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
    weights: [f64; 4],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    let weight_sum = weights.iter().sum::<f64>();
    let invalid_weights = weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
        || (weight_sum - 1.0).abs() > 1.0e-12;
    #[cfg(test)]
    {
        if invalid_weights {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::InvalidPatchWeights { weights });
        }
    }
    #[cfg(not(test))]
    debug_assert!(!invalid_weights);
    let target_edge = sorted_edge(edge);
    let boundary_node_map = boundary_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node_id in target_edge {
        if !boundary_node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryNode { node_id });
        }
    }
    let incident_faces = cavity
        .boundary_faces
        .iter()
        .filter(|face| {
            face_edges(face.node_ids)
                .into_iter()
                .any(|candidate| sorted_edge(candidate) == target_edge)
        })
        .collect::<Vec<_>>();
    if incident_faces.len() != 2 {
        return Err(
            ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryEdge {
                node_ids: target_edge,
            },
        );
    }
    let mut opposite_nodes = Vec::<u32>::new();
    for face in &incident_faces {
        let Some(opposite) = face
            .node_ids
            .into_iter()
            .find(|node_id| !target_edge.contains(node_id))
        else {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::Split(
                ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
                    node_ids: sorted_face(face.node_ids),
                },
            ));
        };
        if !boundary_node_map.contains_key(&opposite) {
            return Err(
                ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryNode { node_id: opposite },
            );
        }
        opposite_nodes.push(opposite);
    }
    let split_node = boundary_edge_patch_split_node(
        target_edge,
        [opposite_nodes[0], opposite_nodes[1]],
        &boundary_node_map,
        weights,
    );
    let split_faces = split_constrained_cavity_boundary_faces_on_edge_patch(
        &cavity.boundary_faces,
        target_edge,
        split_node.node_id,
    )
    .map_err(ConstrainedCavityBoundaryEdgeSplitError::Split)?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)
        .map_err(ConstrainedCavityBoundaryEdgeSplitError::Validation)?;
    Ok((split_cavity, split_node))
}

pub fn split_constrained_cavity_boundary_face_at_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    face: [u32; 3],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryFaceSplitError> {
    split_constrained_cavity_boundary_face_at_barycentric(
        cavity,
        boundary_nodes,
        face,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    )
}

pub fn split_constrained_cavity_boundary_face_at_barycentric(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    face: [u32; 3],
    barycentric: [f64; 3],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryFaceSplitError> {
    let barycentric_sum = barycentric.iter().sum::<f64>();
    if barycentric
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || (barycentric_sum - 1.0).abs() > 1.0e-12
    {
        return Err(
            ConstrainedCavityBoundaryFaceSplitError::InvalidBarycentricCoordinates { barycentric },
        );
    }
    let target_face = sorted_face(face);
    let boundary_node_map = boundary_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node_id in target_face {
        if !boundary_node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityBoundaryFaceSplitError::MissingBoundaryNode { node_id });
        }
    }
    if !cavity
        .boundary_faces
        .iter()
        .any(|boundary_face| sorted_face(boundary_face.node_ids) == target_face)
    {
        return Err(
            ConstrainedCavityBoundaryFaceSplitError::MissingBoundaryFace {
                node_ids: target_face,
            },
        );
    }
    let split_node = boundary_face_split_node(target_face, &boundary_node_map, barycentric);
    let split_faces = split_constrained_cavity_boundary_faces(
        &cavity.boundary_faces,
        target_face,
        split_node.node_id,
    )
    .map_err(ConstrainedCavityBoundaryFaceSplitError::Split)?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)
        .map_err(ConstrainedCavityBoundaryFaceSplitError::Validation)?;
    Ok((split_cavity, split_node))
}

pub fn split_constrained_cavity_boundary_faces_at_centroids(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    faces: &[[u32; 3]],
) -> Result<(ConstrainedCavity, Vec<ConstrainedCavityNode>), ConstrainedCavityBoundaryFaceSplitError>
{
    let mut seen_faces = BTreeSet::<[u32; 3]>::new();
    let mut split_cavity = cavity.clone();
    let mut split_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut current_nodes = boundary_nodes.to_vec();
    for face in faces {
        let target_face = sorted_face(*face);
        if !seen_faces.insert(target_face) {
            return Err(
                ConstrainedCavityBoundaryFaceSplitError::DuplicateBoundaryFace {
                    node_ids: target_face,
                },
            );
        }
        let (next_cavity, split_node) = split_constrained_cavity_boundary_face_at_centroid(
            &split_cavity,
            &current_nodes,
            target_face,
        )?;
        current_nodes.push(split_node.clone());
        split_nodes.push(split_node);
        split_cavity = next_cavity;
    }
    Ok((split_cavity, split_nodes))
}

pub fn split_constrained_cavity_boundary_patch_at_centroids(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge_patches: &[[u32; 2]],
    faces: &[[u32; 3]],
) -> Result<ConstrainedCavityBoundaryPatchSplit, ConstrainedCavityBoundaryPatchSplitError> {
    let mut split_cavity = cavity.clone();
    let mut current_nodes = boundary_nodes.to_vec();
    let mut split_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut steps = Vec::<ConstrainedCavityBoundaryPatchSplitStep>::new();

    for edge in edge_patches {
        let target_edge = sorted_edge(*edge);
        let (next_cavity, split_node) = split_constrained_cavity_boundary_edge_patch_at_centroid(
            &split_cavity,
            &current_nodes,
            target_edge,
        )
        .map_err(ConstrainedCavityBoundaryPatchSplitError::Edge)?;
        steps.push(ConstrainedCavityBoundaryPatchSplitStep::EdgePatch {
            node_ids: target_edge,
            split_node_id: split_node.node_id,
        });
        current_nodes.push(split_node.clone());
        split_nodes.push(split_node);
        split_cavity = next_cavity;
    }

    let mut seen_faces = BTreeSet::<[u32; 3]>::new();
    for face in faces {
        let target_face = sorted_face(*face);
        if !seen_faces.insert(target_face) {
            return Err(ConstrainedCavityBoundaryPatchSplitError::Face(
                ConstrainedCavityBoundaryFaceSplitError::DuplicateBoundaryFace {
                    node_ids: target_face,
                },
            ));
        }
        let (next_cavity, split_node) = split_constrained_cavity_boundary_face_at_centroid(
            &split_cavity,
            &current_nodes,
            target_face,
        )
        .map_err(ConstrainedCavityBoundaryPatchSplitError::Face)?;
        steps.push(ConstrainedCavityBoundaryPatchSplitStep::Face {
            node_ids: target_face,
            split_node_id: split_node.node_id,
        });
        current_nodes.push(split_node.clone());
        split_nodes.push(split_node);
        split_cavity = next_cavity;
    }

    Ok(ConstrainedCavityBoundaryPatchSplit {
        cavity: split_cavity,
        split_nodes,
        steps,
    })
}

pub fn split_constrained_cavity_source_edge(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    source_nodes: &[ConstrainedCavityNode],
    source_tetrahedra: &[CavityTetrahedron],
    edge: [u32; 2],
) -> Result<ConstrainedCavitySourceEdgeSplit, ConstrainedCavitySourceEdgeSplitError> {
    let target_edge = sorted_edge(edge);
    let boundary_node_map = boundary_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node_id in target_edge {
        if !boundary_node_map.contains_key(&node_id) {
            return Err(ConstrainedCavitySourceEdgeSplitError::MissingBoundaryNode { node_id });
        }
    }
    if !cavity.boundary_faces.iter().any(|face| {
        face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == target_edge)
    }) {
        return Err(ConstrainedCavitySourceEdgeSplitError::MissingBoundaryEdge {
            node_ids: target_edge,
        });
    }

    let source_node_map = source_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in source_tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !source_node_map.contains_key(&node_id) {
                return Err(ConstrainedCavitySourceEdgeSplitError::MissingSourceNode { node_id });
            }
        }
    }
    for tetrahedron_id in &cavity.removed_tetrahedron_ids {
        if !source_tetrahedra
            .iter()
            .any(|tetrahedron| tetrahedron.tetrahedron_id == *tetrahedron_id)
        {
            return Err(
                ConstrainedCavitySourceEdgeSplitError::MissingRemovedSourceTetrahedron {
                    tetrahedron_id: *tetrahedron_id,
                },
            );
        }
    }

    let mut split_node_id = source_node_map
        .keys()
        .chain(boundary_node_map.keys())
        .copied()
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    while source_node_map.contains_key(&split_node_id)
        || boundary_node_map.contains_key(&split_node_id)
    {
        split_node_id = split_node_id.saturating_add(1);
    }
    let points = target_edge.map(|node_id| boundary_node_map[&node_id]);
    let split_node = ConstrainedCavityNode {
        node_id: split_node_id,
        coordinates_m: [
            0.5 * (points[0][0] + points[1][0]),
            0.5 * (points[0][1] + points[1][1]),
            0.5 * (points[0][2] + points[1][2]),
        ],
    };
    let mut node_map_with_split = source_node_map;
    node_map_with_split.insert(split_node.node_id, split_node.coordinates_m);

    let selected_original_tetrahedron_ids = cavity
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut selected_tetrahedron_ids = BTreeSet::<u32>::new();
    let mut split_source_tetrahedra =
        Vec::<CavityTetrahedron>::with_capacity(source_tetrahedra.len() + 8);
    let mut next_tetrahedron_id = source_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.tetrahedron_id)
        .max()
        .unwrap_or_default()
        .saturating_add(1);
    let mut incident_count = 0_usize;

    for tetrahedron in source_tetrahedra {
        let incident = target_edge
            .iter()
            .all(|node_id| tetrahedron.node_ids.contains(node_id));
        if !incident {
            if selected_original_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id) {
                selected_tetrahedron_ids.insert(tetrahedron.tetrahedron_id);
            }
            split_source_tetrahedra.push(tetrahedron.clone());
            continue;
        }

        incident_count += 1;
        let opposite_nodes = tetrahedron
            .node_ids
            .into_iter()
            .filter(|node_id| !target_edge.contains(node_id))
            .collect::<Vec<_>>();
        if opposite_nodes.len() != 2 {
            return Err(
                ConstrainedCavitySourceEdgeSplitError::DegenerateSplitTetrahedron {
                    tetrahedron_id: tetrahedron.tetrahedron_id,
                },
            );
        }
        let child_node_ids = [
            [
                target_edge[0],
                split_node.node_id,
                opposite_nodes[0],
                opposite_nodes[1],
            ],
            [
                split_node.node_id,
                target_edge[1],
                opposite_nodes[0],
                opposite_nodes[1],
            ],
        ];
        for child in child_node_ids {
            let points = child.map(|node_id| node_map_with_split[&node_id]);
            let (oriented_node_ids, volume_m3) = orient_tetrahedron_node_ids(child, points);
            if volume_m3 <= 0.0 {
                return Err(
                    ConstrainedCavitySourceEdgeSplitError::DegenerateSplitTetrahedron {
                        tetrahedron_id: tetrahedron.tetrahedron_id,
                    },
                );
            }
            let oriented_points = oriented_node_ids.map(|node_id| node_map_with_split[&node_id]);
            let child_tetrahedron = CavityTetrahedron {
                tetrahedron_id: next_tetrahedron_id,
                component_id: tetrahedron.component_id,
                node_ids: oriented_node_ids,
                source_surface_element_id: tetrahedron.source_surface_element_id,
                region_ids: tetrahedron.region_ids.clone(),
                volume_m3,
                aspect_ratio: tetrahedron_edge_aspect_ratio(oriented_points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(oriented_points).abs(),
            };
            if selected_original_tetrahedron_ids.contains(&tetrahedron.tetrahedron_id) {
                selected_tetrahedron_ids.insert(child_tetrahedron.tetrahedron_id);
            }
            split_source_tetrahedra.push(child_tetrahedron);
            next_tetrahedron_id = next_tetrahedron_id.saturating_add(1);
        }
    }

    if incident_count == 0 {
        return Err(
            ConstrainedCavitySourceEdgeSplitError::NoIncidentSourceTetrahedron {
                node_ids: target_edge,
            },
        );
    }

    let selected_indices = split_source_tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            selected_tetrahedron_ids
                .contains(&tetrahedron.tetrahedron_id)
                .then_some(index)
        })
        .collect::<BTreeSet<_>>();
    let split_cavity = build_constrained_cavity_from_index_set(
        &split_source_tetrahedra,
        &selected_indices,
        cavity.protected_node_ids.clone(),
    );
    validate_constrained_cavity(&split_cavity)
        .map_err(ConstrainedCavitySourceEdgeSplitError::Validation)?;

    Ok(ConstrainedCavitySourceEdgeSplit {
        cavity: split_cavity,
        split_node,
        source_tetrahedra: split_source_tetrahedra,
    })
}

fn split_constrained_cavity_boundary_face_on_edge(
    face: &ConstrainedCavityBoundaryFace,
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 2], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let sorted_split_edge = sorted_edge(edge);
    if !face_edges(face.node_ids)
        .into_iter()
        .any(|candidate| sorted_edge(candidate) == sorted_split_edge)
    {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        });
    }
    let source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    let source_edge_id = source_edges.get(&sorted_split_edge).copied().flatten();
    let [a, b] = edge;
    let c = face
        .node_ids
        .into_iter()
        .find(|node_id| *node_id != a && *node_id != b)
        .ok_or(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        })?;
    Ok([
        edge_split_child_boundary_face(
            face,
            [a, split_node_id, c],
            split_node_id,
            sorted_split_edge,
            source_edge_id,
            &source_edges,
        ),
        edge_split_child_boundary_face(
            face,
            [split_node_id, b, c],
            split_node_id,
            sorted_split_edge,
            source_edge_id,
            &source_edges,
        ),
    ])
}

fn split_constrained_cavity_boundary_faces_on_edge(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let split_edge = sorted_edge(edge);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 1);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
        }
        if face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == split_edge)
        {
            split_faces.extend(split_constrained_cavity_boundary_face_on_edge(
                face,
                edge,
                split_node_id,
            )?);
        } else {
            split_faces.push(face.clone());
        }
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

fn split_constrained_cavity_boundary_faces_on_edge_patch(
    faces: &[ConstrainedCavityBoundaryFace],
    edge: [u32; 2],
    split_node_id: u32,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let split_edge = sorted_edge(edge);
    let incident = faces
        .iter()
        .filter(|face| {
            face_edges(face.node_ids)
                .into_iter()
                .any(|candidate| sorted_edge(candidate) == split_edge)
        })
        .collect::<Vec<_>>();
    if incident.len() != 2 {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: [split_edge[0], split_edge[1], split_node_id],
        });
    }
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 2);
    for face in faces {
        if face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == split_edge)
        {
            split_faces.extend(split_constrained_cavity_boundary_face_on_edge_patch(
                face,
                split_edge,
                split_node_id,
            )?);
        } else {
            split_faces.push(face.clone());
        }
    }
    Ok(split_faces)
}

fn split_constrained_cavity_boundary_face_on_edge_patch(
    face: &ConstrainedCavityBoundaryFace,
    split_edge: [u32; 2],
    split_node_id: u32,
) -> Result<[ConstrainedCavityBoundaryFace; 2], ConstrainedCavityBoundarySplitError> {
    if face.node_ids.contains(&split_node_id) {
        return Err(
            ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
                node_id: split_node_id,
            },
        );
    }
    let opposite = face
        .node_ids
        .into_iter()
        .find(|node_id| !split_edge.contains(node_id))
        .ok_or(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        })?;
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    Ok([
        split_child_boundary_face(
            face,
            [split_edge[0], opposite, split_node_id],
            &perimeter_source_edges,
        ),
        split_child_boundary_face(
            face,
            [opposite, split_edge[1], split_node_id],
            &perimeter_source_edges,
        ),
    ])
}

fn split_constrained_cavity_boundary_faces_on_three_edges(
    faces: &[ConstrainedCavityBoundaryFace],
    face_node_ids: [u32; 3],
    edge_split_node_ids: BTreeMap<[u32; 2], u32>,
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityBoundarySplitError> {
    let target = sorted_face(face_node_ids);
    let mut split_faces = Vec::<ConstrainedCavityBoundaryFace>::with_capacity(faces.len() + 6);
    let mut found = false;
    for face in faces {
        if sorted_face(face.node_ids) == target {
            found = true;
            split_faces.extend(split_constrained_cavity_boundary_face_on_three_edges(
                face,
                &edge_split_node_ids,
            )?);
            continue;
        }
        let split_edges = face_edges(face.node_ids)
            .into_iter()
            .filter_map(|edge| {
                edge_split_node_ids
                    .get(&sorted_edge(edge))
                    .copied()
                    .map(|node_id| (edge, node_id))
            })
            .collect::<Vec<_>>();
        if split_edges.is_empty() {
            split_faces.push(face.clone());
            continue;
        }
        if split_edges.len() > 1 {
            return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
                node_ids: sorted_face(face.node_ids),
            });
        }
        let (edge, split_node_id) = split_edges[0];
        split_faces.extend(split_constrained_cavity_boundary_face_on_edge(
            face,
            edge,
            split_node_id,
        )?);
    }
    if !found {
        return Err(ConstrainedCavityBoundarySplitError::MissingBoundaryFace { node_ids: target });
    }
    Ok(split_faces)
}

fn split_constrained_cavity_boundary_face_on_three_edges(
    face: &ConstrainedCavityBoundaryFace,
    edge_split_node_ids: &BTreeMap<[u32; 2], u32>,
) -> Result<[ConstrainedCavityBoundaryFace; 4], ConstrainedCavityBoundarySplitError> {
    let [a, b, c] = face.node_ids;
    let ab = *edge_split_node_ids.get(&sorted_edge([a, b])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let bc = *edge_split_node_ids.get(&sorted_edge([b, c])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let ca = *edge_split_node_ids.get(&sorted_edge([c, a])).ok_or(
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        },
    )?;
    let perimeter_source_edges = boundary_face_source_edges(face).map_err(|_| {
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: sorted_face(face.node_ids),
        }
    })?;
    Ok([
        three_edge_split_child_boundary_face(
            face,
            [a, ab, ca],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ab, b, bc],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ca, bc, c],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
        three_edge_split_child_boundary_face(
            face,
            [ab, bc, ca],
            &perimeter_source_edges,
            edge_split_node_ids,
        ),
    ])
}

fn split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: parent.outside_tetrahedron_ids.clone(),
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            perimeter_source_edges
                .get(&sorted_edge(edge))
                .copied()
                .flatten()
        }),
        region_ids: parent.region_ids.clone(),
    }
}

fn three_edge_split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
    edge_split_node_ids: &BTreeMap<[u32; 2], u32>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: parent.outside_tetrahedron_ids.clone(),
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            let original_edge =
                edge_split_node_ids
                    .iter()
                    .find_map(|(split_edge, split_node_id)| {
                        if edge.contains(split_node_id)
                            && edge.into_iter().any(|node_id| {
                                node_id != *split_node_id && split_edge.contains(&node_id)
                            })
                        {
                            Some(*split_edge)
                        } else {
                            None
                        }
                    })?;
            perimeter_source_edges
                .get(&original_edge)
                .copied()
                .flatten()
        }),
        region_ids: parent.region_ids.clone(),
    }
}

fn edge_split_child_boundary_face(
    parent: &ConstrainedCavityBoundaryFace,
    node_ids: [u32; 3],
    split_node_id: u32,
    split_edge: [u32; 2],
    split_edge_source_id: Option<u32>,
    perimeter_source_edges: &BTreeMap<[u32; 2], Option<u32>>,
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: parent.outside_tetrahedron_ids.clone(),
        source_face_id: parent.source_face_id,
        source_edge_ids: face_edges(node_ids).map(|edge| {
            let sorted = sorted_edge(edge);
            if edge.contains(&split_node_id)
                && edge
                    .into_iter()
                    .any(|node_id| node_id != split_node_id && split_edge.contains(&node_id))
            {
                split_edge_source_id
            } else {
                perimeter_source_edges.get(&sorted).copied().flatten()
            }
        }),
        region_ids: parent.region_ids.clone(),
    }
}

fn validate_refill_options(
    options: ConstrainedCavityRefillOptions,
) -> Result<(), ConstrainedCavityRefillError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 <= 0.0
        || !options.max_aspect_ratio.is_finite()
        || options.max_aspect_ratio <= 0.0
        || !options.min_scaled_jacobian.is_finite()
        || options.min_scaled_jacobian < 0.0
        || !options.volume_relative_tolerance.is_finite()
        || options.volume_relative_tolerance < 0.0
        || !options.min_protected_node_distance_m.is_finite()
        || options.min_protected_node_distance_m < 0.0
    {
        return Err(ConstrainedCavityRefillError::InvalidOptions);
    }
    Ok(())
}

fn boundary_node_coordinates(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
) -> Result<BTreeMap<u32, Point3>, ConstrainedCavityRefillError> {
    let coordinates = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for face in &cavity.boundary_faces {
        for node_id in face.node_ids {
            if !coordinates.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillError::MissingBoundaryNode { node_id });
            }
        }
    }
    Ok(coordinates)
}

fn cavity_boundary_node_ids(cavity: &ConstrainedCavity) -> BTreeSet<u32> {
    cavity
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids)
        .collect()
}

fn candidate_respects_protected_boundary_distance(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    point: Point3,
    options: ConstrainedCavityRefillOptions,
) -> bool {
    if options.min_protected_node_distance_m <= 0.0 || cavity.protected_node_ids.is_empty() {
        return true;
    }
    let min_distance_squared = options.min_protected_node_distance_m.powi(2);
    cavity.protected_node_ids.iter().all(|node_id| {
        boundary_nodes.get(node_id).is_none_or(|protected_point| {
            distance_squared(point, *protected_point) > min_distance_squared
        })
    })
}

fn cavity_boundary_triangles(
    cavity: &ConstrainedCavity,
    nodes: &BTreeMap<u32, Point3>,
) -> Result<Vec<Triangle3>, ConstrainedCavityRefillError> {
    cavity
        .boundary_faces
        .iter()
        .map(|face| {
            Ok([
                *nodes.get(&face.node_ids[0]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[0],
                    },
                )?,
                *nodes.get(&face.node_ids[1]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[1],
                    },
                )?,
                *nodes.get(&face.node_ids[2]).ok_or(
                    ConstrainedCavityRefillError::MissingBoundaryNode {
                        node_id: face.node_ids[2],
                    },
                )?,
            ])
        })
        .collect()
}

fn single_tetrahedron_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefill>, ConstrainedCavityValidationError> {
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    if node_ids.len() != 4 {
        return Ok(None);
    }
    let points = [
        boundary_nodes[&node_ids[0]],
        boundary_nodes[&node_ids[1]],
        boundary_nodes[&node_ids[2]],
        boundary_nodes[&node_ids[3]],
    ];
    let Some(tetrahedron) = raw_refill_tetrahedron(
        [node_ids[0], node_ids[1], node_ids[2], node_ids[3]],
        points,
        options,
    ) else {
        return Ok(None);
    };
    let refill =
        refill_from_tetrahedra(cavity, vec![tetrahedron], options.volume_relative_tolerance)?;
    Ok(Some(refill))
}

fn boundary_node_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let boundary_triangles = cavity
        .boundary_faces
        .iter()
        .map(|face| {
            [
                boundary_nodes[&face.node_ids[0]],
                boundary_nodes[&face.node_ids[1]],
                boundary_nodes[&face.node_ids[2]],
            ]
        })
        .collect::<Vec<_>>();
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_nodes[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut first_rejection = None::<&'static str>;
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options) {
            Ok(tetrahedron) => refill_tetrahedra.push(tetrahedron),
            Err(reason) => {
                if first_rejection.is_none() {
                    first_rejection = Some(boundary_node_refill_rejection_reason(reason));
                }
            }
        }
    }
    if refill_tetrahedra.is_empty() {
        if let Some(refill) = boundary_node_exact_cover_refill_candidate(
            cavity,
            boundary_nodes,
            &boundary_triangles,
            options,
        )? {
            return Ok(Ok(improve_refill_with_local_flips(
                cavity,
                &boundary_nodes,
                &refill,
                options,
            )
            .unwrap_or(refill)));
        }
        return Ok(Err(
            first_rejection.unwrap_or("boundary_node_delaunay_empty")
        ));
    }
    match refill_from_tetrahedra(
        cavity,
        refill_tetrahedra.clone(),
        options.volume_relative_tolerance,
    ) {
        Ok(refill) => Ok(Ok(improve_refill_with_local_flips(
            cavity,
            &boundary_nodes,
            &refill,
            options,
        )
        .unwrap_or(refill))),
        Err(_) => {
            if let Some(refill) = boundary_node_exact_cover_refill_candidate(
                cavity,
                boundary_nodes,
                &boundary_triangles,
                options,
            )? {
                return Ok(Ok(improve_refill_with_local_flips(
                    cavity,
                    &boundary_nodes,
                    &refill,
                    options,
                )
                .unwrap_or(refill)));
            }
            let (completed_cavity, completed_tetrahedra, inserted_nodes) =
                match complete_missing_boundary_face_tetrahedra(
                    cavity,
                    boundary_nodes,
                    refill_tetrahedra,
                    &boundary_triangles,
                    options,
                )? {
                    Ok(completed_tetrahedra) => completed_tetrahedra,
                    Err(reason) => return Ok(Err(reason)),
                };
            let mut refill = match refill_from_tetrahedra(
                &completed_cavity,
                completed_tetrahedra,
                options.volume_relative_tolerance,
            ) {
                Ok(refill) => refill,
                Err(err) => return Ok(Err(boundary_node_refill_validation_reason(&err))),
            };
            refill.inserted_nodes = inserted_nodes;
            refill = improve_refill_with_local_flips(
                &completed_cavity,
                &boundary_nodes,
                &refill,
                options,
            )
            .unwrap_or(refill);
            Ok(Ok(refill))
        }
    }
}

fn boundary_node_refill_rejection_reason(reason: &'static str) -> &'static str {
    match reason {
        "star_tetrahedron_min_volume" => "boundary_node_tetrahedron_min_volume",
        "star_tetrahedron_aspect_ratio" => "boundary_node_tetrahedron_aspect_ratio",
        "star_tetrahedron_scaled_jacobian" => "boundary_node_tetrahedron_scaled_jacobian",
        other => other,
    }
}

fn boundary_node_refill_validation_reason(
    error: &ConstrainedCavityValidationError,
) -> &'static str {
    match refill_validation_reason(error) {
        "boundary_face_count_mismatch" => "boundary_node_boundary_face_count_mismatch",
        "missing_boundary_face" => "boundary_node_missing_boundary_face",
        "unexpected_boundary_face" => "boundary_node_unexpected_boundary_face",
        "volume_mismatch" => "boundary_node_volume_mismatch",
        "boundary_source_face_mismatch" => "boundary_node_boundary_source_face_mismatch",
        "boundary_source_edge_mismatch" => "boundary_node_boundary_source_edge_mismatch",
        "boundary_region_mismatch" => "boundary_node_boundary_region_mismatch",
        "invalid_cavity" => "boundary_node_invalid_cavity",
        other => other,
    }
}

fn centroid_interior_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let Some(coordinates_m) = cavity_boundary_node_centroid(cavity, boundary_nodes) else {
        return Ok(Err("centroid_interior_refill_empty_boundary"));
    };
    if point_in_closed_triangle_surface(
        coordinates_m,
        boundary_triangles,
        MeshingTolerance::default(),
    ) != PointInClosedSurface::Inside
    {
        return Ok(Err("centroid_interior_refill_outside_cavity"));
    }
    let node = ConstrainedCavityNode {
        node_id: next_cavity_node_id(cavity),
        coordinates_m,
    };
    match star_refill_candidate_with_rejection_reason(cavity, boundary_nodes, node.clone(), options)
    {
        Ok(Ok(mut refill)) => {
            refill.inserted_nodes.push(node);
            Ok(Ok(refill))
        }
        Ok(Err(reason)) => Ok(Err(centroid_interior_refill_rejection_reason(reason))),
        Err(err) => Err(err),
    }
}

fn cavity_boundary_node_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> Option<Point3> {
    let node_ids = cavity_boundary_node_ids(cavity);
    if node_ids.is_empty() {
        return None;
    }
    let mut centroid = [0.0_f64; 3];
    for node_id in &node_ids {
        let point = boundary_nodes.get(node_id)?;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let scale = 1.0 / node_ids.len() as f64;
    Some([
        centroid[0] * scale,
        centroid[1] * scale,
        centroid[2] * scale,
    ])
}

fn next_cavity_node_id(cavity: &ConstrainedCavity) -> u32 {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn centroid_interior_refill_rejection_reason(reason: &'static str) -> &'static str {
    match reason {
        "star_tetrahedron_min_volume" => "centroid_interior_refill_tetrahedron_min_volume",
        "star_tetrahedron_aspect_ratio" => "centroid_interior_refill_tetrahedron_aspect_ratio",
        "star_tetrahedron_scaled_jacobian" => {
            "centroid_interior_refill_tetrahedron_scaled_jacobian"
        }
        other => other,
    }
}

fn two_interior_node_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let mut best = None::<ConstrainedCavityRefill>;
    let mut first_rejection = None::<&'static str>;
    for left in 0..interior_candidates.len() {
        for right in (left + 1)..interior_candidates.len() {
            let pair = [
                interior_candidates[left].clone(),
                interior_candidates[right].clone(),
            ];
            let mut points = boundary_node_ids
                .iter()
                .map(|node_id| ConnectivityPoint {
                    node_id: *node_id,
                    coordinates_m: boundary_nodes[node_id],
                    is_super: false,
                })
                .collect::<Vec<_>>();
            points.extend(pair.iter().map(|node| ConnectivityPoint {
                node_id: node.node_id,
                coordinates_m: node.coordinates_m,
                is_super: false,
            }));
            let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
            for tetrahedron in tetrahedralize_points(&points) {
                let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
                let tetrahedron_points = tetrahedron
                    .vertices
                    .map(|index| points[index].coordinates_m);
                if point_in_closed_triangle_surface(
                    tetrahedron_centroid(tetrahedron_points),
                    boundary_triangles,
                    MeshingTolerance::default(),
                ) != PointInClosedSurface::Inside
                {
                    continue;
                }
                match raw_refill_tetrahedron_with_rejection_reason(
                    node_ids,
                    tetrahedron_points,
                    options,
                ) {
                    Ok(tetrahedron) => refill_tetrahedra.push(tetrahedron),
                    Err(reason) => {
                        if first_rejection.is_none() {
                            first_rejection = Some(boundary_node_refill_rejection_reason(reason));
                        }
                    }
                }
            }
            if refill_tetrahedra.is_empty() {
                if first_rejection.is_none() {
                    first_rejection = Some("two_interior_delaunay_empty");
                }
                continue;
            }
            match refill_from_tetrahedra(
                cavity,
                refill_tetrahedra.clone(),
                options.volume_relative_tolerance,
            ) {
                Ok(mut refill) => {
                    refill.inserted_nodes = pair.to_vec();
                    if best
                        .as_ref()
                        .is_none_or(|current| refill_is_better(&refill, current))
                    {
                        best = Some(refill);
                    }
                }
                Err(err) => {
                    if let Some(mut refill) = exact_cover_refill_from_candidate_tetrahedra(
                        cavity,
                        &refill_tetrahedra,
                        options,
                    )? {
                        refill.inserted_nodes = pair.to_vec();
                        if best
                            .as_ref()
                            .is_none_or(|current| refill_is_better(&refill, current))
                        {
                            best = Some(refill);
                        }
                        continue;
                    }
                    if first_rejection.is_none() {
                        first_rejection = Some(boundary_node_refill_validation_reason(&err));
                    }
                }
            }
        }
    }
    Ok(best
        .map(Ok)
        .unwrap_or_else(|| Err(first_rejection.unwrap_or("two_interior_no_candidate"))))
}

fn multi_interior_node_refill_candidate(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, boundary_nodes) else {
        return Ok(Err("multi_interior_empty_boundary"));
    };
    let selected_interior_nodes =
        selected_multi_interior_nodes(interior_candidates, cavity_centroid);
    if selected_interior_nodes.len() < 3 {
        return Ok(Err("multi_interior_too_few_candidates"));
    }
    let mut points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_nodes[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    points.extend(
        selected_interior_nodes
            .iter()
            .map(|node| ConnectivityPoint {
                node_id: node.node_id,
                coordinates_m: node.coordinates_m,
                is_super: false,
            }),
    );

    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut first_rejection = None::<&'static str>;
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(node_ids)) {
            continue;
        }
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options) {
            Ok(tetrahedron) => refill_tetrahedra.push(tetrahedron),
            Err(reason) => {
                if first_rejection.is_none() {
                    first_rejection = Some(boundary_node_refill_rejection_reason(reason));
                }
            }
        }
        if refill_tetrahedra.len() > MAX_MULTI_INTERIOR_REFILL_CANDIDATES {
            return Ok(Err("multi_interior_over_candidate_limit"));
        }
    }
    if refill_tetrahedra.is_empty() {
        return Ok(Err(
            first_rejection.unwrap_or("multi_interior_delaunay_empty")
        ));
    }
    let Some(mut refill) =
        exact_cover_refill_from_candidate_tetrahedra(cavity, &refill_tetrahedra, options)?
    else {
        return Ok(Err(multi_interior_exact_cover_failure_reason(
            cavity,
            &refill_tetrahedra,
            options,
        )));
    };
    let used_node_ids = refill
        .tetrahedra
        .iter()
        .flat_map(|tetrahedron| tetrahedron.node_ids)
        .collect::<BTreeSet<_>>();
    refill.inserted_nodes = selected_interior_nodes
        .into_iter()
        .filter(|node| used_node_ids.contains(&node.node_id))
        .collect();
    Ok(Ok(refill))
}

fn selected_multi_interior_nodes(
    interior_candidates: &[ConstrainedCavityNode],
    cavity_centroid: Point3,
) -> Vec<ConstrainedCavityNode> {
    let mut nodes = interior_candidates.to_vec();
    nodes.sort_by(|left, right| {
        distance_squared(left.coordinates_m, cavity_centroid)
            .total_cmp(&distance_squared(right.coordinates_m, cavity_centroid))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    nodes.truncate(MAX_MULTI_INTERIOR_REFILL_NODES);
    nodes
}

#[cfg(test)]
fn multi_interior_exact_cover_failure_reason(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
) -> &'static str {
    let mut search =
        BoundaryExactCoverSearch::new(cavity, candidates, options.volume_relative_tolerance);
    let (selected, trace) = search.search_with_trace();
    if selected.is_some() {
        return "multi_interior_exact_cover_candidate_unclassified";
    }
    match trace.dead_end.map(|dead_end| dead_end.reason) {
        Some("attempt_limit") => "multi_interior_exact_cover_attempt_limit",
        Some("volume_overflow") => "multi_interior_exact_cover_volume_overflow",
        Some("boundary_incomplete") => "multi_interior_exact_cover_boundary_incomplete",
        Some("interior_incomplete") => "multi_interior_exact_cover_interior_incomplete",
        Some("volume_mismatch") => "multi_interior_exact_cover_volume_mismatch",
        Some("candidates_exhausted") => "multi_interior_exact_cover_candidates_exhausted",
        Some("boundary_face_candidates_exhausted") => {
            "multi_interior_exact_cover_boundary_face_candidates_exhausted"
        }
        Some("boundary_face_no_raw_candidate") => {
            "multi_interior_exact_cover_boundary_face_no_raw_candidate"
        }
        Some("boundary_face_no_addable_candidate") => {
            "multi_interior_exact_cover_boundary_face_no_addable_candidate"
        }
        Some("interior_face_candidates_exhausted") => {
            "multi_interior_exact_cover_interior_face_candidates_exhausted"
        }
        Some("interior_face_no_raw_candidate") => {
            "multi_interior_exact_cover_interior_face_no_raw_candidate"
        }
        Some("interior_face_no_addable_candidate") => {
            "multi_interior_exact_cover_interior_face_no_addable_candidate"
        }
        Some("forced_interior_mate_no_candidate_contains_face") => {
            "multi_interior_exact_cover_forced_mate_missing_candidate"
        }
        Some("forced_interior_mate_face_count_conflict") => {
            "multi_interior_exact_cover_forced_mate_face_count_conflict"
        }
        Some("forced_interior_mate_future_mate_conflict") => {
            "multi_interior_exact_cover_forced_mate_future_conflict"
        }
        Some("forced_interior_mate_volume_overflow") => {
            "multi_interior_exact_cover_forced_mate_volume_overflow"
        }
        _ => "multi_interior_exact_cover_not_found",
    }
}

#[cfg(not(test))]
fn multi_interior_exact_cover_failure_reason(
    _cavity: &ConstrainedCavity,
    _candidates: &[ConstrainedCavityRefillTetrahedron],
    _options: ConstrainedCavityRefillOptions,
) -> &'static str {
    "multi_interior_exact_cover_not_found"
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_node_completion(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryNodeCompletionDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            refill_tetrahedra.push(tetrahedron);
        }
    }
    let mut aggregate = BoundaryNodeCompletionDiagnostic {
        reason: "boundary_node_completion_no_missing_faces",
        missing_face_count: 0,
        cap_candidate_count: 0,
        outside_candidate_count: 0,
        duplicate_candidate_count: 0,
        max_rejected_scaled_jacobian: 0.0,
        rejected_scaled_jacobian_bins: BTreeMap::new(),
        max_rejected_cap_height_ratio: 0.0,
        rejected_cap_height_ratio_bins: BTreeMap::new(),
        rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_cap_node_ids: BTreeMap::new(),
        split_cap_candidate_count: 0,
        split_cap_pass_count: 0,
        max_split_cap_scaled_jacobian: 0.0,
        split_cap_scaled_jacobian_bins: BTreeMap::new(),
        split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        split_cap_apex_limited_node_ids: BTreeMap::new(),
        edge_split_cap_candidate_count: 0,
        edge_split_cap_pass_count: 0,
        max_edge_split_cap_scaled_jacobian: 0.0,
        edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        three_edge_split_cap_candidate_count: 0,
        three_edge_split_cap_pass_count: 0,
        max_three_edge_split_cap_scaled_jacobian: 0.0,
        three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
        three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    loop {
        let missing_faces = missing_refill_boundary_faces(cavity, &refill_tetrahedra)
            .map_err(ConstrainedCavityRefillError::Validation)?;
        let Some(missing_face) = missing_faces.first().copied() else {
            break;
        };
        aggregate.missing_face_count = missing_faces.len();
        let diagnostic = diagnostic_boundary_face_completion(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tetrahedra,
            &boundary_triangles,
            options,
            missing_faces.len(),
        );
        aggregate.cap_candidate_count += diagnostic.cap_candidate_count;
        aggregate.outside_candidate_count += diagnostic.outside_candidate_count;
        aggregate.duplicate_candidate_count += diagnostic.duplicate_candidate_count;
        aggregate.max_rejected_scaled_jacobian = aggregate
            .max_rejected_scaled_jacobian
            .max(diagnostic.max_rejected_scaled_jacobian);
        aggregate.max_rejected_cap_height_ratio = aggregate
            .max_rejected_cap_height_ratio
            .max(diagnostic.max_rejected_cap_height_ratio);
        for (bin, count) in diagnostic.rejected_scaled_jacobian_bins {
            *aggregate
                .rejected_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_cap_height_ratio_bins {
            *aggregate
                .rejected_cap_height_ratio_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.rejected_scaled_jacobian_worst_corner_bins {
            *aggregate
                .rejected_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.rejected_cap_node_ids {
            *aggregate.rejected_cap_node_ids.entry(node_id).or_default() += count;
        }
        aggregate.split_cap_candidate_count += diagnostic.split_cap_candidate_count;
        aggregate.split_cap_pass_count += diagnostic.split_cap_pass_count;
        aggregate.max_split_cap_scaled_jacobian = aggregate
            .max_split_cap_scaled_jacobian
            .max(diagnostic.max_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_bins {
            *aggregate
                .split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.split_cap_apex_limited_node_ids {
            *aggregate
                .split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.edge_split_cap_candidate_count += diagnostic.edge_split_cap_candidate_count;
        aggregate.edge_split_cap_pass_count += diagnostic.edge_split_cap_pass_count;
        aggregate.max_edge_split_cap_scaled_jacobian = aggregate
            .max_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.edge_split_cap_apex_limited_node_ids {
            *aggregate
                .edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        aggregate.three_edge_split_cap_candidate_count +=
            diagnostic.three_edge_split_cap_candidate_count;
        aggregate.three_edge_split_cap_pass_count += diagnostic.three_edge_split_cap_pass_count;
        aggregate.max_three_edge_split_cap_scaled_jacobian = aggregate
            .max_three_edge_split_cap_scaled_jacobian
            .max(diagnostic.max_three_edge_split_cap_scaled_jacobian);
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_bins
                .entry(bin)
                .or_default() += count;
        }
        for (bin, count) in diagnostic.three_edge_split_cap_scaled_jacobian_worst_corner_bins {
            *aggregate
                .three_edge_split_cap_scaled_jacobian_worst_corner_bins
                .entry(bin)
                .or_default() += count;
        }
        for (node_id, count) in diagnostic.three_edge_split_cap_apex_limited_node_ids {
            *aggregate
                .three_edge_split_cap_apex_limited_node_ids
                .entry(node_id)
                .or_default() += count;
        }
        for (reason, count) in diagnostic.rejected_by_reason {
            *aggregate.rejected_by_reason.entry(reason).or_default() += count;
        }
        let Some(tetrahedron) = best_boundary_face_completion_tetrahedron(
            missing_face,
            cavity,
            &boundary_node_map,
            &refill_tetrahedra,
            &boundary_triangles,
            options,
        ) else {
            aggregate.reason = "boundary_node_completion_no_candidate";
            return Ok(aggregate);
        };
        refill_tetrahedra.push(tetrahedron);
    }
    if aggregate.missing_face_count == 0 {
        return Ok(BoundaryNodeCompletionDiagnostic {
            reason: "boundary_node_completion_no_missing_faces",
            missing_face_count: 0,
            cap_candidate_count: 0,
            outside_candidate_count: 0,
            duplicate_candidate_count: 0,
            max_rejected_scaled_jacobian: 0.0,
            rejected_scaled_jacobian_bins: BTreeMap::new(),
            max_rejected_cap_height_ratio: 0.0,
            rejected_cap_height_ratio_bins: BTreeMap::new(),
            rejected_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            rejected_cap_node_ids: BTreeMap::new(),
            split_cap_candidate_count: 0,
            split_cap_pass_count: 0,
            max_split_cap_scaled_jacobian: 0.0,
            split_cap_scaled_jacobian_bins: BTreeMap::new(),
            split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            split_cap_apex_limited_node_ids: BTreeMap::new(),
            edge_split_cap_candidate_count: 0,
            edge_split_cap_pass_count: 0,
            max_edge_split_cap_scaled_jacobian: 0.0,
            edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            three_edge_split_cap_candidate_count: 0,
            three_edge_split_cap_pass_count: 0,
            max_three_edge_split_cap_scaled_jacobian: 0.0,
            three_edge_split_cap_scaled_jacobian_bins: BTreeMap::new(),
            three_edge_split_cap_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
            three_edge_split_cap_apex_limited_node_ids: BTreeMap::new(),
            rejected_by_reason: BTreeMap::new(),
        });
    }
    aggregate.reason = "boundary_node_completion_completed";
    Ok(aggregate)
}

#[cfg(test)]
pub(crate) fn diagnostic_interior_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    interior_candidates: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<InteriorStarQualityDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = InteriorStarQualityDiagnostic {
        candidate_count: 0,
        pass_count: 0,
        scaled_worst_face_candidate_count: 0,
        scaled_worst_face_pass_count: 0,
        max_min_scaled_jacobian: 0.0,
        max_scaled_worst_face_min_scaled_jacobian: 0.0,
        min_scaled_jacobian_bins: BTreeMap::new(),
        min_scaled_jacobian_worst_corner_bins: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    let mut seen_interior_nodes = BTreeSet::<u32>::new();
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    for node in interior_candidates {
        if !seen_interior_nodes.insert(node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("duplicate_interior_node")
                .or_default() += 1;
            continue;
        }
        if boundary_node_ids.contains(&node.node_id) {
            *diagnostic
                .rejected_by_reason
                .entry("interior_node_reuses_boundary_node")
                .or_default() += 1;
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            *diagnostic
                .rejected_by_reason
                .entry("protected_boundary_distance")
                .or_default() += 1;
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            *diagnostic
                .rejected_by_reason
                .entry("interior_point_outside_cavity")
                .or_default() += 1;
            continue;
        }
        diagnostic.candidate_count += 1;
        match star_refill_candidate_with_rejection_reason(
            cavity,
            &boundary_node_map,
            node.clone(),
            diagnostic_options,
        ) {
            Ok(Ok(refill)) => {
                let min_quality = refill
                    .tetrahedra
                    .iter()
                    .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
                    .fold(f64::INFINITY, f64::min);
                if min_quality.is_finite() {
                    diagnostic.max_min_scaled_jacobian =
                        diagnostic.max_min_scaled_jacobian.max(min_quality);
                    *diagnostic
                        .min_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(min_quality))
                        .or_default() += 1;
                    if let Some(worst_tetrahedron) =
                        refill.tetrahedra.iter().min_by(|left, right| {
                            left.exact_scaled_jacobian
                                .total_cmp(&right.exact_scaled_jacobian)
                        })
                    {
                        let points = worst_tetrahedron.node_ids.map(|node_id| {
                            if node_id == node.node_id {
                                node.coordinates_m
                            } else {
                                boundary_node_map[&node_id]
                            }
                        });
                        *diagnostic
                            .min_scaled_jacobian_worst_corner_bins
                            .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                            .or_default() += 1;
                    }
                    if min_quality >= options.min_scaled_jacobian {
                        diagnostic.pass_count += 1;
                    }
                    if let Some((scaled_count, scaled_quality)) = scaled_worst_face_star_quality(
                        cavity,
                        &boundary_node_map,
                        &boundary_triangles,
                        node,
                        &refill,
                        diagnostic_options,
                    ) {
                        diagnostic.scaled_worst_face_candidate_count += scaled_count;
                        diagnostic.max_scaled_worst_face_min_scaled_jacobian = diagnostic
                            .max_scaled_worst_face_min_scaled_jacobian
                            .max(scaled_quality);
                        diagnostic.scaled_worst_face_pass_count +=
                            usize::from(scaled_quality >= options.min_scaled_jacobian);
                    }
                }
            }
            Ok(Err(reason)) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
            Err(err) => {
                *diagnostic
                    .rejected_by_reason
                    .entry(boundary_node_refill_validation_reason(&err))
                    .or_default() += 1;
            }
        }
    }
    Ok(diagnostic)
}

#[cfg(test)]
fn scaled_worst_face_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    node: &ConstrainedCavityNode,
    refill: &ConstrainedCavityRefill,
    options: ConstrainedCavityRefillOptions,
) -> Option<(usize, f64)> {
    let worst_tetrahedron = refill.tetrahedra.iter().min_by(|left, right| {
        left.exact_scaled_jacobian
            .total_cmp(&right.exact_scaled_jacobian)
    })?;
    let face_nodes = worst_tetrahedron
        .node_ids
        .into_iter()
        .filter(|node_id| *node_id != node.node_id)
        .collect::<Vec<_>>();
    if face_nodes.len() != 3 {
        return None;
    }
    let face_points = face_nodes
        .iter()
        .map(|node_id| boundary_nodes.get(node_id).copied())
        .collect::<Option<Vec<_>>>()?;
    let face_centroid = [
        (face_points[0][0] + face_points[1][0] + face_points[2][0]) / 3.0,
        (face_points[0][1] + face_points[1][1] + face_points[2][1]) / 3.0,
        (face_points[0][2] + face_points[1][2] + face_points[2][2]) / 3.0,
    ];
    let direction = [
        node.coordinates_m[0] - face_centroid[0],
        node.coordinates_m[1] - face_centroid[1],
        node.coordinates_m[2] - face_centroid[2],
    ];
    let distance_squared =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    if !distance_squared.is_finite()
        || distance_squared <= MeshingTolerance::default().absolute_m.powi(2)
    {
        return None;
    }

    let mut candidate_count = 0_usize;
    let mut best_quality = 0.0_f64;
    for scale in [0.5, 0.7, 0.85, 1.15, 1.35, 1.6, 2.0] {
        let coordinates_m = [
            face_centroid[0] + direction[0] * scale,
            face_centroid[1] + direction[1] * scale,
            face_centroid[2] + direction[2] * scale,
        ];
        if point_in_closed_triangle_surface(
            coordinates_m,
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        candidate_count += 1;
        let adjusted = ConstrainedCavityNode {
            node_id: node.node_id,
            coordinates_m,
        };
        let Ok(Ok(refill)) =
            star_refill_candidate_with_rejection_reason(cavity, boundary_nodes, adjusted, options)
        else {
            continue;
        };
        let min_quality = refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if min_quality.is_finite() {
            best_quality = best_quality.max(min_quality);
        }
    }
    (candidate_count > 0).then_some((candidate_count, best_quality))
}

#[cfg(test)]
fn diagnostic_boundary_face_completion(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
    missing_face_count: usize,
) -> BoundaryNodeCompletionDiagnostic {
    let mut cap_candidate_count = 0_usize;
    let mut outside_candidate_count = 0_usize;
    let mut duplicate_candidate_count = 0_usize;
    let mut max_rejected_scaled_jacobian = 0.0_f64;
    let mut rejected_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut max_rejected_cap_height_ratio = 0.0_f64;
    let mut rejected_cap_height_ratio_bins = BTreeMap::<String, usize>::new();
    let mut rejected_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut rejected_cap_node_ids = BTreeMap::<u32, usize>::new();
    let mut split_cap_candidate_count = 0_usize;
    let mut split_cap_pass_count = 0_usize;
    let mut max_split_cap_scaled_jacobian = 0.0_f64;
    let mut split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut split_cap_scaled_jacobian_worst_corner_bins = BTreeMap::<&'static str, usize>::new();
    let mut split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut edge_split_cap_candidate_count = 0_usize;
    let mut edge_split_cap_pass_count = 0_usize;
    let mut max_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut three_edge_split_cap_candidate_count = 0_usize;
    let mut three_edge_split_cap_pass_count = 0_usize;
    let mut max_three_edge_split_cap_scaled_jacobian = 0.0_f64;
    let mut three_edge_split_cap_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut three_edge_split_cap_scaled_jacobian_worst_corner_bins =
        BTreeMap::<&'static str, usize>::new();
    let mut three_edge_split_cap_apex_limited_node_ids = BTreeMap::<u32, usize>::new();
    let mut rejected_by_reason = BTreeMap::<&'static str, usize>::new();
    let mut saw_non_duplicate = false;
    for node_id in cavity_boundary_node_ids(cavity) {
        if face.contains(&node_id) {
            continue;
        }
        let node_ids = [face[0], face[1], face[2], node_id];
        let points = node_ids.map(|id| boundary_nodes[&id]);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            outside_candidate_count += 1;
            continue;
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
            Ok(tetrahedron) => {
                cap_candidate_count += 1;
                if refill_tetrahedra.iter().any(|existing| {
                    sorted_tetrahedron_nodes(existing.node_ids)
                        == sorted_tetrahedron_nodes(tetrahedron.node_ids)
                }) {
                    duplicate_candidate_count += 1;
                } else {
                    saw_non_duplicate = true;
                }
            }
            Err(reason) => {
                *rejected_cap_node_ids.entry(node_id).or_default() += 1;
                let exact_scaled_jacobian = tetrahedron_scaled_jacobian(points);
                if exact_scaled_jacobian.is_finite() {
                    max_rejected_scaled_jacobian =
                        max_rejected_scaled_jacobian.max(exact_scaled_jacobian);
                    *rejected_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(exact_scaled_jacobian))
                        .or_default() += 1;
                    *rejected_scaled_jacobian_worst_corner_bins
                        .entry(diagnostic_scaled_jacobian_worst_corner_label(points))
                        .or_default() += 1;
                }
                let cap_height_ratio =
                    diagnostic_face_apex_height_ratio(face, node_id, boundary_nodes);
                if cap_height_ratio.is_finite() {
                    max_rejected_cap_height_ratio =
                        max_rejected_cap_height_ratio.max(cap_height_ratio);
                    *rejected_cap_height_ratio_bins
                        .entry(diagnostic_height_ratio_bin(cap_height_ratio))
                        .or_default() += 1;
                }
                if let Some((split_min_quality, split_worst_corner)) =
                    diagnostic_split_cap_min_scaled_jacobian(face, node_id, boundary_nodes, options)
                {
                    split_cap_candidate_count += 1;
                    max_split_cap_scaled_jacobian =
                        max_split_cap_scaled_jacobian.max(split_min_quality);
                    *split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(split_min_quality))
                        .or_default() += 1;
                    *split_cap_scaled_jacobian_worst_corner_bins
                        .entry(split_worst_corner)
                        .or_default() += 1;
                    if split_worst_corner == "apex" {
                        *split_cap_apex_limited_node_ids.entry(node_id).or_default() += 1;
                    }
                    if split_min_quality >= options.min_scaled_jacobian {
                        split_cap_pass_count += 1;
                    }
                }
                if let Some((edge_split_min_quality, edge_split_worst_corner)) =
                    diagnostic_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    edge_split_cap_candidate_count += 1;
                    max_edge_split_cap_scaled_jacobian =
                        max_edge_split_cap_scaled_jacobian.max(edge_split_min_quality);
                    *edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(edge_split_min_quality))
                        .or_default() += 1;
                    *edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(edge_split_worst_corner)
                        .or_default() += 1;
                    if edge_split_worst_corner == "apex" {
                        *edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if edge_split_min_quality >= options.min_scaled_jacobian {
                        edge_split_cap_pass_count += 1;
                    }
                }
                if let Some((three_edge_split_min_quality, three_edge_split_worst_corner)) =
                    diagnostic_three_edge_split_cap_min_scaled_jacobian(
                        face,
                        node_id,
                        boundary_nodes,
                        options,
                    )
                {
                    three_edge_split_cap_candidate_count += 1;
                    max_three_edge_split_cap_scaled_jacobian =
                        max_three_edge_split_cap_scaled_jacobian.max(three_edge_split_min_quality);
                    *three_edge_split_cap_scaled_jacobian_bins
                        .entry(diagnostic_scaled_jacobian_bin(three_edge_split_min_quality))
                        .or_default() += 1;
                    *three_edge_split_cap_scaled_jacobian_worst_corner_bins
                        .entry(three_edge_split_worst_corner)
                        .or_default() += 1;
                    if three_edge_split_worst_corner == "apex" {
                        *three_edge_split_cap_apex_limited_node_ids
                            .entry(node_id)
                            .or_default() += 1;
                    }
                    if three_edge_split_min_quality >= options.min_scaled_jacobian {
                        three_edge_split_cap_pass_count += 1;
                    }
                }
                *rejected_by_reason
                    .entry(boundary_node_refill_rejection_reason(reason))
                    .or_default() += 1;
            }
        }
    }
    let reason = if saw_non_duplicate {
        "boundary_node_completion_has_candidate"
    } else if duplicate_candidate_count > 0 {
        "boundary_node_completion_duplicate_tetrahedron"
    } else {
        "boundary_node_completion_no_candidate"
    };
    BoundaryNodeCompletionDiagnostic {
        reason,
        missing_face_count,
        cap_candidate_count,
        outside_candidate_count,
        duplicate_candidate_count,
        max_rejected_scaled_jacobian,
        rejected_scaled_jacobian_bins,
        max_rejected_cap_height_ratio,
        rejected_cap_height_ratio_bins,
        rejected_scaled_jacobian_worst_corner_bins,
        rejected_cap_node_ids,
        split_cap_candidate_count,
        split_cap_pass_count,
        max_split_cap_scaled_jacobian,
        split_cap_scaled_jacobian_bins,
        split_cap_scaled_jacobian_worst_corner_bins,
        split_cap_apex_limited_node_ids,
        edge_split_cap_candidate_count,
        edge_split_cap_pass_count,
        max_edge_split_cap_scaled_jacobian,
        edge_split_cap_scaled_jacobian_bins,
        edge_split_cap_scaled_jacobian_worst_corner_bins,
        edge_split_cap_apex_limited_node_ids,
        three_edge_split_cap_candidate_count,
        three_edge_split_cap_pass_count,
        max_three_edge_split_cap_scaled_jacobian,
        three_edge_split_cap_scaled_jacobian_bins,
        three_edge_split_cap_scaled_jacobian_worst_corner_bins,
        three_edge_split_cap_apex_limited_node_ids,
        rejected_by_reason,
    }
}

#[cfg(test)]
fn diagnostic_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|split_node| {
            split_completion_tetrahedra_for_node(
                face,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tetrahedra| {
                tetrahedra
                    .iter()
                    .map(|tetrahedron| {
                        let points = tetrahedron.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tetrahedron.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[cfg(test)]
fn diagnostic_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    boundary_face_edge_split_node_candidates(face, boundary_nodes)
        .into_iter()
        .filter_map(|(edge, split_node)| {
            edge_split_completion_tetrahedra_for_node(
                face,
                edge,
                cap_node_id,
                &split_node,
                boundary_nodes,
                diagnostic_options,
            )
            .map(|tetrahedra| {
                tetrahedra
                    .iter()
                    .map(|tetrahedron| {
                        let points = tetrahedron.node_ids.map(|node_id| {
                            if node_id == split_node.node_id {
                                split_node.coordinates_m
                            } else {
                                boundary_nodes[&node_id]
                            }
                        });
                        (
                            tetrahedron.exact_scaled_jacobian,
                            diagnostic_scaled_jacobian_worst_corner_label(points),
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .unwrap_or((f64::INFINITY, "face_vertex"))
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

#[cfg(test)]
fn diagnostic_three_edge_split_cap_min_scaled_jacobian(
    face: [u32; 3],
    cap_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
    options: ConstrainedCavityRefillOptions,
) -> Option<(f64, &'static str)> {
    let diagnostic_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let split_nodes = boundary_face_mid_edge_split_nodes(face, boundary_nodes);
    let split_node_by_edge = face_edges(face)
        .into_iter()
        .zip(split_nodes.iter())
        .map(|(edge, node)| (sorted_edge(edge), node.node_id))
        .collect::<BTreeMap<_, _>>();
    let split_node_coordinates = split_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    three_edge_split_completion_tetrahedra_for_node(
        face,
        cap_node_id,
        &split_node_by_edge,
        &split_node_coordinates,
        boundary_nodes,
        diagnostic_options,
    )
    .map(|tetrahedra| {
        tetrahedra
            .iter()
            .map(|tetrahedron| {
                let points = tetrahedron.node_ids.map(|node_id| {
                    split_node_coordinates
                        .get(&node_id)
                        .copied()
                        .unwrap_or_else(|| boundary_nodes[&node_id])
                });
                (
                    tetrahedron.exact_scaled_jacobian,
                    diagnostic_scaled_jacobian_worst_corner_label(points),
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap_or((f64::INFINITY, "face_vertex"))
    })
}

struct BoundaryExactCoverSearch<'a> {
    candidates: &'a [ConstrainedCavityRefillTetrahedron],
    candidate_faces: Vec<[[u32; 3]; 4]>,
    boundary_faces: BTreeSet<[u32; 3]>,
    target_volume_m3: f64,
    volume_tolerance_m3: f64,
    max_attempt_count: usize,
    attempts: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct BoundaryExactCoverSolution {
    selected_indices: Vec<usize>,
    min_scaled_jacobian: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundaryExactCoverRootAvailability {
    zero_raw_candidate_face_count: usize,
    zero_addable_candidate_face_count: usize,
    min_raw_candidate_count: usize,
    min_addable_candidate_count: usize,
    max_addable_candidate_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct BoundaryExactCoverDeadEnd {
    reason: &'static str,
    face: Option<[u32; 3]>,
    depth: usize,
    selected_tetrahedra: Vec<[u32; 4]>,
    selected_roles: Vec<&'static str>,
    current_volume_m3: f64,
    candidate_volume_m3: f64,
    target_volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct BoundaryExactCoverTrace {
    dead_end: Option<BoundaryExactCoverDeadEnd>,
    dead_ends: Vec<BoundaryExactCoverDeadEnd>,
    dead_end_reason_counts: BTreeMap<&'static str, usize>,
    dead_end_faces_by_reason: BTreeMap<&'static str, BTreeSet<[u32; 3]>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ForcedInteriorMateFailure {
    NoAddableMate {
        face: Option<[u32; 3]>,
        reason: ForcedInteriorMateNoAddableReason,
    },
    VolumeOverflow {
        current_volume_m3: f64,
        candidate_volume_m3: f64,
        target_volume_m3: f64,
    },
}

impl ForcedInteriorMateFailure {
    fn reason(self) -> &'static str {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { reason, .. } => reason.as_str(),
            ForcedInteriorMateFailure::VolumeOverflow { .. } => {
                "forced_interior_mate_volume_overflow"
            }
        }
    }

    fn face(self) -> Option<[u32; 3]> {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { face, .. } => face,
            ForcedInteriorMateFailure::VolumeOverflow { .. } => None,
        }
    }

    fn volume(self) -> Option<(f64, f64, f64)> {
        match self {
            ForcedInteriorMateFailure::NoAddableMate { .. } => None,
            ForcedInteriorMateFailure::VolumeOverflow {
                current_volume_m3,
                candidate_volume_m3,
                target_volume_m3,
            } => Some((current_volume_m3, candidate_volume_m3, target_volume_m3)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedInteriorMateNoAddableReason {
    NoCandidateContainsFace,
    FaceCountConflict,
    FutureMateConflict,
}

impl ForcedInteriorMateNoAddableReason {
    fn as_str(self) -> &'static str {
        match self {
            ForcedInteriorMateNoAddableReason::NoCandidateContainsFace => {
                "forced_interior_mate_no_candidate_contains_face"
            }
            ForcedInteriorMateNoAddableReason::FaceCountConflict => {
                "forced_interior_mate_face_count_conflict"
            }
            ForcedInteriorMateNoAddableReason::FutureMateConflict => {
                "forced_interior_mate_future_mate_conflict"
            }
        }
    }
}

impl<'a> BoundaryExactCoverSearch<'a> {
    fn new(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTetrahedron],
        volume_relative_tolerance: f64,
    ) -> Self {
        Self::with_attempt_limit(cavity, candidates, volume_relative_tolerance, 5_000)
    }

    fn with_attempt_limit(
        cavity: &ConstrainedCavity,
        candidates: &'a [ConstrainedCavityRefillTetrahedron],
        volume_relative_tolerance: f64,
        max_attempt_count: usize,
    ) -> Self {
        Self {
            candidates,
            candidate_faces: candidates
                .iter()
                .map(|candidate| tetrahedron_faces(candidate.node_ids).map(sorted_face))
                .collect(),
            boundary_faces: cavity
                .boundary_faces
                .iter()
                .map(|face| sorted_face(face.node_ids))
                .collect(),
            target_volume_m3: cavity.target_volume_m3,
            volume_tolerance_m3: cavity.target_volume_m3.max(1.0e-18) * volume_relative_tolerance,
            max_attempt_count,
            attempts: 0,
        }
    }

    fn search_best(&mut self) -> Option<Vec<usize>> {
        let mut best = None::<BoundaryExactCoverSolution>;
        self.search_best_from(
            0.0,
            f64::INFINITY,
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut best,
        );
        best.map(|solution| solution.selected_indices)
    }

    fn search_with_trace(&mut self) -> (Option<Vec<usize>>, BoundaryExactCoverTrace) {
        let mut trace = BoundaryExactCoverTrace {
            dead_end: None,
            dead_ends: Vec::new(),
            dead_end_reason_counts: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        };
        let result = self.search_from_traced(
            0.0,
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut trace,
        );
        (result, trace)
    }

    #[cfg(test)]
    fn search_without_forced_with_trace(
        &mut self,
    ) -> (Option<Vec<usize>>, BoundaryExactCoverTrace) {
        let mut trace = BoundaryExactCoverTrace {
            dead_end: None,
            dead_ends: Vec::new(),
            dead_end_reason_counts: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        };
        let result = self.search_from_without_forced_traced(
            0.0,
            &mut BTreeMap::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut trace,
        );
        (result, trace)
    }

    fn record_dead_end(
        &self,
        trace: &mut BoundaryExactCoverTrace,
        selected: &[usize],
        selected_roles: &[&'static str],
        reason: &'static str,
    ) {
        self.record_dead_end_for_face(trace, selected, selected_roles, reason, None, None);
    }

    fn record_dead_end_for_face(
        &self,
        trace: &mut BoundaryExactCoverTrace,
        selected: &[usize],
        selected_roles: &[&'static str],
        reason: &'static str,
        face: Option<[u32; 3]>,
        volume: Option<(f64, f64, f64)>,
    ) {
        *trace.dead_end_reason_counts.entry(reason).or_default() += 1;
        if let Some(face) = face {
            trace
                .dead_end_faces_by_reason
                .entry(reason)
                .or_default()
                .insert(face);
        }
        let (current_volume_m3, candidate_volume_m3, target_volume_m3) =
            volume.unwrap_or((0.0, 0.0, self.target_volume_m3));
        let dead_end = BoundaryExactCoverDeadEnd {
            reason,
            face,
            depth: selected.len(),
            selected_tetrahedra: selected
                .iter()
                .map(|candidate_index| self.candidates[*candidate_index].node_ids)
                .collect(),
            selected_roles: selected_roles.to_vec(),
            current_volume_m3,
            candidate_volume_m3,
            target_volume_m3,
        };
        if trace.dead_end.is_none() {
            trace.dead_end = Some(dead_end.clone());
        }
        if trace.dead_ends.len() < 128 {
            trace.dead_ends.push(dead_end);
        }
    }

    #[cfg(test)]
    fn root_boundary_availability(&self) -> BoundaryExactCoverRootAvailability {
        let face_counts = BTreeMap::<[u32; 3], usize>::new();
        let selected = Vec::<usize>::new();
        let mut zero_raw = 0_usize;
        let mut zero_addable = 0_usize;
        let mut min_raw = usize::MAX;
        let mut min_addable = usize::MAX;
        let mut max_addable = 0_usize;
        for face in &self.boundary_faces {
            let raw_count = self
                .candidate_faces
                .iter()
                .filter(|candidate_faces| candidate_faces.contains(face))
                .count();
            let addable_count = (0..self.candidates.len())
                .filter(|candidate_index| {
                    self.candidate_can_be_added_for_face(
                        *candidate_index,
                        *face,
                        &face_counts,
                        &selected,
                    )
                })
                .count();
            zero_raw += usize::from(raw_count == 0);
            zero_addable += usize::from(addable_count == 0);
            min_raw = min_raw.min(raw_count);
            min_addable = min_addable.min(addable_count);
            max_addable = max_addable.max(addable_count);
        }
        if self.boundary_faces.is_empty() {
            min_raw = 0;
            min_addable = 0;
        }
        BoundaryExactCoverRootAvailability {
            zero_raw_candidate_face_count: zero_raw,
            zero_addable_candidate_face_count: zero_addable,
            min_raw_candidate_count: min_raw,
            min_addable_candidate_count: min_addable,
            max_addable_candidate_count: max_addable,
        }
    }

    fn search_from_traced(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
        trace: &mut BoundaryExactCoverTrace,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count {
            self.record_dead_end(trace, selected, selected_roles, "attempt_limit");
            return None;
        }
        if current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
            self.record_dead_end(trace, selected, selected_roles, "volume_overflow");
            return None;
        }
        let (forced_volume_m3, forced_indices) = match self.propagate_forced_interior_mates_traced(
            current_volume_m3,
            face_counts,
            selected,
            selected_roles,
        ) {
            Ok(forced) => forced,
            Err(reason) => {
                self.record_dead_end_for_face(
                    trace,
                    selected,
                    selected_roles,
                    reason.reason(),
                    reason.face(),
                    reason.volume(),
                );
                return None;
            }
        };
        let current_volume_m3 = current_volume_m3 + forced_volume_m3;
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                return Some(selected.clone());
            }
            let reason = if !boundary_ok {
                "boundary_incomplete"
            } else if !interior_ok {
                "interior_incomplete"
            } else {
                "volume_mismatch"
            };
            self.record_dead_end(trace, selected, selected_roles, reason);
            self.rollback_selected_candidates_with_roles(
                &forced_indices,
                face_counts,
                selected,
                selected_roles,
            );
            return None;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            selected_roles.push("branch");
            if let Some(result) = self.search_from_traced(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
                selected_roles,
                trace,
            ) {
                return Some(result);
            }
            selected_roles.pop();
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        let (reason, face) = self.candidates_exhausted_reason_and_face(face_counts, selected);
        self.record_dead_end_for_face(trace, selected, selected_roles, reason, face, None);
        self.rollback_selected_candidates_with_roles(
            &forced_indices,
            face_counts,
            selected,
            selected_roles,
        );
        None
    }

    #[cfg(test)]
    fn search_from_without_forced_traced(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
        trace: &mut BoundaryExactCoverTrace,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count {
            self.record_dead_end(trace, selected, selected_roles, "attempt_limit");
            return None;
        }
        if current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
            self.record_dead_end(trace, selected, selected_roles, "volume_overflow");
            return None;
        }
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                return Some(selected.clone());
            }
            let reason = if !boundary_ok {
                "boundary_incomplete"
            } else if !interior_ok {
                "interior_incomplete"
            } else {
                "volume_mismatch"
            };
            self.record_dead_end(trace, selected, selected_roles, reason);
            return None;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            selected_roles.push("branch");
            if let Some(result) = self.search_from_without_forced_traced(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
                selected_roles,
                trace,
            ) {
                return Some(result);
            }
            selected_roles.pop();
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        self.record_dead_end(
            trace,
            selected,
            selected_roles,
            self.candidates_exhausted_reason(face_counts, selected),
        );
        None
    }

    #[cfg(test)]
    fn search(&mut self) -> Option<Vec<usize>> {
        self.search_from(0.0, &mut BTreeMap::new(), &mut Vec::new())
    }

    #[cfg(test)]
    fn search_from(
        &mut self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count
            || current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3
        {
            return None;
        }
        let Some((forced_volume_m3, forced_indices)) =
            self.propagate_forced_interior_mates(current_volume_m3, face_counts, selected)
        else {
            return None;
        };
        let current_volume_m3 = current_volume_m3 + forced_volume_m3;
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                return Some(selected.clone());
            }
            self.rollback_selected_candidates(&forced_indices, face_counts, selected);
            return None;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            if let Some(result) = self.search_from(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                face_counts,
                selected,
            ) {
                return Some(result);
            }
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        self.rollback_selected_candidates(&forced_indices, face_counts, selected);
        None
    }

    fn search_best_from(
        &mut self,
        current_volume_m3: f64,
        current_min_scaled_jacobian: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        best: &mut Option<BoundaryExactCoverSolution>,
    ) {
        self.attempts += 1;
        if self.attempts > self.max_attempt_count
            || current_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3
        {
            return;
        }
        if best
            .as_ref()
            .is_some_and(|solution| current_min_scaled_jacobian <= solution.min_scaled_jacobian)
        {
            return;
        }
        let Some((forced_volume_m3, forced_indices)) =
            self.propagate_forced_interior_mates(current_volume_m3, face_counts, selected)
        else {
            return;
        };
        let current_volume_m3 = current_volume_m3 + forced_volume_m3;
        let current_min_scaled_jacobian = forced_indices
            .iter()
            .map(|index| self.candidates[*index].exact_scaled_jacobian)
            .fold(current_min_scaled_jacobian, f64::min);
        if best
            .as_ref()
            .is_some_and(|solution| current_min_scaled_jacobian <= solution.min_scaled_jacobian)
        {
            self.rollback_selected_candidates(&forced_indices, face_counts, selected);
            return;
        }
        let Some(candidate_indices) = self.next_cover_candidates(face_counts, selected) else {
            let boundary_ok = self
                .boundary_faces
                .iter()
                .all(|face| face_counts.get(face).copied().unwrap_or(0) == 1);
            let interior_ok = face_counts
                .iter()
                .all(|(face, count)| self.boundary_faces.contains(face) || *count == 2);
            if boundary_ok
                && interior_ok
                && (current_volume_m3 - self.target_volume_m3).abs() <= self.volume_tolerance_m3
            {
                *best = Some(BoundaryExactCoverSolution {
                    selected_indices: selected.clone(),
                    min_scaled_jacobian: current_min_scaled_jacobian,
                });
            }
            self.rollback_selected_candidates(&forced_indices, face_counts, selected);
            return;
        };
        for candidate_index in candidate_indices {
            if selected.contains(&candidate_index) {
                continue;
            }
            for face in self.candidate_faces[candidate_index] {
                *face_counts.entry(face).or_default() += 1;
            }
            selected.push(candidate_index);
            self.search_best_from(
                current_volume_m3 + self.candidates[candidate_index].volume_m3,
                current_min_scaled_jacobian
                    .min(self.candidates[candidate_index].exact_scaled_jacobian),
                face_counts,
                selected,
                best,
            );
            selected.pop();
            self.remove_candidate_faces(candidate_index, face_counts);
        }
        self.rollback_selected_candidates(&forced_indices, face_counts, selected);
    }

    fn propagate_forced_interior_mates(
        &self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) -> Option<(f64, Vec<usize>)> {
        let mut forced_indices = Vec::<usize>::new();
        let mut forced_volume_m3 = 0.0;
        loop {
            let forced_candidate = self.forced_interior_mate(face_counts, selected)?;
            let Some(candidate_index) = forced_candidate else {
                return Some((forced_volume_m3, forced_indices));
            };
            let next_volume_m3 =
                current_volume_m3 + forced_volume_m3 + self.candidates[candidate_index].volume_m3;
            if next_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
                self.rollback_selected_candidates(&forced_indices, face_counts, selected);
                return None;
            }
            self.add_candidate_faces(candidate_index, face_counts);
            selected.push(candidate_index);
            forced_indices.push(candidate_index);
            forced_volume_m3 += self.candidates[candidate_index].volume_m3;
        }
    }

    fn propagate_forced_interior_mates_traced(
        &self,
        current_volume_m3: f64,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
    ) -> Result<(f64, Vec<usize>), ForcedInteriorMateFailure> {
        let mut forced_indices = Vec::<usize>::new();
        let mut forced_volume_m3 = 0.0;
        loop {
            let forced_candidate = self
                .forced_interior_mate_traced(face_counts, selected)
                .ok_or_else(|| {
                    let (face, reason) =
                        self.forced_interior_mate_no_addable_reason(face_counts, selected);
                    ForcedInteriorMateFailure::NoAddableMate { face, reason }
                })?;
            let Some(candidate_index) = forced_candidate else {
                return Ok((forced_volume_m3, forced_indices));
            };
            let next_volume_m3 =
                current_volume_m3 + forced_volume_m3 + self.candidates[candidate_index].volume_m3;
            if next_volume_m3 > self.target_volume_m3 + self.volume_tolerance_m3 {
                self.rollback_selected_candidates_with_roles(
                    &forced_indices,
                    face_counts,
                    selected,
                    selected_roles,
                );
                return Err(ForcedInteriorMateFailure::VolumeOverflow {
                    current_volume_m3: current_volume_m3 + forced_volume_m3,
                    candidate_volume_m3: self.candidates[candidate_index].volume_m3,
                    target_volume_m3: self.target_volume_m3,
                });
            }
            self.add_candidate_faces(candidate_index, face_counts);
            selected.push(candidate_index);
            selected_roles.push("forced");
            forced_indices.push(candidate_index);
            forced_volume_m3 += self.candidates[candidate_index].volume_m3;
        }
    }

    fn forced_interior_mate(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Option<usize>> {
        self.forced_interior_mate_traced(face_counts, selected)
    }

    fn forced_interior_mate_traced(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Option<usize>> {
        let mut forced = None::<usize>;
        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            face,
                            face_counts,
                            selected,
                        )
                })
                .collect::<Vec<_>>();
            match candidates.as_slice() {
                [] => return None,
                [candidate] => {
                    if forced.is_none() {
                        forced = Some(*candidate);
                    }
                }
                _ => {}
            }
        }
        Some(forced)
    }

    fn forced_interior_mate_no_addable_reason(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> (Option<[u32; 3]>, ForcedInteriorMateNoAddableReason) {
        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let mate_indices = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_faces[*candidate_index].contains(&face)
                })
                .collect::<Vec<_>>();
            if mate_indices.is_empty() {
                return (
                    Some(face),
                    ForcedInteriorMateNoAddableReason::NoCandidateContainsFace,
                );
            }
            if mate_indices.iter().all(|candidate_index| {
                self.candidate_faces[*candidate_index]
                    .iter()
                    .any(|candidate_face| {
                        let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                        if self.boundary_faces.contains(candidate_face) {
                            count != 0
                        } else {
                            count >= 2
                        }
                    })
            }) {
                return (
                    Some(face),
                    ForcedInteriorMateNoAddableReason::FaceCountConflict,
                );
            }
            if mate_indices.iter().all(|candidate_index| {
                !self.candidate_faces[*candidate_index]
                    .iter()
                    .all(|candidate_face| {
                        let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                        self.boundary_faces.contains(candidate_face)
                            || count == 1
                            || self.interior_face_has_future_mate(
                                *candidate_index,
                                *candidate_face,
                                face_counts,
                                selected,
                            )
                    })
            }) {
                return (
                    Some(face),
                    ForcedInteriorMateNoAddableReason::FutureMateConflict,
                );
            }
        }
        (
            None,
            ForcedInteriorMateNoAddableReason::NoCandidateContainsFace,
        )
    }

    fn rollback_selected_candidates(
        &self,
        indices: &[usize],
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
    ) {
        for candidate_index in indices.iter().rev() {
            let Some(position) = selected
                .iter()
                .rposition(|selected_index| selected_index == candidate_index)
            else {
                continue;
            };
            selected.remove(position);
            self.remove_candidate_faces(*candidate_index, face_counts);
        }
    }

    fn rollback_selected_candidates_with_roles(
        &self,
        indices: &[usize],
        face_counts: &mut BTreeMap<[u32; 3], usize>,
        selected: &mut Vec<usize>,
        selected_roles: &mut Vec<&'static str>,
    ) {
        for candidate_index in indices.iter().rev() {
            let Some(position) = selected
                .iter()
                .rposition(|selected_index| selected_index == candidate_index)
            else {
                continue;
            };
            selected.remove(position);
            if position < selected_roles.len() {
                selected_roles.remove(position);
            }
            self.remove_candidate_faces(*candidate_index, face_counts);
        }
    }

    fn add_candidate_faces(
        &self,
        candidate_index: usize,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
    ) {
        for face in self.candidate_faces[candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    fn remove_candidate_faces(
        &self,
        candidate_index: usize,
        face_counts: &mut BTreeMap<[u32; 3], usize>,
    ) {
        for face in self.candidate_faces[candidate_index] {
            if let Some(count) = face_counts.get_mut(&face) {
                *count -= 1;
                if *count == 0 {
                    face_counts.remove(&face);
                }
            }
        }
    }

    fn next_cover_candidates(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> Option<Vec<usize>> {
        let mut best = None::<Vec<usize>>;
        for face in self
            .boundary_faces
            .iter()
            .filter(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
        {
            let mut candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            *face,
                            face_counts,
                            selected,
                        )
                })
                .collect::<Vec<_>>();
            self.sort_cover_candidates(&mut candidates);
            if best
                .as_ref()
                .is_none_or(|current| candidates.len() < current.len())
            {
                best = Some(candidates);
            }
        }
        if best.is_some() {
            return best;
        }

        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let mut candidates = (0..self.candidates.len())
                .filter(|candidate_index| {
                    !selected.contains(candidate_index)
                        && self.candidate_can_be_added_for_face(
                            *candidate_index,
                            face,
                            face_counts,
                            selected,
                        )
                })
                .collect::<Vec<_>>();
            self.sort_cover_candidates(&mut candidates);
            if best
                .as_ref()
                .is_none_or(|current| candidates.len() < current.len())
            {
                best = Some(candidates);
            }
        }
        best
    }

    #[cfg(test)]
    fn candidates_exhausted_reason(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> &'static str {
        self.candidates_exhausted_reason_and_face(face_counts, selected)
            .0
    }

    fn candidates_exhausted_reason_and_face(
        &self,
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> (&'static str, Option<[u32; 3]>) {
        for face in self
            .boundary_faces
            .iter()
            .filter(|face| face_counts.get(*face).copied().unwrap_or(0) == 0)
        {
            let raw_count = self.raw_candidate_count_for_face(*face, selected);
            if raw_count == 0 {
                return ("boundary_face_no_raw_candidate", Some(*face));
            }
            let addable_count = self.addable_candidate_count_for_face(*face, face_counts, selected);
            if addable_count == 0 {
                return ("boundary_face_no_addable_candidate", Some(*face));
            }
            return ("boundary_face_candidates_exhausted", Some(*face));
        }

        for face in face_counts.iter().filter_map(|(face, count)| {
            (!self.boundary_faces.contains(face) && *count == 1).then_some(*face)
        }) {
            let raw_count = self.raw_candidate_count_for_face(face, selected);
            if raw_count == 0 {
                return ("interior_face_no_raw_candidate", Some(face));
            }
            let addable_count = self.addable_candidate_count_for_face(face, face_counts, selected);
            if addable_count == 0 {
                return ("interior_face_no_addable_candidate", Some(face));
            }
            return ("interior_face_candidates_exhausted", Some(face));
        }

        ("candidates_exhausted", None)
    }

    fn raw_candidate_count_for_face(&self, face: [u32; 3], selected: &[usize]) -> usize {
        (0..self.candidates.len())
            .filter(|candidate_index| {
                !selected.contains(candidate_index)
                    && self.candidate_faces[*candidate_index].contains(&face)
            })
            .count()
    }

    fn addable_candidate_count_for_face(
        &self,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> usize {
        (0..self.candidates.len())
            .filter(|candidate_index| {
                !selected.contains(candidate_index)
                    && self.candidate_can_be_added_for_face(
                        *candidate_index,
                        face,
                        face_counts,
                        selected,
                    )
            })
            .count()
    }

    fn candidate_can_be_added_for_face(
        &self,
        candidate_index: usize,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> bool {
        self.candidate_faces[candidate_index].contains(&face)
            && self.candidate_faces[candidate_index]
                .iter()
                .all(|candidate_face| {
                    let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                    if self.boundary_faces.contains(candidate_face) {
                        count == 0
                    } else {
                        count < 2
                    }
                })
            && self.candidate_faces[candidate_index]
                .iter()
                .all(|candidate_face| {
                    let count = face_counts.get(candidate_face).copied().unwrap_or(0);
                    self.boundary_faces.contains(candidate_face)
                        || count == 1
                        || self.interior_face_has_future_mate(
                            candidate_index,
                            *candidate_face,
                            face_counts,
                            selected,
                        )
                })
    }

    fn interior_face_has_future_mate(
        &self,
        candidate_index: usize,
        face: [u32; 3],
        face_counts: &BTreeMap<[u32; 3], usize>,
        selected: &[usize],
    ) -> bool {
        (0..self.candidates.len()).any(|mate_index| {
            !selected.contains(&mate_index)
                && mate_index != candidate_index
                && self.candidate_faces[mate_index].contains(&face)
                && self.candidate_faces[mate_index].iter().all(|mate_face| {
                    let count = face_counts.get(mate_face).copied().unwrap_or(0)
                        + usize::from(self.candidate_faces[candidate_index].contains(mate_face));
                    if self.boundary_faces.contains(mate_face) {
                        count == 0
                    } else {
                        count < 2
                    }
                })
        })
    }

    fn sort_cover_candidates(&self, candidates: &mut [usize]) {
        candidates.sort_by(|left, right| {
            self.candidates[*right]
                .exact_scaled_jacobian
                .total_cmp(&self.candidates[*left].exact_scaled_jacobian)
        });
    }
}

fn boundary_node_exact_cover_refill_candidate(
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

fn exact_cover_refill_from_candidate_tetrahedra(
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

fn exact_cover_refill_from_on_demand_interior_mates(
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

#[cfg(test)]
fn exact_cover_trace_faces_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<[u32; 3]>> {
    trace
        .dead_end_faces_by_reason
        .iter()
        .map(|(reason, faces)| (*reason, faces.iter().copied().collect::<Vec<_>>()))
        .collect()
}

#[cfg(test)]
fn exact_cover_trace_selected_tetrahedra_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<[u32; 4]>> {
    let mut selected_tetrahedra_by_reason = BTreeMap::<&'static str, Vec<[u32; 4]>>::new();
    for dead_end in &trace.dead_ends {
        selected_tetrahedra_by_reason
            .entry(dead_end.reason)
            .or_insert_with(|| dead_end.selected_tetrahedra.clone());
    }
    selected_tetrahedra_by_reason
}

#[cfg(test)]
fn exact_cover_trace_selected_roles_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut selected_roles_by_reason = BTreeMap::<&'static str, Vec<&'static str>>::new();
    for dead_end in &trace.dead_ends {
        selected_roles_by_reason
            .entry(dead_end.reason)
            .or_insert_with(|| dead_end.selected_roles.clone());
    }
    selected_roles_by_reason
}

#[cfg(test)]
fn diagnostic_unforced_exact_cover_for_candidates(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    volume_relative_tolerance: f64,
) -> (bool, usize, usize, BTreeMap<&'static str, usize>) {
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        candidates,
        volume_relative_tolerance,
        250,
    );
    let (selected, trace) = search.search_without_forced_with_trace();
    (
        selected.is_some(),
        selected.map(|selected| selected.len()).unwrap_or(0),
        search.attempts,
        trace.dead_end_reason_counts,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundaryExactCoverDiagnostic {
        boundary_node_count: node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        solid_candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        zero_candidate_boundary_faces: Vec::new(),
        min_boundary_face_candidate_count: 0,
        min_candidate_boundary_faces: Vec::new(),
        max_boundary_face_candidate_count: 0,
        zero_solid_candidate_boundary_face_count: 0,
        zero_solid_candidate_boundary_faces: Vec::new(),
        min_solid_boundary_face_candidate_count: 0,
        min_solid_candidate_boundary_faces: Vec::new(),
        max_solid_boundary_face_candidate_count: 0,
        zero_addable_boundary_face_count: 0,
        zero_addable_boundary_faces: Vec::new(),
        min_addable_boundary_face_candidate_count: 0,
        min_addable_candidate_boundary_faces: Vec::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
    };
    if node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut solid_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        solid_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    diagnostic.solid_candidate_count = solid_candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    let solid_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            solid_candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.min_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    diagnostic.zero_solid_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(solid_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_solid_candidate_boundary_face_count = solid_face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_solid_boundary_face_candidate_count = solid_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    diagnostic.min_solid_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(solid_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_solid_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    diagnostic.max_solid_boundary_face_candidate_count = solid_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let root_face_counts = BTreeMap::<[u32; 3], usize>::new();
    let root_selected = Vec::<usize>::new();
    let addable_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            search.addable_candidate_count_for_face(*face, &root_face_counts, &root_selected)
        })
        .collect::<Vec<_>>();
    diagnostic.zero_addable_boundary_faces = boundary_faces
        .iter()
        .zip(addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_addable_boundary_face_count = addable_face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_addable_boundary_face_candidate_count = addable_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    diagnostic.min_addable_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_addable_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.dead_end_reason = dead_end.reason;
        diagnostic.dead_end_face = dead_end.face;
        diagnostic.dead_end_depth = dead_end.depth;
        diagnostic.dead_end_selected_tetrahedra = dead_end.selected_tetrahedra;
        diagnostic.dead_end_current_volume_m3 = dead_end.current_volume_m3;
        diagnostic.dead_end_candidate_volume_m3 = dead_end.candidate_volume_m3;
        diagnostic.dead_end_target_volume_m3 = dead_end.target_volume_m3;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_mates_for_face(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverMateDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }

    let search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    let selected = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            selected_keys
                .contains(&sorted_tetrahedron_nodes(candidate.node_ids))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for candidate_index in &selected {
        for face in search.candidate_faces[*candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let target_face = sorted_face(target_face);
    let mut diagnostics = Vec::<BoundaryExactCoverMateCandidateDiagnostic>::new();
    for candidate_index in 0..candidates.len() {
        if selected.contains(&candidate_index)
            || !search.candidate_faces[candidate_index].contains(&target_face)
        {
            continue;
        }
        let mut conflicting_faces = Vec::<[u32; 3]>::new();
        let mut missing_future_mate_faces = Vec::<[u32; 3]>::new();
        for candidate_face in search.candidate_faces[candidate_index] {
            let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
            if if boundary_faces.contains(&candidate_face) {
                count != 0
            } else {
                count >= 2
            } {
                conflicting_faces.push(candidate_face);
            }
            if !boundary_faces.contains(&candidate_face)
                && count == 0
                && !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                )
            {
                missing_future_mate_faces.push(candidate_face);
            }
        }
        let addable = search.candidate_can_be_added_for_face(
            candidate_index,
            target_face,
            &face_counts,
            &selected,
        );
        diagnostics.push(BoundaryExactCoverMateCandidateDiagnostic {
            node_ids: candidates[candidate_index].node_ids,
            exact_scaled_jacobian: candidates[candidate_index].exact_scaled_jacobian,
            addable,
            conflicting_faces,
            missing_future_mate_faces,
        });
    }
    diagnostics.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverMateDiagnostic {
        target_face,
        candidate_count: diagnostics.len(),
        addable_count: diagnostics
            .iter()
            .filter(|candidate| candidate.addable)
            .count(),
        candidates: diagnostics,
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_support_node_exact_cover_mates_for_face(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverMateDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
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
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    let search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    let selected = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            selected_keys
                .contains(&sorted_tetrahedron_nodes(candidate.node_ids))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for candidate_index in &selected {
        for face in search.candidate_faces[*candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let target_face = sorted_face(target_face);
    let mut diagnostics = Vec::<BoundaryExactCoverMateCandidateDiagnostic>::new();
    for candidate_index in 0..candidates.len() {
        if selected.contains(&candidate_index)
            || !search.candidate_faces[candidate_index].contains(&target_face)
        {
            continue;
        }
        let mut conflicting_faces = Vec::<[u32; 3]>::new();
        let mut missing_future_mate_faces = Vec::<[u32; 3]>::new();
        for candidate_face in search.candidate_faces[candidate_index] {
            let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
            if if boundary_faces.contains(&candidate_face) {
                count != 0
            } else {
                count >= 2
            } {
                conflicting_faces.push(candidate_face);
            }
            if !boundary_faces.contains(&candidate_face)
                && count == 0
                && !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                )
            {
                missing_future_mate_faces.push(candidate_face);
            }
        }
        let addable = search.candidate_can_be_added_for_face(
            candidate_index,
            target_face,
            &face_counts,
            &selected,
        );
        diagnostics.push(BoundaryExactCoverMateCandidateDiagnostic {
            node_ids: candidates[candidate_index].node_ids,
            exact_scaled_jacobian: candidates[candidate_index].exact_scaled_jacobian,
            addable,
            conflicting_faces,
            missing_future_mate_faces,
        });
    }
    diagnostics.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverMateDiagnostic {
        target_face,
        candidate_count: diagnostics.len(),
        addable_count: diagnostics
            .iter()
            .filter(|candidate| candidate.addable)
            .count(),
        candidates: diagnostics,
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_face_candidate_sources(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverFaceCandidateSourceDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let target_face = sorted_face(target_face);
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = BoundaryExactCoverFaceCandidateSourceDiagnostic {
        target_face,
        fourth_node_count: 0,
        centroid_inside_count: 0,
        solid_pass_count: 0,
        relaxed_pass_count: 0,
        outside_surface_count: 0,
        solid_rejected_by_reason: BTreeMap::new(),
        relaxed_rejected_by_reason: BTreeMap::new(),
        relaxed_candidate_node_ids: Vec::new(),
    };
    let face_nodes = target_face
        .map(|node_id| boundary_node_map.get(&node_id).copied())
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(ConstrainedCavityRefillError::MissingBoundaryNode {
            node_id: target_face[0],
        })?;
    for fourth_node_id in cavity_boundary_node_ids(cavity) {
        if target_face.contains(&fourth_node_id) {
            continue;
        }
        let Some(fourth_point) = boundary_node_map.get(&fourth_node_id).copied() else {
            return Err(ConstrainedCavityRefillError::MissingBoundaryNode {
                node_id: fourth_node_id,
            });
        };
        diagnostic.fourth_node_count += 1;
        let node_ids = [
            target_face[0],
            target_face[1],
            target_face[2],
            fourth_node_id,
        ];
        let points = [face_nodes[0], face_nodes[1], face_nodes[2], fourth_point];
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            diagnostic.outside_surface_count += 1;
            continue;
        }
        diagnostic.centroid_inside_count += 1;
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
            Ok(_) => diagnostic.solid_pass_count += 1,
            Err(reason) => {
                *diagnostic
                    .solid_rejected_by_reason
                    .entry(reason)
                    .or_default() += 1
            }
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, relaxed_options) {
            Ok(tetrahedron) => {
                diagnostic.relaxed_pass_count += 1;
                diagnostic
                    .relaxed_candidate_node_ids
                    .push(sorted_tetrahedron_nodes(tetrahedron.node_ids));
            }
            Err(reason) => {
                *diagnostic
                    .relaxed_rejected_by_reason
                    .entry(reason)
                    .or_default() += 1
            }
        }
    }
    diagnostic.relaxed_candidate_node_ids.sort();
    diagnostic.relaxed_candidate_node_ids.dedup();
    Ok(diagnostic)
}

pub fn selected_exact_cover_face_count_blockers(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverFaceCountBlockers, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let target_face = sorted_face(target_face);
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }

    let mut blockers = Vec::<BoundaryExactCoverFaceCountBlocker>::new();
    let mut candidate_count = 0_usize;
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
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .contains(&target_face)
                    {
                        continue;
                    }
                    if selected_keys.contains(&sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    candidate_count += 1;
                    let mut conflicting_faces = Vec::<[u32; 3]>::new();
                    let mut blocking_selected_tetrahedra = Vec::<[u32; 4]>::new();
                    for candidate_face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
                        let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
                        let conflicts = if boundary_faces.contains(&candidate_face) {
                            count != 0
                        } else {
                            count >= 2
                        };
                        if conflicts {
                            conflicting_faces.push(candidate_face);
                            if let Some(selected_tetrahedra) =
                                selected_tetrahedra_by_face.get(&candidate_face)
                            {
                                blocking_selected_tetrahedra
                                    .extend(selected_tetrahedra.iter().copied());
                            }
                        }
                    }
                    if !conflicting_faces.is_empty() {
                        blocking_selected_tetrahedra.sort();
                        blocking_selected_tetrahedra.dedup();
                        blockers.push(BoundaryExactCoverFaceCountBlocker {
                            node_ids: tetrahedron.node_ids,
                            exact_scaled_jacobian: tetrahedron.exact_scaled_jacobian,
                            conflicting_faces,
                            blocking_selected_tetrahedra,
                        });
                    }
                }
            }
        }
    }
    blockers.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverFaceCountBlockers {
        target_face,
        selected_tetrahedron_count: selected_tetrahedron_node_ids.len(),
        candidate_count,
        blocker_count: blockers.len(),
        blockers,
    })
}

pub fn selected_exact_cover_saturated_component(
    cavity: &ConstrainedCavity,
    selected_tetrahedron_node_ids: &[[u32; 4]],
    seed_face: [u32; 3],
) -> BoundaryExactCoverSaturatedComponent {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let seed_face = sorted_face(seed_face);
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }
    let saturated_faces = selected_tetrahedra_by_face
        .iter()
        .filter_map(|(face, selected_tetrahedra)| {
            (!boundary_faces.contains(face) && selected_tetrahedra.len() >= 2).then_some(*face)
        })
        .collect::<BTreeSet<_>>();
    let mut component_faces = BTreeSet::<[u32; 3]>::new();
    let mut component_tetrahedra = BTreeSet::<[u32; 4]>::new();
    let mut pending = Vec::<[u32; 3]>::new();
    if saturated_faces.contains(&seed_face) {
        pending.push(seed_face);
    }
    while let Some(face) = pending.pop() {
        if !component_faces.insert(face) {
            continue;
        }
        let Some(selected_tetrahedra) = selected_tetrahedra_by_face.get(&face) else {
            continue;
        };
        for selected_tetrahedron in selected_tetrahedra {
            if component_tetrahedra.insert(*selected_tetrahedron) {
                for adjacent_face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
                    if saturated_faces.contains(&adjacent_face)
                        && !component_faces.contains(&adjacent_face)
                    {
                        pending.push(adjacent_face);
                    }
                }
            }
        }
    }
    BoundaryExactCoverSaturatedComponent {
        seed_face,
        saturated_face_count: saturated_faces.len(),
        component_face_count: component_faces.len(),
        component_tetrahedron_count: component_tetrahedra.len(),
        component_faces: component_faces.into_iter().collect(),
        component_tetrahedra: component_tetrahedra.into_iter().collect(),
    }
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_interior_mate_closure(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
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
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    let touches_boundary = tetrahedron_faces(tetrahedron.node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
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
    let initial_candidate_count = candidates.len();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }
    for _ in 0..4 {
        let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
        for candidate in &candidates {
            for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
                *face_counts.entry(face).or_default() += 1;
            }
        }
        let missing_faces = face_counts
            .iter()
            .filter_map(|(face, count)| {
                (!boundary_faces.contains(face) && *count == 1).then_some(*face)
            })
            .collect::<Vec<_>>();
        if missing_faces.is_empty() {
            break;
        }
        let mut added = false;
        for face in missing_faces {
            if let Some(indices) = all_candidates_by_face.get(&face) {
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
        }
        if !added {
            break;
        }
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let (selected, trace) = search.search_with_trace();
    let Some(selected) = selected else {
        let dead_end = trace.dead_end.clone();
        return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
            initial_candidate_count,
            candidate_count: candidates.len(),
            injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
            found_cover: false,
            selected_tetrahedron_count: 0,
            search_attempt_count: search.attempts,
            reason: if search.attempts > 5_000 {
                "search_exhausted"
            } else {
                "cover_not_found"
            },
            dead_end_reason: dead_end
                .as_ref()
                .map(|dead_end| dead_end.reason)
                .unwrap_or("not_evaluated"),
            dead_end_face: dead_end.as_ref().and_then(|dead_end| dead_end.face),
            dead_end_depth: dead_end
                .as_ref()
                .map(|dead_end| dead_end.depth)
                .unwrap_or(0),
            dead_end_selected_tetrahedra: dead_end
                .as_ref()
                .map(|dead_end| dead_end.selected_tetrahedra.clone())
                .unwrap_or_default(),
            dead_end_current_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.current_volume_m3)
                .unwrap_or(0.0),
            dead_end_candidate_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.candidate_volume_m3)
                .unwrap_or(0.0),
            dead_end_target_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.target_volume_m3)
                .unwrap_or(0.0),
            dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
            dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
            dead_end_selected_tetrahedra_by_reason: exact_cover_trace_selected_tetrahedra_by_reason(
                &trace,
            ),
            dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(&trace),
            unforced_found_cover: false,
            unforced_selected_tetrahedron_count: 0,
            unforced_search_attempt_count: 0,
            unforced_dead_end_reason_histogram: BTreeMap::new(),
        });
    };
    Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
        initial_candidate_count,
        candidate_count: candidates.len(),
        injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
        found_cover: true,
        selected_tetrahedron_count: selected.len(),
        search_attempt_count: search.attempts,
        reason: "cover_found",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
        dead_end_faces_by_reason: BTreeMap::new(),
        dead_end_selected_tetrahedra_by_reason: BTreeMap::new(),
        dead_end_selected_roles_by_reason: BTreeMap::new(),
        unforced_found_cover: false,
        unforced_selected_tetrahedron_count: 0,
        unforced_search_attempt_count: 0,
        unforced_dead_end_reason_histogram: BTreeMap::new(),
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
        cavity,
        boundary_nodes,
        &[],
        options,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    excluded_tetrahedron_node_ids: &[[u32; 4]],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
    let excluded_keys = excluded_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
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
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    let touches_boundary = tetrahedron_faces(tetrahedron.node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    let candidate_key = sorted_tetrahedron_nodes(tetrahedron.node_ids);
                    if touches_boundary
                        && !excluded_keys.contains(&candidate_key)
                        && candidate_keys.insert(candidate_key)
                    {
                        candidates.push(tetrahedron.clone());
                    }
                    all_candidates.push(tetrahedron);
                }
            }
        }
    }
    let initial_candidate_count = candidates.len();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }
    let mut total_attempts = 0_usize;
    for _ in 0..64 {
        let mut search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let (selected, trace) = search.search_with_trace();
        total_attempts += search.attempts;
        if let Some(selected) = selected {
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: true,
                selected_tetrahedron_count: selected.len(),
                search_attempt_count: total_attempts,
                reason: "cover_found",
                dead_end_reason: "not_evaluated",
                dead_end_face: None,
                dead_end_depth: 0,
                dead_end_selected_tetrahedra: Vec::new(),
                dead_end_current_volume_m3: 0.0,
                dead_end_candidate_volume_m3: 0.0,
                dead_end_target_volume_m3: 0.0,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover: false,
                unforced_selected_tetrahedron_count: 0,
                unforced_search_attempt_count: 0,
                unforced_dead_end_reason_histogram: BTreeMap::new(),
            });
        }
        let Some(dead_end) = trace.dead_end.clone() else {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: "cover_not_found",
                dead_end_reason: "not_evaluated",
                dead_end_face: None,
                dead_end_depth: 0,
                dead_end_selected_tetrahedra: Vec::new(),
                dead_end_current_volume_m3: 0.0,
                dead_end_candidate_volume_m3: 0.0,
                dead_end_target_volume_m3: 0.0,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        };
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
        if future_mate_dead_ends.is_empty() && no_candidate_dead_end_faces.is_empty() {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: if total_attempts > 5_000 {
                    "search_exhausted"
                } else {
                    "cover_not_found"
                },
                dead_end_reason: dead_end.reason,
                dead_end_face: dead_end.face,
                dead_end_depth: dead_end.depth,
                dead_end_selected_tetrahedra: dead_end.selected_tetrahedra,
                dead_end_current_volume_m3: dead_end.current_volume_m3,
                dead_end_candidate_volume_m3: dead_end.candidate_volume_m3,
                dead_end_target_volume_m3: dead_end.target_volume_m3,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        }
        let search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let mut mate_faces = BTreeSet::<[u32; 3]>::new();
        mate_faces.extend(no_candidate_dead_end_faces);
        for future_dead_end in &future_mate_dead_ends {
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
                let candidate_key = sorted_tetrahedron_nodes(candidate.node_ids);
                if !excluded_keys.contains(&candidate_key) && candidate_keys.insert(candidate_key) {
                    candidates.push(candidate.clone());
                    added = true;
                    break;
                }
            }
        }
        if !added {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: if total_attempts > 5_000 {
                    "search_exhausted"
                } else {
                    "cover_not_found"
                },
                dead_end_reason: dead_end.reason,
                dead_end_face: dead_end.face,
                dead_end_depth: dead_end.depth,
                dead_end_selected_tetrahedra: dead_end.selected_tetrahedra,
                dead_end_current_volume_m3: dead_end.current_volume_m3,
                dead_end_candidate_volume_m3: dead_end.candidate_volume_m3,
                dead_end_target_volume_m3: dead_end.target_volume_m3,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        }
    }

    Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
        initial_candidate_count,
        candidate_count: candidates.len(),
        injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
        found_cover: false,
        selected_tetrahedron_count: 0,
        search_attempt_count: total_attempts,
        reason: "iteration_limit",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
        dead_end_faces_by_reason: BTreeMap::new(),
        dead_end_selected_tetrahedra_by_reason: BTreeMap::new(),
        dead_end_selected_roles_by_reason: BTreeMap::new(),
        unforced_found_cover: false,
        unforced_selected_tetrahedron_count: 0,
        unforced_search_attempt_count: 0,
        unforced_dead_end_reason_histogram: BTreeMap::new(),
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundarySteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundarySteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let Some(centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };
    if point_in_closed_triangle_surface(centroid, &boundary_triangles, MeshingTolerance::default())
        != PointInClosedSurface::Inside
    {
        diagnostic.reason = "steiner_point_outside_cavity";
        return Ok(diagnostic);
    }
    let steiner_node_id = next_cavity_node_id(cavity);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    node_points.insert(steiner_node_id, centroid);
    let mut node_ids = boundary_node_ids.clone();
    node_ids.push(steiner_node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_points[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
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
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_patch_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryPatchSteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut diagnostic = BoundaryPatchSteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        missing_face_count: 0,
        patch_count: 0,
        steiner_node_count: 0,
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }

    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    diagnostic.missing_face_count = missing_faces.len();
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    diagnostic.patch_count = components.len();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut node_ids = boundary_node_ids.clone();
    let mut next_node_id = next_cavity_node_id(cavity);
    for component in components {
        let mut patch_node_ids = BTreeSet::<u32>::new();
        for face_index in component {
            patch_node_ids.extend(missing_faces[face_index]);
        }
        let Some(surface_point) = centroid_of_node_set(&patch_node_ids, &boundary_node_map) else {
            continue;
        };
        let Some(point) =
            patch_steiner_point_inside_cavity(surface_point, cavity_centroid, &boundary_triangles)
        else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, point);
        node_ids.push(next_node_id);
        diagnostic.steiner_node_count += 1;
        next_node_id = next_node_id.saturating_add(1);
    }
    if diagnostic.steiner_node_count == 0 {
        diagnostic.reason = "no_valid_patch_steiner_points";
        return Ok(diagnostic);
    }

    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                    let points = tetrahedron_node_ids.map(|node_id| node_points[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
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
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 1_024 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_support_node_exact_cover(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<SupportNodeExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
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
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        return Ok(SupportNodeExactCoverDiagnostic {
            candidate_node_count: candidate_nodes.len(),
            candidate_count: 0,
            root_zero_raw_boundary_face_count: 0,
            root_zero_raw_boundary_faces: Vec::new(),
            root_min_raw_boundary_face_candidate_count: 0,
            root_min_raw_candidate_boundary_faces: Vec::new(),
            root_max_raw_boundary_face_candidate_count: 0,
            root_zero_addable_boundary_face_count: 0,
            root_zero_addable_boundary_faces: Vec::new(),
            root_min_addable_boundary_face_candidate_count: 0,
            root_min_addable_candidate_boundary_faces: Vec::new(),
            root_max_addable_boundary_face_candidate_count: 0,
            selected_tetrahedron_count: 0,
            search_attempt_count: 0,
            found_cover: false,
            reason: "no_candidate_tetrahedra",
            dead_end_reason: "not_evaluated",
            dead_end_face: None,
            dead_end_depth: 0,
            dead_end_reason_histogram: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        });
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let root_raw_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    let root_zero_raw_boundary_faces = boundary_faces
        .iter()
        .zip(root_raw_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect::<Vec<_>>();
    let root_min_raw_boundary_face_candidate_count = root_raw_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let root_min_raw_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(root_raw_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == root_min_raw_boundary_face_candidate_count).then_some(*face)
        })
        .collect::<Vec<_>>();
    let root_max_raw_boundary_face_candidate_count = root_raw_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let root_face_counts = BTreeMap::<[u32; 3], usize>::new();
    let root_selected = Vec::<usize>::new();
    let root_addable_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            search.addable_candidate_count_for_face(*face, &root_face_counts, &root_selected)
        })
        .collect::<Vec<_>>();
    let root_zero_addable_boundary_faces = boundary_faces
        .iter()
        .zip(root_addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect::<Vec<_>>();
    let root_min_addable_boundary_face_candidate_count = root_addable_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let root_min_addable_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(root_addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == root_min_addable_boundary_face_candidate_count).then_some(*face)
        })
        .collect::<Vec<_>>();
    let root_max_addable_boundary_face_candidate_count = root_addable_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let (selected, trace) = search.search_with_trace();
    let dead_end = trace.dead_end.clone();
    let dead_end_faces_by_reason = exact_cover_trace_faces_by_reason(&trace);
    Ok(SupportNodeExactCoverDiagnostic {
        candidate_node_count: candidate_nodes.len(),
        candidate_count: candidates.len(),
        root_zero_raw_boundary_face_count: root_zero_raw_boundary_faces.len(),
        root_zero_raw_boundary_faces,
        root_min_raw_boundary_face_candidate_count,
        root_min_raw_candidate_boundary_faces,
        root_max_raw_boundary_face_candidate_count,
        root_zero_addable_boundary_face_count: root_zero_addable_boundary_faces.len(),
        root_zero_addable_boundary_faces,
        root_min_addable_boundary_face_candidate_count,
        root_min_addable_candidate_boundary_faces,
        root_max_addable_boundary_face_candidate_count,
        selected_tetrahedron_count: selected.as_ref().map(Vec::len).unwrap_or(0),
        search_attempt_count: search.attempts,
        found_cover: selected.is_some(),
        reason: if selected.is_some() {
            "cover_found"
        } else if search.attempts > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        },
        dead_end_reason: dead_end
            .as_ref()
            .map(|dead_end| dead_end.reason)
            .unwrap_or("not_evaluated"),
        dead_end_face: dead_end.as_ref().and_then(|dead_end| dead_end.face),
        dead_end_depth: dead_end.map(|dead_end| dead_end.depth).unwrap_or(0),
        dead_end_reason_histogram: trace.dead_end_reason_counts,
        dead_end_faces_by_reason,
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapQualityDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let mut diagnostic = MissingFaceLocalCapQualityDiagnostic {
        missing_face_count: missing_faces.len(),
        pass_face_count: 0,
        failed_face_count: 0,
        candidate_count: 0,
        candidate_source_bins: BTreeMap::new(),
        max_scaled_jacobian: 0.0,
        max_failed_face_scaled_jacobian: 0.0,
        failed_face_scaled_jacobian_bins: BTreeMap::new(),
        failed_face_source_bins: BTreeMap::new(),
        rejected_by_reason: BTreeMap::new(),
    };
    if missing_faces.is_empty() {
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        return Ok(diagnostic);
    };
    let mut next_node_id = next_cavity_node_id(cavity);
    for face in missing_faces {
        let Some(surface_point) = face_centroid(face, &boundary_node_map) else {
            continue;
        };
        let mut face_passed = false;
        let mut best_failed_face_quality = 0.0_f64;
        let mut best_failed_face_source = None::<&'static str>;
        for apex in
            local_cap_apex_candidates(face, surface_point, cavity_centroid, &boundary_node_map)
        {
            let tetrahedron_points = [
                boundary_node_map[&face[0]],
                boundary_node_map[&face[1]],
                boundary_node_map[&face[2]],
                apex.coordinates_m,
            ];
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(tetrahedron_points),
                &boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                *diagnostic
                    .rejected_by_reason
                    .entry("cap_centroid_outside_cavity")
                    .or_default() += 1;
                continue;
            }
            while boundary_node_map.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            diagnostic.candidate_count += 1;
            *diagnostic
                .candidate_source_bins
                .entry(apex.source)
                .or_default() += 1;
            let exact_scaled_jacobian = tetrahedron_scaled_jacobian(tetrahedron_points);
            match raw_refill_tetrahedron_with_rejection_reason(
                [face[0], face[1], face[2], next_node_id],
                tetrahedron_points,
                options,
            ) {
                Ok(tetrahedron) => {
                    diagnostic.max_scaled_jacobian = diagnostic
                        .max_scaled_jacobian
                        .max(tetrahedron.exact_scaled_jacobian);
                    face_passed = true;
                }
                Err(reason) => {
                    if exact_scaled_jacobian.is_finite() {
                        if exact_scaled_jacobian > best_failed_face_quality {
                            best_failed_face_quality = exact_scaled_jacobian;
                            best_failed_face_source = Some(apex.source);
                        }
                    }
                    *diagnostic.rejected_by_reason.entry(reason).or_default() += 1;
                }
            }
            next_node_id = next_node_id.saturating_add(1);
        }
        diagnostic.pass_face_count += usize::from(face_passed);
        if !face_passed && best_failed_face_quality.is_finite() && best_failed_face_quality > 0.0 {
            diagnostic.failed_face_count += 1;
            diagnostic.max_failed_face_scaled_jacobian = diagnostic
                .max_failed_face_scaled_jacobian
                .max(best_failed_face_quality);
            *diagnostic
                .failed_face_scaled_jacobian_bins
                .entry(diagnostic_scaled_jacobian_bin(best_failed_face_quality))
                .or_default() += 1;
            if let Some(source) = best_failed_face_source {
                *diagnostic
                    .failed_face_source_bins
                    .entry(source)
                    .or_default() += 1;
            }
        }
    }
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_local_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let missing_face_patches = missing_face_components(&missing_faces, MissingFaceLink::Node);
    let mut diagnostic = MissingFaceLocalCapStitchDiagnostic {
        missing_face_count: missing_faces.len(),
        missing_faces: missing_faces.clone(),
        patch_count: missing_face_patches.len(),
        patch_size_histogram: component_size_histogram(
            missing_face_patches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
        ),
        patch_capped_face_count_histogram: BTreeMap::new(),
        incomplete_patch_size_histogram: BTreeMap::new(),
        uncapped_faces: Vec::new(),
        capped_face_count: 0,
        inserted_node_count: 0,
        side_connector_candidate_count: 0,
        candidate_tetrahedron_count: 0,
        cap_side_face_count: 0,
        zero_mate_cap_side_face_count: 0,
        min_cap_side_face_mate_count: 0,
        max_cap_side_face_mate_count: 0,
        open_interior_face_count: 0,
        open_interior_component_count: 0,
        open_interior_component_size_histogram: BTreeMap::new(),
        candidate_with_orphan_interior_face_count: 0,
        candidate_without_orphan_interior_face_count: 0,
        root_boundary_zero_raw_candidate_face_count: 0,
        root_boundary_zero_addable_candidate_face_count: 0,
        root_boundary_min_raw_candidate_count: 0,
        root_boundary_min_addable_candidate_count: 0,
        root_boundary_max_addable_candidate_count: 0,
        cover_dead_end_reason: "not_evaluated",
        cover_dead_end_depth: 0,
        cover_dead_end_reason_histogram: BTreeMap::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_tetrahedra = boundary_refill_tetrahedra;
    let mut inserted_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut next_node_id = next_cavity_node_id(cavity);
    let cap_tetrahedron_start = candidate_tetrahedra.len();
    let mut capped_missing_face_indices = BTreeSet::<usize>::new();
    for (face_index, face) in missing_faces.iter().enumerate() {
        let Some(surface_point) = face_centroid(*face, &boundary_node_map) else {
            continue;
        };
        let Some((coordinates_m, cap_tetrahedron)) = best_local_cap_for_face(
            *face,
            surface_point,
            cavity_centroid,
            next_node_id,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, coordinates_m);
        inserted_nodes.push(ConstrainedCavityNode {
            node_id: next_node_id,
            coordinates_m,
        });
        candidate_tetrahedra.push(cap_tetrahedron);
        diagnostic.capped_face_count += 1;
        capped_missing_face_indices.insert(face_index);
        next_node_id = next_node_id.saturating_add(1);
    }
    for patch in &missing_face_patches {
        let capped_count = patch
            .iter()
            .filter(|face_index| capped_missing_face_indices.contains(face_index))
            .count();
        *diagnostic
            .patch_capped_face_count_histogram
            .entry(capped_count)
            .or_default() += 1;
        if capped_count < patch.len() {
            diagnostic.uncapped_faces.extend(
                patch
                    .iter()
                    .filter(|face_index| !capped_missing_face_indices.contains(face_index))
                    .map(|face_index| missing_faces[*face_index]),
            );
            *diagnostic
                .incomplete_patch_size_histogram
                .entry(patch.len())
                .or_default() += 1;
        }
    }
    diagnostic.inserted_node_count = inserted_nodes.len();
    if diagnostic.capped_face_count < diagnostic.missing_face_count {
        diagnostic.reason = "incomplete_local_caps";
        diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
        return Ok(diagnostic);
    }
    let cap_tetrahedron_count = candidate_tetrahedra.len() - cap_tetrahedron_start;

    let connector_points = node_points
        .iter()
        .map(|(node_id, coordinates_m)| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: *coordinates_m,
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut seen_tetrahedra = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();
    for tetrahedron in tetrahedralize_points(&connector_points) {
        let node_ids = tetrahedron
            .vertices
            .map(|index| connector_points[index].node_id);
        if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(node_ids)) {
            continue;
        }
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| connector_points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            candidate_tetrahedra.push(tetrahedron);
        }
    }
    diagnostic.side_connector_candidate_count = append_cap_side_connector_tetrahedra(
        cap_tetrahedron_start,
        cap_tetrahedron_count,
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &node_points,
        &inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<BTreeSet<_>>(),
        &boundary_triangles,
        options,
    );
    diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
    let cap_side_mate_counts = cap_side_face_mate_counts(
        &candidate_tetrahedra[cap_tetrahedron_start..cap_tetrahedron_start + cap_tetrahedron_count],
        &candidate_tetrahedra,
        &inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<BTreeSet<_>>(),
    );
    diagnostic.cap_side_face_count = cap_side_mate_counts.len();
    diagnostic.zero_mate_cap_side_face_count = cap_side_mate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().max().unwrap_or(0);
    let open_interior_faces = open_interior_refill_faces(cavity, &candidate_tetrahedra);
    diagnostic.open_interior_face_count = open_interior_faces.len();
    diagnostic.open_interior_component_count =
        missing_face_components(&open_interior_faces, MissingFaceLink::Node).len();
    diagnostic.open_interior_component_size_histogram = component_size_histogram(
        missing_face_component_sizes(&open_interior_faces, MissingFaceLink::Node),
    );
    let (with_orphan, without_orphan) =
        candidate_orphan_interior_face_counts(cavity, &candidate_tetrahedra);
    diagnostic.candidate_with_orphan_interior_face_count = with_orphan;
    diagnostic.candidate_without_orphan_interior_face_count = without_orphan;
    if candidate_tetrahedra.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidate_tetrahedra.len() > 4_096 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        &candidate_tetrahedra,
        options.volume_relative_tolerance,
        25_000,
    );
    let root_availability = search.root_boundary_availability();
    diagnostic.root_boundary_zero_raw_candidate_face_count =
        root_availability.zero_raw_candidate_face_count;
    diagnostic.root_boundary_zero_addable_candidate_face_count =
        root_availability.zero_addable_candidate_face_count;
    diagnostic.root_boundary_min_raw_candidate_count = root_availability.min_raw_candidate_count;
    diagnostic.root_boundary_min_addable_candidate_count =
        root_availability.min_addable_candidate_count;
    diagnostic.root_boundary_max_addable_candidate_count =
        root_availability.max_addable_candidate_count;
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.cover_dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.cover_dead_end_reason = dead_end.reason;
        diagnostic.cover_dead_end_depth = dead_end.depth;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 25_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidate_tetrahedra[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_shared_patch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Node,
        "incomplete_shared_patch_caps",
        false,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_edge_subpatch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Edge,
        "incomplete_edge_subpatch_caps",
        false,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_missing_face_hybrid_subpatch_cap_stitch(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_missing_face_shared_cap_stitch_with_link(
        cavity,
        boundary_nodes,
        options,
        MissingFaceLink::Edge,
        "incomplete_hybrid_subpatch_caps",
        true,
    )
}

#[cfg(test)]
fn diagnostic_missing_face_shared_cap_stitch_with_link(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
    patch_link: MissingFaceLink,
    incomplete_reason: &'static str,
    fallback_to_face_caps: bool,
) -> Result<MissingFaceLocalCapStitchDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let missing_face_patches = missing_face_components(&missing_faces, patch_link);
    let mut diagnostic = MissingFaceLocalCapStitchDiagnostic {
        missing_face_count: missing_faces.len(),
        missing_faces: missing_faces.clone(),
        patch_count: missing_face_patches.len(),
        patch_size_histogram: component_size_histogram(
            missing_face_patches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
        ),
        patch_capped_face_count_histogram: BTreeMap::new(),
        incomplete_patch_size_histogram: BTreeMap::new(),
        uncapped_faces: Vec::new(),
        capped_face_count: 0,
        inserted_node_count: 0,
        side_connector_candidate_count: 0,
        candidate_tetrahedron_count: 0,
        cap_side_face_count: 0,
        zero_mate_cap_side_face_count: 0,
        min_cap_side_face_mate_count: 0,
        max_cap_side_face_mate_count: 0,
        open_interior_face_count: 0,
        open_interior_component_count: 0,
        open_interior_component_size_histogram: BTreeMap::new(),
        candidate_with_orphan_interior_face_count: 0,
        candidate_without_orphan_interior_face_count: 0,
        root_boundary_zero_raw_candidate_face_count: 0,
        root_boundary_zero_addable_candidate_face_count: 0,
        root_boundary_min_raw_candidate_count: 0,
        root_boundary_min_addable_candidate_count: 0,
        root_boundary_max_addable_candidate_count: 0,
        cover_dead_end_reason: "not_evaluated",
        cover_dead_end_depth: 0,
        cover_dead_end_reason_histogram: BTreeMap::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_tetrahedra = boundary_refill_tetrahedra;
    let mut inserted_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut next_node_id = next_cavity_node_id(cavity);
    let cap_tetrahedron_start = candidate_tetrahedra.len();
    for patch in &missing_face_patches {
        let faces = patch
            .iter()
            .map(|face_index| missing_faces[*face_index])
            .collect::<Vec<_>>();
        if let Some((coordinates_m, mut cap_tetrahedra)) = best_shared_patch_cap_for_faces(
            &faces,
            cavity_centroid,
            next_node_id,
            &boundary_node_map,
            &boundary_triangles,
            options,
        ) {
            while node_points.contains_key(&next_node_id) {
                next_node_id = next_node_id.saturating_add(1);
            }
            node_points.insert(next_node_id, coordinates_m);
            inserted_nodes.push(ConstrainedCavityNode {
                node_id: next_node_id,
                coordinates_m,
            });
            diagnostic.capped_face_count += cap_tetrahedra.len();
            *diagnostic
                .patch_capped_face_count_histogram
                .entry(cap_tetrahedra.len())
                .or_default() += 1;
            candidate_tetrahedra.append(&mut cap_tetrahedra);
            next_node_id = next_node_id.saturating_add(1);
            continue;
        }

        let mut capped_count = 0_usize;
        if fallback_to_face_caps {
            for face in &faces {
                let Some(surface_point) = face_centroid(*face, &boundary_node_map) else {
                    continue;
                };
                while node_points.contains_key(&next_node_id) {
                    next_node_id = next_node_id.saturating_add(1);
                }
                let Some((coordinates_m, cap_tetrahedron)) = best_local_cap_for_face(
                    *face,
                    surface_point,
                    cavity_centroid,
                    next_node_id,
                    &boundary_node_map,
                    &boundary_triangles,
                    options,
                ) else {
                    continue;
                };
                node_points.insert(next_node_id, coordinates_m);
                inserted_nodes.push(ConstrainedCavityNode {
                    node_id: next_node_id,
                    coordinates_m,
                });
                candidate_tetrahedra.push(cap_tetrahedron);
                capped_count += 1;
                next_node_id = next_node_id.saturating_add(1);
            }
            diagnostic.capped_face_count += capped_count;
        }
        *diagnostic
            .patch_capped_face_count_histogram
            .entry(capped_count)
            .or_default() += 1;
        if capped_count < patch.len() {
            diagnostic.uncapped_faces.extend(
                patch
                    .iter()
                    .filter(|face_index| {
                        let face = missing_faces[**face_index];
                        !candidate_tetrahedra[cap_tetrahedron_start..]
                            .iter()
                            .any(|tetrahedron| {
                                tetrahedron_faces(tetrahedron.node_ids)
                                    .map(sorted_face)
                                    .contains(&face)
                            })
                    })
                    .map(|face_index| missing_faces[*face_index]),
            );
            *diagnostic
                .incomplete_patch_size_histogram
                .entry(patch.len())
                .or_default() += 1;
        }
    }
    diagnostic.inserted_node_count = inserted_nodes.len();
    if diagnostic.capped_face_count < diagnostic.missing_face_count {
        diagnostic.reason = incomplete_reason;
        diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
        return Ok(diagnostic);
    }
    let cap_tetrahedron_count = candidate_tetrahedra.len() - cap_tetrahedron_start;

    let connector_points = node_points
        .iter()
        .map(|(node_id, coordinates_m)| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: *coordinates_m,
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut seen_tetrahedra = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();
    for tetrahedron in tetrahedralize_points(&connector_points) {
        let node_ids = tetrahedron
            .vertices
            .map(|index| connector_points[index].node_id);
        if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(node_ids)) {
            continue;
        }
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| connector_points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            candidate_tetrahedra.push(tetrahedron);
        }
    }
    let inserted_node_ids = inserted_nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    diagnostic.side_connector_candidate_count = append_cap_side_connector_tetrahedra(
        cap_tetrahedron_start,
        cap_tetrahedron_count,
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &node_points,
        &inserted_node_ids,
        &boundary_triangles,
        options,
    );
    diagnostic.candidate_tetrahedron_count = candidate_tetrahedra.len();
    let cap_side_mate_counts = cap_side_face_mate_counts(
        &candidate_tetrahedra[cap_tetrahedron_start..cap_tetrahedron_start + cap_tetrahedron_count],
        &candidate_tetrahedra,
        &inserted_node_ids,
    );
    diagnostic.cap_side_face_count = cap_side_mate_counts.len();
    diagnostic.zero_mate_cap_side_face_count = cap_side_mate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_cap_side_face_mate_count =
        cap_side_mate_counts.iter().copied().max().unwrap_or(0);
    let open_interior_faces = open_interior_refill_faces(cavity, &candidate_tetrahedra);
    diagnostic.open_interior_face_count = open_interior_faces.len();
    diagnostic.open_interior_component_count =
        missing_face_components(&open_interior_faces, MissingFaceLink::Node).len();
    diagnostic.open_interior_component_size_histogram = component_size_histogram(
        missing_face_component_sizes(&open_interior_faces, MissingFaceLink::Node),
    );
    let (with_orphan, without_orphan) =
        candidate_orphan_interior_face_counts(cavity, &candidate_tetrahedra);
    diagnostic.candidate_with_orphan_interior_face_count = with_orphan;
    diagnostic.candidate_without_orphan_interior_face_count = without_orphan;
    if candidate_tetrahedra.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidate_tetrahedra.len() > 4_096 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        &candidate_tetrahedra,
        options.volume_relative_tolerance,
        25_000,
    );
    let root_availability = search.root_boundary_availability();
    diagnostic.root_boundary_zero_raw_candidate_face_count =
        root_availability.zero_raw_candidate_face_count;
    diagnostic.root_boundary_zero_addable_candidate_face_count =
        root_availability.zero_addable_candidate_face_count;
    diagnostic.root_boundary_min_raw_candidate_count = root_availability.min_raw_candidate_count;
    diagnostic.root_boundary_min_addable_candidate_count =
        root_availability.min_addable_candidate_count;
    diagnostic.root_boundary_max_addable_candidate_count =
        root_availability.max_addable_candidate_count;
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.cover_dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.cover_dead_end_reason = dead_end.reason;
        diagnostic.cover_dead_end_depth = dead_end.depth;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 25_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidate_tetrahedra[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_missing_face_clusters(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryMissingFaceClusterDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let points = cavity_boundary_node_ids(cavity)
        .into_iter()
        .map(|node_id| ConnectivityPoint {
            node_id,
            coordinates_m: boundary_node_map[&node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    let edge_component_sizes = missing_face_component_sizes(&missing_faces, MissingFaceLink::Edge);
    let node_components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    let node_component_sizes = node_components.iter().map(Vec::len).collect::<Vec<_>>();
    let mut node_component_common_node_count_histogram = BTreeMap::<usize, usize>::new();
    let mut node_component_common_node_ids = BTreeMap::<u32, usize>::new();
    for component in &node_components {
        let common_node_ids = missing_face_component_common_node_ids(&missing_faces, component);
        *node_component_common_node_count_histogram
            .entry(common_node_ids.len())
            .or_default() += 1;
        for node_id in common_node_ids {
            *node_component_common_node_ids.entry(node_id).or_default() += 1;
        }
    }
    Ok(BoundaryMissingFaceClusterDiagnostic {
        missing_face_count: missing_faces.len(),
        edge_component_count: edge_component_sizes.len(),
        edge_component_size_histogram: component_size_histogram(edge_component_sizes),
        node_component_count: node_component_sizes.len(),
        node_component_size_histogram: component_size_histogram(node_component_sizes),
        node_component_common_node_count_histogram,
        node_component_common_node_ids,
    })
}

fn star_refill_candidate_with_rejection_reason(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    interior_node: ConstrainedCavityNode,
    options: ConstrainedCavityRefillOptions,
) -> Result<Result<ConstrainedCavityRefill, &'static str>, ConstrainedCavityValidationError> {
    let mut tetrahedra =
        Vec::<ConstrainedCavityRefillTetrahedron>::with_capacity(cavity.boundary_faces.len());
    for face in &cavity.boundary_faces {
        let node_ids = [
            face.node_ids[0],
            face.node_ids[1],
            face.node_ids[2],
            interior_node.node_id,
        ];
        let points = [
            boundary_nodes[&face.node_ids[0]],
            boundary_nodes[&face.node_ids[1]],
            boundary_nodes[&face.node_ids[2]],
            interior_node.coordinates_m,
        ];
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(reason) => return Ok(Err(reason)),
            };
        tetrahedra.push(tetrahedron);
    }
    let refill = refill_from_tetrahedra(cavity, tetrahedra, options.volume_relative_tolerance)?;
    Ok(Ok(refill))
}

fn raw_refill_tetrahedron(
    node_ids: [u32; 4],
    points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefillTetrahedron> {
    raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options).ok()
}

fn refill_component_node_map(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    nodes: &[ConstrainedCavityNode],
) -> Result<BTreeMap<u32, Point3>, ConstrainedCavityRefillTetrahedronFlipError> {
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for tetrahedron in tetrahedra {
        for node_id in tetrahedron.node_ids {
            if !node_map.contains_key(&node_id) {
                return Err(ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id });
            }
        }
    }
    Ok(node_map)
}

fn refill_tetrahedra_from_flip_candidate(
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    node_map: &BTreeMap<u32, Point3>,
    flip: &LocalTetrahedronFlipCandidate,
    options: ConstrainedCavityRefillOptions,
) -> Result<Vec<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityRefillTetrahedronFlipError> {
    let removed_indices = flip
        .removed_tetrahedron_ids
        .iter()
        .map(|tetrahedron_id| *tetrahedron_id as usize)
        .collect::<BTreeSet<_>>();
    if removed_indices
        .iter()
        .any(|index| *index >= tetrahedra.len())
    {
        return Err(
            ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
                reason: "removed_tetrahedron_out_of_bounds",
            },
        );
    }
    let mut candidate_tetrahedra = tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            (!removed_indices.contains(&index)).then_some(tetrahedron.clone())
        })
        .collect::<Vec<_>>();
    let mut candidate_keys = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();
    for node_ids in &flip.created_tetrahedra {
        let key = sorted_tetrahedron_nodes(*node_ids);
        if !candidate_keys.insert(key) {
            return Err(
                ConstrainedCavityRefillTetrahedronFlipError::InvalidFlipTopology {
                    reason: "duplicate_created_tetrahedron",
                },
            );
        }
        let mut points = [[0.0; 3]; 4];
        for (point, node_id) in points.iter_mut().zip(node_ids) {
            *point = *node_map.get(node_id).ok_or(
                ConstrainedCavityRefillTetrahedronFlipError::MissingNode { node_id: *node_id },
            )?;
        }
        match raw_refill_tetrahedron_with_rejection_reason(*node_ids, points, options) {
            Ok(tetrahedron) => candidate_tetrahedra.push(tetrahedron),
            Err(reason) => {
                return Err(
                    ConstrainedCavityRefillTetrahedronFlipError::RejectedCreatedTetrahedron {
                        node_ids: *node_ids,
                        reason,
                    },
                );
            }
        }
    }
    Ok(candidate_tetrahedra)
}

fn raw_refill_tetrahedron_with_rejection_reason(
    mut node_ids: [u32; 4],
    points: [Point3; 4],
    options: ConstrainedCavityRefillOptions,
) -> Result<ConstrainedCavityRefillTetrahedron, &'static str> {
    let mut signed_volume_m3 = tetrahedron_signed_volume(points);
    if signed_volume_m3 < 0.0 {
        node_ids.swap(1, 2);
        signed_volume_m3 = -signed_volume_m3;
    }
    let volume_m3 = signed_volume_m3.abs();
    if volume_m3 < options.min_volume_m3 {
        return Err("star_tetrahedron_min_volume");
    }
    let aspect_ratio = tetrahedron_edge_aspect_ratio(points);
    if !aspect_ratio.is_finite() || aspect_ratio > options.max_aspect_ratio {
        return Err("star_tetrahedron_aspect_ratio");
    }
    let exact_scaled_jacobian = tetrahedron_scaled_jacobian(points);
    if !exact_scaled_jacobian.is_finite() || exact_scaled_jacobian < options.min_scaled_jacobian {
        return Err("star_tetrahedron_scaled_jacobian");
    }
    Ok(ConstrainedCavityRefillTetrahedron {
        node_ids,
        volume_m3,
        aspect_ratio,
        exact_scaled_jacobian,
    })
}

fn local_tetrahedron_flip_error_reason(error: &LocalTetrahedronFlipError) -> &'static str {
    match error {
        LocalTetrahedronFlipError::DegenerateTetrahedron { .. } => "degenerate_tetrahedron",
        LocalTetrahedronFlipError::NoSharedFace => "no_shared_face",
        LocalTetrahedronFlipError::NoSharedEdge => "no_shared_edge",
        LocalTetrahedronFlipError::InvalidEdgeRing => "invalid_edge_ring",
        LocalTetrahedronFlipError::InvalidQualityThresholds => "invalid_quality_thresholds",
        LocalTetrahedronFlipError::MissingNode { .. } => "missing_node",
        LocalTetrahedronFlipError::NonPositiveVolume { .. } => "non_positive_volume",
        LocalTetrahedronFlipError::VolumeBelowThreshold { .. } => "volume_below_threshold",
        LocalTetrahedronFlipError::ScaledJacobianBelowThreshold { .. } => {
            "scaled_jacobian_below_threshold"
        }
    }
}

fn refill_from_tetrahedra(
    cavity: &ConstrainedCavity,
    tetrahedra: Vec<ConstrainedCavityRefillTetrahedron>,
    volume_relative_tolerance: f64,
) -> Result<ConstrainedCavityRefill, ConstrainedCavityValidationError> {
    let boundary_faces = boundary_faces_from_refill_tetrahedra(cavity, &tetrahedra)?;
    validate_constrained_cavity_boundary_preserved(cavity, &boundary_faces)?;
    let total_volume_m3 = tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.volume_m3)
        .sum::<f64>();
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        total_volume_m3,
        volume_relative_tolerance,
    )?;
    Ok(ConstrainedCavityRefill {
        tetrahedra,
        boundary_faces,
        inserted_nodes: Vec::new(),
        total_volume_m3,
    })
}

fn boundary_faces_from_refill_tetrahedra(
    cavity: &ConstrainedCavity,
    tetrahedra: &[ConstrainedCavityRefillTetrahedron],
) -> Result<Vec<ConstrainedCavityBoundaryFace>, ConstrainedCavityValidationError> {
    let cavity_faces = boundary_face_map(&cavity.boundary_faces)?;
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
                cavity_faces
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
    Ok(boundary_faces)
}

fn improve_refill_with_local_flips(
    cavity: &ConstrainedCavity,
    node_coordinates: &BTreeMap<u32, Point3>,
    refill: &ConstrainedCavityRefill,
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefill> {
    if refill.tetrahedra.len() < 2 {
        return None;
    }
    let mut coordinates = node_coordinates.clone();
    for node in &refill.inserted_nodes {
        coordinates.insert(node.node_id, node.coordinates_m);
    }
    let thresholds = LocalTetrahedronFlipQualityThresholds {
        min_volume_m3: options.min_volume_m3,
        min_scaled_jacobian: options.min_scaled_jacobian,
    };
    let mut best = None::<ConstrainedCavityRefill>;

    for left_index in 0..refill.tetrahedra.len() {
        for right_index in (left_index + 1)..refill.tetrahedra.len() {
            let left = LocalTetrahedron {
                tetrahedron_id: left_index as u32,
                node_ids: refill.tetrahedra[left_index].node_ids,
            };
            let right = LocalTetrahedron {
                tetrahedron_id: right_index as u32,
                node_ids: refill.tetrahedra[right_index].node_ids,
            };
            let Ok(flip) = two_to_three_face_flip_candidate(left, right) else {
                continue;
            };
            if evaluate_local_tetrahedron_flip_quality(&flip, &coordinates, thresholds).is_err() {
                continue;
            }

            let Some(candidate) =
                refill_from_local_flip_candidate(cavity, &coordinates, refill, &flip, options)
            else {
                continue;
            };
            if !refill_is_better(&candidate, refill) {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|current| refill_is_better(&candidate, current))
            {
                best = Some(candidate);
            }
        }
    }

    for left_index in 0..refill.tetrahedra.len() {
        for middle_index in (left_index + 1)..refill.tetrahedra.len() {
            for right_index in (middle_index + 1)..refill.tetrahedra.len() {
                let tetrahedra = [
                    LocalTetrahedron {
                        tetrahedron_id: left_index as u32,
                        node_ids: refill.tetrahedra[left_index].node_ids,
                    },
                    LocalTetrahedron {
                        tetrahedron_id: middle_index as u32,
                        node_ids: refill.tetrahedra[middle_index].node_ids,
                    },
                    LocalTetrahedron {
                        tetrahedron_id: right_index as u32,
                        node_ids: refill.tetrahedra[right_index].node_ids,
                    },
                ];
                for edge in
                    common_tetrahedron_edges(tetrahedra.map(|tetrahedron| tetrahedron.node_ids))
                {
                    let Ok(flip) = three_to_two_edge_flip_candidate(tetrahedra, edge) else {
                        continue;
                    };
                    if evaluate_local_tetrahedron_flip_quality(&flip, &coordinates, thresholds)
                        .is_err()
                    {
                        continue;
                    }
                    let Some(candidate) = refill_from_local_flip_candidate(
                        cavity,
                        &coordinates,
                        refill,
                        &flip,
                        options,
                    ) else {
                        continue;
                    };
                    if !refill_is_better(&candidate, refill) {
                        continue;
                    }
                    if best
                        .as_ref()
                        .is_none_or(|current| refill_is_better(&candidate, current))
                    {
                        best = Some(candidate);
                    }
                }
            }
        }
    }

    best
}

fn refill_from_local_flip_candidate(
    cavity: &ConstrainedCavity,
    coordinates: &BTreeMap<u32, Point3>,
    refill: &ConstrainedCavityRefill,
    flip: &LocalTetrahedronFlipCandidate,
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefill> {
    let removed_indices = flip
        .removed_tetrahedron_ids
        .iter()
        .map(|tetrahedron_id| *tetrahedron_id as usize)
        .collect::<BTreeSet<_>>();
    if removed_indices
        .iter()
        .any(|index| *index >= refill.tetrahedra.len())
    {
        return None;
    }
    let mut candidate_tetrahedra = refill
        .tetrahedra
        .iter()
        .enumerate()
        .filter_map(|(index, tetrahedron)| {
            (!removed_indices.contains(&index)).then_some(tetrahedron.clone())
        })
        .collect::<Vec<_>>();
    let mut created_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut created_keys = BTreeSet::<[u32; 4]>::new();
    for node_ids in &flip.created_tetrahedra {
        let key = sorted_tetrahedron_nodes(*node_ids);
        if !created_keys.insert(key)
            || candidate_tetrahedra
                .iter()
                .any(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids) == key)
        {
            return None;
        }
        let mut points = [[0.0; 3]; 4];
        for (point, node_id) in points.iter_mut().zip(node_ids) {
            *point = *coordinates.get(node_id)?;
        }
        let tetrahedron =
            raw_refill_tetrahedron_with_rejection_reason(*node_ids, points, options).ok()?;
        created_tetrahedra.push(tetrahedron);
    }
    candidate_tetrahedra.extend(created_tetrahedra);

    let mut candidate = refill_from_tetrahedra(
        cavity,
        candidate_tetrahedra,
        options.volume_relative_tolerance,
    )
    .ok()?;
    candidate.inserted_nodes = refill.inserted_nodes.clone();
    Some(candidate)
}

fn refill_is_better(
    candidate: &ConstrainedCavityRefill,
    current: &ConstrainedCavityRefill,
) -> bool {
    let candidate_min = candidate
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let current_min = current
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    candidate_min > current_min + 1.0e-12
        || ((candidate_min - current_min).abs() <= 1.0e-12
            && candidate.tetrahedra.len() < current.tetrahedra.len())
}

fn record_refill_rejection(rejected_by_reason: &mut BTreeMap<String, usize>, reason: &str) {
    *rejected_by_reason.entry(reason.to_string()).or_default() += 1;
}

fn refill_validation_reason(error: &ConstrainedCavityValidationError) -> &'static str {
    match error {
        ConstrainedCavityValidationError::InvalidRefillVolume { .. } => "volume_mismatch",
        ConstrainedCavityValidationError::BoundaryFaceCountMismatch { .. } => {
            "boundary_face_count_mismatch"
        }
        ConstrainedCavityValidationError::MissingBoundaryFace { .. } => "missing_boundary_face",
        ConstrainedCavityValidationError::UnexpectedBoundaryFace { .. } => {
            "unexpected_boundary_face"
        }
        ConstrainedCavityValidationError::BoundarySourceFaceMismatch { .. } => {
            "boundary_source_face_mismatch"
        }
        ConstrainedCavityValidationError::BoundarySourceEdgeMismatch { .. } => {
            "boundary_source_edge_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryRegionMismatch { .. } => {
            "boundary_region_mismatch"
        }
        ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch { .. } => {
            "boundary_outside_tetrahedron_mismatch"
        }
        ConstrainedCavityValidationError::EmptyRemovedTetrahedronSet
        | ConstrainedCavityValidationError::InvalidTargetVolume { .. }
        | ConstrainedCavityValidationError::TooFewBoundaryFaces { .. }
        | ConstrainedCavityValidationError::DegenerateBoundaryFace { .. }
        | ConstrainedCavityValidationError::DuplicateBoundaryFace { .. }
        | ConstrainedCavityValidationError::NonManifoldBoundaryEdge { .. }
        | ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { .. } => "invalid_cavity",
    }
}

#[cfg(test)]
mod tests;
