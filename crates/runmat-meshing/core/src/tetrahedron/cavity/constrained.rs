#![cfg_attr(test, allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    predicate::{
        distance_squared, orient_tetrahedron_node_ids, point_in_closed_triangle_surface,
        tetrahedron_centroid, tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian,
        tetrahedron_signed_volume, Point3, PointInClosedSurface, Triangle3,
    },
    tetrahedron::reconnect::{
        evaluate_local_tetrahedron_flip_quality, three_to_two_edge_flip_candidate,
        two_to_three_face_flip_candidate, LocalTetrahedron, LocalTetrahedronFlipCandidate,
        LocalTetrahedronFlipError, LocalTetrahedronFlipQualityThresholds,
    },
    tolerance::MeshingTolerance,
};

mod boundary_completion;
mod boundary_operations;
mod boundary_splits;
mod cap_connectors;
mod caps;
mod component_steiner;
mod connectivity;
#[cfg(test)]
mod diagnostic_metrics;
mod exact_cover;
mod geometry;
mod missing_faces;
mod refill_faces;
mod selection;
mod solid_empty;
mod topology;
mod types;
mod validation;

use boundary_completion::*;
use boundary_operations::*;
pub use boundary_operations::{
    split_constrained_cavity_boundary_edge,
    split_constrained_cavity_boundary_edge_patch_at_centroid,
    split_constrained_cavity_boundary_face, split_constrained_cavity_boundary_face_at_barycentric,
    split_constrained_cavity_boundary_face_at_centroid, split_constrained_cavity_boundary_faces,
    split_constrained_cavity_boundary_faces_at_centroids,
    split_constrained_cavity_boundary_patch_at_centroids, split_constrained_cavity_source_edge,
};
use boundary_splits::*;
use cap_connectors::*;
use caps::*;
use component_steiner::*;
use connectivity::*;
#[cfg(test)]
use diagnostic_metrics::*;
use exact_cover::*;
pub use exact_cover::{
    selected_exact_cover_face_count_blockers, selected_exact_cover_saturated_component,
};
use geometry::*;
use missing_faces::*;
use refill_faces::*;
use selection::*;
pub use selection::{
    constrained_cavity_expanded_across_boundary_edge_star_excluding_nodes,
    constrained_cavity_expanded_across_boundary_face,
    constrained_cavity_expanded_across_boundary_faces,
    constrained_cavity_expanded_across_boundary_faces_or_recovered_edge_star,
    constrained_cavity_expanded_across_first_valid_boundary_face,
    constrained_cavity_from_refill_tetrahedron_component,
    constrained_cavity_from_selected_tetrahedra,
    constrained_cavity_from_selected_tetrahedra_with_anchor_trim,
    constrained_cavity_recovered_boundary_edge_star_excluding_nodes,
    constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes,
};
use solid_empty::*;
pub use solid_empty::{
    constrained_cavity_classified_solid_empty_boundary_faces,
    constrained_cavity_solid_empty_boundary_faces,
    recover_constrained_cavity_solid_empty_boundaries,
};
use topology::*;
pub use types::*;
use validation::*;
pub use validation::{
    validate_constrained_cavity, validate_constrained_cavity_boundary_preserved,
    validate_constrained_cavity_refill_volume,
};

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
