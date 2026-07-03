#![cfg_attr(test, allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    predicate::{
        distance_squared, orient_tetrahedron_node_ids, point_in_closed_triangle_surface,
        tetrahedron_centroid, tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, Point3,
        PointInClosedSurface, Triangle3,
    },
    tolerance::MeshingTolerance,
};

mod boundary_completion;
mod boundary_nodes;
mod boundary_operations;
mod boundary_splits;
mod cap_connectors;
mod caps;
mod component_steiner;
mod connectivity;
#[cfg(test)]
mod diagnostic_metrics;
#[cfg(test)]
mod diagnostics;
mod exact_cover;
mod geometry;
mod missing_faces;
mod refill_candidates;
mod refill_faces;
mod refill_tetrahedra;
mod selection;
mod solid_empty;
mod topology;
mod types;
mod validation;

use boundary_completion::*;
use boundary_nodes::{
    boundary_node_coordinates, candidate_respects_protected_boundary_distance,
    cavity_boundary_node_centroid, cavity_boundary_node_ids, cavity_boundary_triangles,
    next_cavity_node_id,
};
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
#[cfg(test)]
use diagnostics::*;
use exact_cover::*;
pub use exact_cover::{
    selected_exact_cover_face_count_blockers, selected_exact_cover_saturated_component,
};
use geometry::*;
use missing_faces::*;
use refill_candidates::{
    boundary_node_refill_candidate, centroid_interior_refill_candidate,
    multi_interior_node_refill_candidate, single_tetrahedron_refill_candidate,
    two_interior_node_refill_candidate,
};
#[cfg(test)]
use refill_candidates::{
    boundary_node_refill_rejection_reason, boundary_node_refill_validation_reason,
    multi_interior_exact_cover_failure_reason,
};
use refill_faces::*;
use refill_tetrahedra::{
    boundary_faces_from_refill_tetrahedra, raw_refill_tetrahedron,
    raw_refill_tetrahedron_with_rejection_reason, record_refill_rejection, refill_from_tetrahedra,
    refill_is_better, refill_validation_reason, star_refill_candidate_with_rejection_reason,
};
pub use refill_tetrahedra::{
    flip_refill_tetrahedra_across_shared_face, flip_refill_tetrahedra_around_shared_edge,
    split_refill_tetrahedra_across_shared_face_at_barycentric,
};
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

#[cfg(test)]
mod tests;
