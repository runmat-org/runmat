#![cfg_attr(test, allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
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
mod pressure;
mod refill_candidates;
mod refill_faces;
mod refill_tetrahedra;
mod selection;
mod solid_empty;
mod topology;
mod types;
mod validation;

#[cfg(test)]
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
#[cfg(test)]
use caps::*;
pub use caps::{
    generate_constrained_cavity_boundary_cap_nodes, generate_constrained_cavity_patch_steiner_nodes,
};
pub use component_steiner::generate_constrained_cavity_component_steiner_nodes;
use connectivity::*;
#[cfg(test)]
use diagnostic_metrics::*;
#[cfg(test)]
use diagnostics::*;
use exact_cover::*;
pub use exact_cover::{
    selected_exact_cover_face_count_blockers, selected_exact_cover_saturated_component,
};
#[cfg(test)]
use geometry::*;
#[cfg(test)]
use missing_faces::*;
pub use pressure::constrained_cavity_refill_pressure_boundary_faces;
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
#[cfg(test)]
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

#[cfg(test)]
mod tests;
