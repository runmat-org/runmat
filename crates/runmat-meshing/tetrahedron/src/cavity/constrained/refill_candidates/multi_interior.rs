use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid, Point3,
        PointInClosedSurface, Triangle3,
    },
    quality::tolerance::MeshingTolerance,
};

#[cfg(test)]
use super::super::BoundaryExactCoverSearch;
use super::super::{
    cavity_boundary_node_centroid, cavity_boundary_node_ids,
    exact_cover_refill_from_candidate_tetrahedra, raw_refill_tetrahedron_with_rejection_reason,
    tetrahedralize_points, topology::sorted_tetrahedron_nodes, ConnectivityPoint,
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefill,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError, MAX_MULTI_INTERIOR_REFILL_CANDIDATES,
    MAX_MULTI_INTERIOR_REFILL_NODES,
};
use super::boundary_node_refill_rejection_reason;

pub(in super::super) fn multi_interior_node_refill_candidate(
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
pub(in super::super) fn multi_interior_exact_cover_failure_reason(
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
pub(in super::super) fn multi_interior_exact_cover_failure_reason(
    _cavity: &ConstrainedCavity,
    _candidates: &[ConstrainedCavityRefillTetrahedron],
    _options: ConstrainedCavityRefillOptions,
) -> &'static str {
    "multi_interior_exact_cover_not_found"
}
