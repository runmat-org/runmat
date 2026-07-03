use std::collections::{BTreeMap, BTreeSet};

use crate::{
    predicate::{
        distance_squared, point_in_closed_triangle_surface, tetrahedron_centroid, Point3,
        PointInClosedSurface, Triangle3,
    },
    tolerance::MeshingTolerance,
};

#[cfg(test)]
use super::BoundaryExactCoverSearch;
use super::{
    boundary_node_exact_cover_refill_candidate, cavity_boundary_node_centroid,
    cavity_boundary_node_ids, complete_missing_boundary_face_tetrahedra,
    exact_cover_refill_from_candidate_tetrahedra, improve_refill_with_local_flips,
    next_cavity_node_id, raw_refill_tetrahedron, raw_refill_tetrahedron_with_rejection_reason,
    refill_from_tetrahedra, refill_is_better, refill_validation_reason,
    star_refill_candidate_with_rejection_reason, tetrahedralize_points,
    topology::sorted_tetrahedron_nodes, ConnectivityPoint, ConstrainedCavity,
    ConstrainedCavityNode, ConstrainedCavityRefill, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron, ConstrainedCavityValidationError,
    MAX_MULTI_INTERIOR_REFILL_CANDIDATES, MAX_MULTI_INTERIOR_REFILL_NODES,
};

pub(super) fn single_tetrahedron_refill_candidate(
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

pub(super) fn boundary_node_refill_candidate(
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

pub(super) fn boundary_node_refill_rejection_reason(reason: &'static str) -> &'static str {
    match reason {
        "star_tetrahedron_min_volume" => "boundary_node_tetrahedron_min_volume",
        "star_tetrahedron_aspect_ratio" => "boundary_node_tetrahedron_aspect_ratio",
        "star_tetrahedron_scaled_jacobian" => "boundary_node_tetrahedron_scaled_jacobian",
        other => other,
    }
}

pub(super) fn boundary_node_refill_validation_reason(
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

pub(super) fn centroid_interior_refill_candidate(
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

pub(super) fn two_interior_node_refill_candidate(
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

pub(super) fn multi_interior_node_refill_candidate(
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
pub(super) fn multi_interior_exact_cover_failure_reason(
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
pub(super) fn multi_interior_exact_cover_failure_reason(
    _cavity: &ConstrainedCavity,
    _candidates: &[ConstrainedCavityRefillTetrahedron],
    _options: ConstrainedCavityRefillOptions,
) -> &'static str {
    "multi_interior_exact_cover_not_found"
}
