use std::collections::BTreeMap;

use crate::{
    predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    tolerance::MeshingTolerance,
};

mod multi_interior;
#[cfg(test)]
pub(super) use multi_interior::multi_interior_exact_cover_failure_reason;
pub(super) use multi_interior::multi_interior_node_refill_candidate;

use super::{
    boundary_node_exact_cover_refill_candidate, cavity_boundary_node_centroid,
    cavity_boundary_node_ids, complete_missing_boundary_face_tetrahedra,
    exact_cover_refill_from_candidate_tetrahedra, improve_refill_with_local_flips,
    next_cavity_node_id, raw_refill_tetrahedron, raw_refill_tetrahedron_with_rejection_reason,
    refill_from_tetrahedra, refill_is_better, refill_validation_reason,
    star_refill_candidate_with_rejection_reason, tetrahedralize_points, ConnectivityPoint,
    ConstrainedCavity, ConstrainedCavityNode, ConstrainedCavityRefill,
    ConstrainedCavityRefillOptions, ConstrainedCavityRefillTetrahedron,
    ConstrainedCavityValidationError,
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
