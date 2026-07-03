use super::super::{
    boundary_completion::complete_missing_boundary_face_tetrahedra,
    exact_cover::boundary_node_exact_cover_refill_candidate,
    refill_tetrahedra::{
        improve_refill_with_local_flips, raw_refill_tetrahedron_with_rejection_reason,
        refill_validation_reason,
    },
    tetrahedralize_points, ConnectivityPoint, ConstrainedCavityRefillTetrahedron,
};
use super::*;

pub(in super::super) fn boundary_node_refill_candidate(
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
                boundary_nodes,
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
            boundary_nodes,
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
                    boundary_nodes,
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
                boundary_nodes,
                &refill,
                options,
            )
            .unwrap_or(refill);
            Ok(Ok(refill))
        }
    }
}

pub(in super::super) fn boundary_node_refill_rejection_reason(
    reason: &'static str,
) -> &'static str {
    match reason {
        "star_tetrahedron_min_volume" => "boundary_node_tetrahedron_min_volume",
        "star_tetrahedron_aspect_ratio" => "boundary_node_tetrahedron_aspect_ratio",
        "star_tetrahedron_scaled_jacobian" => "boundary_node_tetrahedron_scaled_jacobian",
        other => other,
    }
}

pub(in super::super) fn boundary_node_refill_validation_reason(
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
