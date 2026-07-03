use super::*;

#[cfg(test)]
mod aggregate;
#[cfg(test)]
mod face_completion;
#[cfg(test)]
mod split_caps;

#[cfg(test)]
use aggregate::{
    empty_boundary_node_completion_diagnostic, merge_boundary_node_completion_diagnostic,
};
#[cfg(test)]
use face_completion::diagnostic_boundary_face_completion;

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
    let mut aggregate =
        empty_boundary_node_completion_diagnostic("boundary_node_completion_no_missing_faces");
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
        merge_boundary_node_completion_diagnostic(&mut aggregate, diagnostic);
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
        return Ok(empty_boundary_node_completion_diagnostic(
            "boundary_node_completion_no_missing_faces",
        ));
    }
    aggregate.reason = "boundary_node_completion_completed";
    Ok(aggregate)
}
