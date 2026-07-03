use super::super::{
    cavity_boundary_node_centroid, next_cavity_node_id,
    refill_tetrahedra::star_refill_candidate_with_rejection_reason, ConstrainedCavityNode,
};
use super::*;

pub(in super::super) fn centroid_interior_refill_candidate(
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
