use std::collections::BTreeMap;

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    quality::tolerance::MeshingTolerance,
};

use super::super::super::{
    cavity_boundary_node_ids, raw_refill_tetrahedron_with_rejection_reason,
    topology::sorted_tetrahedron_nodes, ConstrainedCavity, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron, ConstrainedCavityValidationError, RefillTetrahedronError,
};

pub(in super::super::super) fn best_boundary_face_completion_tetrahedron(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Result<Option<ConstrainedCavityRefillTetrahedron>, ConstrainedCavityValidationError> {
    let mut best = None::<ConstrainedCavityRefillTetrahedron>;
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
            continue;
        }
        let tetrahedron =
            match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
                Ok(tetrahedron) => tetrahedron,
                Err(RefillTetrahedronError::Rejected(_)) => continue,
                Err(RefillTetrahedronError::Accounting(cause)) => {
                    return Err(ConstrainedCavityValidationError::Accounting(cause))
                }
            };
        if refill_tetrahedra.iter().any(|existing| {
            sorted_tetrahedron_nodes(existing.node_ids)
                == sorted_tetrahedron_nodes(tetrahedron.node_ids)
        }) {
            continue;
        }
        if best.as_ref().is_none_or(|current| {
            tetrahedron
                .exact_scaled_jacobian
                .total_cmp(&current.exact_scaled_jacobian)
                .then_with(|| current.aspect_ratio.total_cmp(&tetrahedron.aspect_ratio))
                .is_gt()
        }) {
            best = Some(tetrahedron);
        }
    }
    Ok(best)
}
