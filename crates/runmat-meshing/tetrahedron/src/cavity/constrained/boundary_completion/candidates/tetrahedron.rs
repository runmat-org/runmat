use std::collections::BTreeMap;

use runmat_meshing_core::{
    predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
        Triangle3,
    },
    tolerance::MeshingTolerance,
};

use super::super::super::{
    cavity_boundary_node_ids, raw_refill_tetrahedron_with_rejection_reason,
    topology::sorted_tetrahedron_nodes, ConstrainedCavity, ConstrainedCavityRefillOptions,
    ConstrainedCavityRefillTetrahedron,
};

pub(in super::super::super) fn best_boundary_face_completion_tetrahedron(
    face: [u32; 3],
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    refill_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    boundary_triangles: &[Triangle3],
    options: ConstrainedCavityRefillOptions,
) -> Option<ConstrainedCavityRefillTetrahedron> {
    cavity_boundary_node_ids(cavity)
        .into_iter()
        .filter(|node_id| !face.contains(node_id))
        .filter_map(|node_id| {
            let node_ids = [face[0], face[1], face[2], node_id];
            let points = node_ids.map(|id| boundary_nodes[&id]);
            if point_in_closed_triangle_surface(
                tetrahedron_centroid(points),
                boundary_triangles,
                MeshingTolerance::default(),
            ) != PointInClosedSurface::Inside
            {
                return None;
            }
            let tetrahedron =
                raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options).ok()?;
            if refill_tetrahedra.iter().any(|existing| {
                sorted_tetrahedron_nodes(existing.node_ids)
                    == sorted_tetrahedron_nodes(tetrahedron.node_ids)
            }) {
                return None;
            }
            Some(tetrahedron)
        })
        .max_by(|left, right| {
            left.exact_scaled_jacobian
                .total_cmp(&right.exact_scaled_jacobian)
                .then_with(|| right.aspect_ratio.total_cmp(&left.aspect_ratio))
        })
}
