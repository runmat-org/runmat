use runmat_meshing_core::quality::predicate::Triangle3;

use super::*;

#[cfg(test)]
pub(super) fn scaled_worst_face_star_quality(
    cavity: &ConstrainedCavity,
    boundary_nodes: &BTreeMap<u32, Point3>,
    boundary_triangles: &[Triangle3],
    node: &ConstrainedCavityNode,
    refill: &ConstrainedCavityRefill,
    options: ConstrainedCavityRefillOptions,
) -> Option<(usize, f64)> {
    let worst_tetrahedron = refill.tetrahedra.iter().min_by(|left, right| {
        left.exact_scaled_jacobian
            .total_cmp(&right.exact_scaled_jacobian)
    })?;
    let face_nodes = worst_tetrahedron
        .node_ids
        .into_iter()
        .filter(|node_id| *node_id != node.node_id)
        .collect::<Vec<_>>();
    if face_nodes.len() != 3 {
        return None;
    }
    let face_points = face_nodes
        .iter()
        .map(|node_id| boundary_nodes.get(node_id).copied())
        .collect::<Option<Vec<_>>>()?;
    let face_centroid = [
        (face_points[0][0] + face_points[1][0] + face_points[2][0]) / 3.0,
        (face_points[0][1] + face_points[1][1] + face_points[2][1]) / 3.0,
        (face_points[0][2] + face_points[1][2] + face_points[2][2]) / 3.0,
    ];
    let direction = [
        node.coordinates_m[0] - face_centroid[0],
        node.coordinates_m[1] - face_centroid[1],
        node.coordinates_m[2] - face_centroid[2],
    ];
    let distance_squared =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    if !distance_squared.is_finite()
        || distance_squared <= MeshingTolerance::default().absolute_m.powi(2)
    {
        return None;
    }

    let mut candidate_count = 0_usize;
    let mut best_quality = 0.0_f64;
    for scale in [0.5, 0.7, 0.85, 1.15, 1.35, 1.6, 2.0] {
        let coordinates_m = [
            face_centroid[0] + direction[0] * scale,
            face_centroid[1] + direction[1] * scale,
            face_centroid[2] + direction[2] * scale,
        ];
        if point_in_closed_triangle_surface(
            coordinates_m,
            boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        candidate_count += 1;
        let adjusted = ConstrainedCavityNode {
            node_id: node.node_id,
            coordinates_m,
        };
        let Ok(Ok(refill)) =
            star_refill_candidate_with_rejection_reason(cavity, boundary_nodes, adjusted, options)
        else {
            continue;
        };
        let min_quality = refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min);
        if min_quality.is_finite() {
            best_quality = best_quality.max(min_quality);
        }
    }
    (candidate_count > 0).then_some((candidate_count, best_quality))
}
