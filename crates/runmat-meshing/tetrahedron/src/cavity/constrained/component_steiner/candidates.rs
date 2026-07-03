use runmat_meshing_core::{
    predicate::{
        tetrahedron_centroid, tetrahedron_circumsphere, tetrahedron_scaled_jacobian, triangle_area,
        Point3,
    },
    tolerance::MeshingTolerance,
};

use super::super::topology::{tetrahedron_edges, tetrahedron_faces};

pub(super) fn component_steiner_candidate_points(
    node_ids: [u32; 4],
    points: [Point3; 4],
    tetrahedron_centroid: Point3,
    cavity_centroid: Point3,
) -> Vec<Point3> {
    let mut candidates = Vec::<Point3>::new();
    candidates.push(tetrahedron_centroid);
    candidates.push(tetrahedron_incenter(points));
    if let Some((center, _)) = tetrahedron_circumsphere(points, MeshingTolerance::default()) {
        candidates.push(center);
        for fraction in [0.25, 0.5, 0.75] {
            candidates.push(interpolate(tetrahedron_centroid, center, fraction));
        }
    }
    for face in tetrahedron_faces(node_ids) {
        let face_points = face.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate| *candidate == node_id)
                .expect("tetrahedron face node should be in tetrahedron");
            points[index]
        });
        let face_centroid = [
            (face_points[0][0] + face_points[1][0] + face_points[2][0]) / 3.0,
            (face_points[0][1] + face_points[1][1] + face_points[2][1]) / 3.0,
            (face_points[0][2] + face_points[1][2] + face_points[2][2]) / 3.0,
        ];
        for fraction in [0.18, 0.33, 0.5, 0.67] {
            candidates.push(interpolate(face_centroid, tetrahedron_centroid, fraction));
        }
        for fraction in [0.12, 0.25, 0.4] {
            candidates.push(interpolate(face_centroid, cavity_centroid, fraction));
        }
    }
    for edge in tetrahedron_edges(node_ids) {
        let edge_points = edge.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate| *candidate == node_id)
                .expect("tetrahedron edge node should be in tetrahedron");
            points[index]
        });
        let midpoint = [
            (edge_points[0][0] + edge_points[1][0]) * 0.5,
            (edge_points[0][1] + edge_points[1][1]) * 0.5,
            (edge_points[0][2] + edge_points[1][2]) * 0.5,
        ];
        for fraction in [0.2, 0.4, 0.6] {
            candidates.push(interpolate(midpoint, tetrahedron_centroid, fraction));
        }
    }
    candidates
}

pub(super) fn component_steiner_candidate_quality_score(
    node_ids: [u32; 4],
    points: [Point3; 4],
    candidate: Point3,
) -> f64 {
    let mut min_quality = f64::INFINITY;
    for face in tetrahedron_faces(node_ids) {
        let face_points = face.map(|node_id| {
            let index = node_ids
                .iter()
                .position(|candidate_id| *candidate_id == node_id)
                .expect("tetrahedron face node should be in tetrahedron");
            points[index]
        });
        let quality = tetrahedron_scaled_jacobian([
            face_points[0],
            face_points[1],
            face_points[2],
            candidate,
        ]);
        min_quality = min_quality.min(quality);
    }
    min_quality
}

fn tetrahedron_incenter(points: [Point3; 4]) -> Point3 {
    let weights = [
        triangle_area([points[1], points[2], points[3]]),
        triangle_area([points[0], points[3], points[2]]),
        triangle_area([points[0], points[1], points[3]]),
        triangle_area([points[0], points[2], points[1]]),
    ];
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= f64::EPSILON {
        return tetrahedron_centroid(points);
    }
    [
        (points[0][0] * weights[0]
            + points[1][0] * weights[1]
            + points[2][0] * weights[2]
            + points[3][0] * weights[3])
            / total,
        (points[0][1] * weights[0]
            + points[1][1] * weights[1]
            + points[2][1] * weights[2]
            + points[3][1] * weights[3])
            / total,
        (points[0][2] * weights[0]
            + points[1][2] * weights[1]
            + points[2][2] * weights[2]
            + points[3][2] * weights[3])
            / total,
    ]
}

fn interpolate(left: Point3, right: Point3, fraction: f64) -> Point3 {
    [
        left[0] * (1.0 - fraction) + right[0] * fraction,
        left[1] * (1.0 - fraction) + right[1] * fraction,
        left[2] * (1.0 - fraction) + right[2] * fraction,
    ]
}
