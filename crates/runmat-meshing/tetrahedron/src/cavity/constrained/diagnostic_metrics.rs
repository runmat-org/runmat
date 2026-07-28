use std::collections::BTreeMap;

use runmat_meshing_core::quality::predicate::Point3;

#[cfg(test)]
pub(super) fn diagnostic_scaled_jacobian_bin(value: f64) -> String {
    if value < 0.01 {
        "lt_0_01".to_string()
    } else if value < 0.05 {
        "lt_0_05".to_string()
    } else if value < 0.10 {
        "lt_0_10".to_string()
    } else if value < 0.15 {
        "lt_0_15".to_string()
    } else {
        "gte_0_15".to_string()
    }
}

#[cfg(test)]
pub(super) fn diagnostic_face_apex_height_ratio(
    face: [u32; 3],
    apex_node_id: u32,
    boundary_nodes: &BTreeMap<u32, Point3>,
) -> f64 {
    let triangle = face.map(|node_id| boundary_nodes[&node_id]);
    let apex = boundary_nodes[&apex_node_id];
    let longest_edge = runmat_meshing_core::quality::predicate::distance(triangle[0], triangle[1])
        .max(runmat_meshing_core::quality::predicate::distance(
            triangle[1],
            triangle[2],
        ))
        .max(runmat_meshing_core::quality::predicate::distance(
            triangle[2],
            triangle[0],
        ));
    if !longest_edge.is_finite() || longest_edge <= f64::EPSILON {
        return 0.0;
    }
    let edge_ab = [
        triangle[1][0] - triangle[0][0],
        triangle[1][1] - triangle[0][1],
        triangle[1][2] - triangle[0][2],
    ];
    let edge_ac = [
        triangle[2][0] - triangle[0][0],
        triangle[2][1] - triangle[0][1],
        triangle[2][2] - triangle[0][2],
    ];
    let normal = [
        edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
        edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
        edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
    ];
    let normal_length =
        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !normal_length.is_finite() || normal_length <= f64::EPSILON {
        return 0.0;
    }
    let apex_delta = [
        apex[0] - triangle[0][0],
        apex[1] - triangle[0][1],
        apex[2] - triangle[0][2],
    ];
    let signed_height =
        (apex_delta[0] * normal[0] + apex_delta[1] * normal[1] + apex_delta[2] * normal[2])
            / normal_length;
    signed_height.abs() / longest_edge
}

#[cfg(test)]
pub(super) fn diagnostic_height_ratio_bin(value: f64) -> String {
    if value < 0.01 {
        "lt_0_01".to_string()
    } else if value < 0.05 {
        "lt_0_05".to_string()
    } else if value < 0.10 {
        "lt_0_10".to_string()
    } else if value < 0.25 {
        "lt_0_25".to_string()
    } else {
        "gte_0_25".to_string()
    }
}

#[cfg(test)]
pub(super) fn diagnostic_scaled_jacobian_worst_corner_label(points: [Point3; 4]) -> &'static str {
    let corners = [
        (0_usize, points[0], points[1], points[2], points[3]),
        (1_usize, points[1], points[0], points[3], points[2]),
        (2_usize, points[2], points[0], points[1], points[3]),
        (3_usize, points[3], points[0], points[2], points[1]),
    ];
    let worst_corner = corners
        .into_iter()
        .map(|(index, origin, first, second, third)| {
            let first = runmat_meshing_core::quality::predicate::sub(first, origin);
            let second = runmat_meshing_core::quality::predicate::sub(second, origin);
            let third = runmat_meshing_core::quality::predicate::sub(third, origin);
            let denominator = runmat_meshing_core::quality::predicate::norm(first)
                * runmat_meshing_core::quality::predicate::norm(second)
                * runmat_meshing_core::quality::predicate::norm(third);
            let scaled_jacobian = if denominator <= f64::EPSILON {
                0.0
            } else {
                (2.0_f64.sqrt()
                    * runmat_meshing_core::quality::predicate::dot(
                        first,
                        runmat_meshing_core::quality::predicate::cross(second, third),
                    )
                    / denominator)
                    .abs()
            };
            (index, scaled_jacobian)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(index, _)| index)
        .unwrap_or(3);
    if worst_corner == 3 {
        "apex"
    } else {
        "face_vertex"
    }
}
