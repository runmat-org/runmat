use runmat_meshing_core::predicate::tetrahedron_scaled_jacobian;

use super::LocalCapApexCandidate;

pub(super) fn optimized_normal_cap_apex_candidate(
    face_points: [[f64; 3]; 3],
    surface_point: [f64; 3],
    direction: [f64; 3],
    max_edge_length: f64,
    source: &'static str,
) -> LocalCapApexCandidate {
    let quality_at = |scale: f64| {
        let distance = max_edge_length * scale;
        let apex = [
            surface_point[0] + direction[0] * distance,
            surface_point[1] + direction[1] * distance,
            surface_point[2] + direction[2] * distance,
        ];
        tetrahedron_scaled_jacobian([face_points[0], face_points[1], face_points[2], apex])
    };
    let mut low = 0.02_f64;
    let mut high = 1.75_f64;
    for _ in 0..28 {
        let left = low + (high - low) / 3.0;
        let right = high - (high - low) / 3.0;
        if quality_at(left) < quality_at(right) {
            low = left;
        } else {
            high = right;
        }
    }
    let scale = (low + high) * 0.5;
    let distance = max_edge_length * scale;
    LocalCapApexCandidate {
        coordinates_m: [
            surface_point[0] + direction[0] * distance,
            surface_point[1] + direction[1] * distance,
            surface_point[2] + direction[2] * distance,
        ],
        source,
    }
}

pub(super) fn optimized_inplane_inward_cap_apex_candidate(
    face_points: [[f64; 3]; 3],
    surface_point: [f64; 3],
    inward_direction: [f64; 3],
    in_plane_direction: [f64; 3],
    max_edge_length: f64,
) -> LocalCapApexCandidate {
    let quality_at = |inward_scale: f64, lateral_scale: f64| {
        let apex = [
            surface_point[0]
                + inward_direction[0] * max_edge_length * inward_scale
                + in_plane_direction[0] * max_edge_length * lateral_scale,
            surface_point[1]
                + inward_direction[1] * max_edge_length * inward_scale
                + in_plane_direction[1] * max_edge_length * lateral_scale,
            surface_point[2]
                + inward_direction[2] * max_edge_length * inward_scale
                + in_plane_direction[2] * max_edge_length * lateral_scale,
        ];
        tetrahedron_scaled_jacobian([face_points[0], face_points[1], face_points[2], apex])
    };
    let mut best = (0.16_f64, 0.24_f64, quality_at(0.16, 0.24));
    for inward_scale in [0.04, 0.08, 0.14, 0.22, 0.34, 0.5, 0.72] {
        for lateral_scale in [0.04, 0.1, 0.18, 0.28, 0.42, 0.6] {
            let quality = quality_at(inward_scale, lateral_scale);
            if quality > best.2 {
                best = (inward_scale, lateral_scale, quality);
            }
        }
    }
    let (inward_scale, lateral_scale, _) = best;
    LocalCapApexCandidate {
        coordinates_m: [
            surface_point[0]
                + inward_direction[0] * max_edge_length * inward_scale
                + in_plane_direction[0] * max_edge_length * lateral_scale,
            surface_point[1]
                + inward_direction[1] * max_edge_length * inward_scale
                + in_plane_direction[1] * max_edge_length * lateral_scale,
            surface_point[2]
                + inward_direction[2] * max_edge_length * inward_scale
                + in_plane_direction[2] * max_edge_length * lateral_scale,
        ],
        source: "inplane_inward_optimized",
    }
}
