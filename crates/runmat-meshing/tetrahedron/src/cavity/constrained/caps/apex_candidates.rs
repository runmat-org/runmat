use std::collections::BTreeMap;

use super::super::geometry::{cross, distance, dot, midpoint, normalize};

mod optimized;

use optimized::{optimized_inplane_inward_cap_apex_candidate, optimized_normal_cap_apex_candidate};

#[derive(Debug, Clone, Copy)]
pub(in super::super) struct LocalCapApexCandidate {
    pub(in super::super) coordinates_m: [f64; 3],
    pub(in super::super) source: &'static str,
}

pub(in super::super) fn local_cap_apex_candidates(
    face: [u32; 3],
    surface_point: [f64; 3],
    cavity_centroid: [f64; 3],
    node_coordinates: &BTreeMap<u32, [f64; 3]>,
) -> Vec<LocalCapApexCandidate> {
    let mut candidates = Vec::<LocalCapApexCandidate>::new();
    for fraction in [0.03, 0.06, 0.1, 0.16, 0.25, 0.38, 0.55, 0.75] {
        candidates.push(LocalCapApexCandidate {
            coordinates_m: [
                surface_point[0] + (cavity_centroid[0] - surface_point[0]) * fraction,
                surface_point[1] + (cavity_centroid[1] - surface_point[1]) * fraction,
                surface_point[2] + (cavity_centroid[2] - surface_point[2]) * fraction,
            ],
            source: "centroid_inward",
        });
    }

    let Some(first) = node_coordinates.get(&face[0]).copied() else {
        return candidates;
    };
    let Some(second) = node_coordinates.get(&face[1]).copied() else {
        return candidates;
    };
    let Some(third) = node_coordinates.get(&face[2]).copied() else {
        return candidates;
    };
    let first_edge = [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ];
    let second_edge = [
        third[0] - first[0],
        third[1] - first[1],
        third[2] - first[2],
    ];
    let normal = cross(first_edge, second_edge);
    let Some(unit_normal) = normalize(normal) else {
        return candidates;
    };
    let max_edge_length = distance(first, second)
        .max(distance(second, third))
        .max(distance(third, first));
    if !max_edge_length.is_finite() || max_edge_length <= 0.0 {
        return candidates;
    }
    let inward = [
        cavity_centroid[0] - surface_point[0],
        cavity_centroid[1] - surface_point[1],
        cavity_centroid[2] - surface_point[2],
    ];
    for direction in [
        unit_normal,
        [-unit_normal[0], -unit_normal[1], -unit_normal[2]],
    ] {
        let source = if direction == unit_normal {
            "normal_positive"
        } else {
            "normal_negative"
        };
        for scale in [0.08, 0.14, 0.22, 0.35, 0.55, 0.85, 1.25] {
            let distance = max_edge_length * scale;
            candidates.push(LocalCapApexCandidate {
                coordinates_m: [
                    surface_point[0] + direction[0] * distance,
                    surface_point[1] + direction[1] * distance,
                    surface_point[2] + direction[2] * distance,
                ],
                source,
            });
        }
        candidates.push(optimized_normal_cap_apex_candidate(
            [first, second, third],
            surface_point,
            direction,
            max_edge_length,
            if direction == unit_normal {
                "normal_optimized_positive"
            } else {
                "normal_optimized_negative"
            },
        ));
    }
    if let Some(inward_direction) = normalize(inward) {
        let in_plane_targets = [
            first,
            second,
            third,
            midpoint(first, second),
            midpoint(second, third),
            midpoint(third, first),
        ];
        let mut seen_directions = Vec::<[f64; 3]>::new();
        for target in in_plane_targets {
            let projected = [
                target[0] - surface_point[0],
                target[1] - surface_point[1],
                target[2] - surface_point[2],
            ];
            let normal_projection = dot(projected, unit_normal);
            let in_plane = [
                projected[0] - unit_normal[0] * normal_projection,
                projected[1] - unit_normal[1] * normal_projection,
                projected[2] - unit_normal[2] * normal_projection,
            ];
            let Some(in_plane_direction) = normalize(in_plane) else {
                continue;
            };
            if seen_directions
                .iter()
                .any(|seen| dot(*seen, in_plane_direction).abs() > 0.98)
            {
                continue;
            }
            seen_directions.push(in_plane_direction);
            for inward_scale in [0.08, 0.16, 0.28, 0.42] {
                for lateral_scale in [0.12, 0.24, 0.38] {
                    candidates.push(LocalCapApexCandidate {
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
                        source: "inplane_inward",
                    });
                    candidates.push(LocalCapApexCandidate {
                        coordinates_m: [
                            surface_point[0] + inward_direction[0] * max_edge_length * inward_scale
                                - in_plane_direction[0] * max_edge_length * lateral_scale,
                            surface_point[1] + inward_direction[1] * max_edge_length * inward_scale
                                - in_plane_direction[1] * max_edge_length * lateral_scale,
                            surface_point[2] + inward_direction[2] * max_edge_length * inward_scale
                                - in_plane_direction[2] * max_edge_length * lateral_scale,
                        ],
                        source: "inplane_inward",
                    });
                }
            }
            candidates.push(optimized_inplane_inward_cap_apex_candidate(
                [first, second, third],
                surface_point,
                inward_direction,
                in_plane_direction,
                max_edge_length,
            ));
            candidates.push(optimized_inplane_inward_cap_apex_candidate(
                [first, second, third],
                surface_point,
                inward_direction,
                [
                    -in_plane_direction[0],
                    -in_plane_direction[1],
                    -in_plane_direction[2],
                ],
                max_edge_length,
            ));
        }
    }
    candidates
}
