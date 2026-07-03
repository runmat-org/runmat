use runmat_meshing_cad::CadFaceEvaluationFrame;

use crate::math::{dot, sub};

use super::{boundary_triangulation_points, FaceCurveSegment, FaceTriangulationPoint, SurfaceNode};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TriangulationPoint {
    pub(super) uv: [f64; 2],
    pub(super) original_index: Option<usize>,
    pub(super) is_super: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TriangulationTriangle {
    pub(super) point_indices: [usize; 3],
}

pub(super) fn super_triangle_points(points: &[FaceTriangulationPoint]) -> [TriangulationPoint; 3] {
    let mut min = points[0].uv;
    let mut max = points[0].uv;
    for point in points {
        min[0] = min[0].min(point.uv[0]);
        min[1] = min[1].min(point.uv[1]);
        max[0] = max[0].max(point.uv[0]);
        max[1] = max[1].max(point.uv[1]);
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let span = (max[0] - min[0]).max(max[1] - min[1]).max(1.0);
    [
        TriangulationPoint {
            uv: [center[0] - 32.0 * span, center[1] - span],
            original_index: None,
            is_super: true,
        },
        TriangulationPoint {
            uv: [center[0], center[1] + 32.0 * span],
            original_index: None,
            is_super: true,
        },
        TriangulationPoint {
            uv: [center[0] + 32.0 * span, center[1] - span],
            original_index: None,
            is_super: true,
        },
    ]
}

pub(super) fn circumcircle_contains(triangle: [[f64; 2]; 3], point: [f64; 2]) -> bool {
    let ax = triangle[0][0] - point[0];
    let ay = triangle[0][1] - point[1];
    let bx = triangle[1][0] - point[0];
    let by = triangle[1][1] - point[1];
    let cx = triangle[2][0] - point[0];
    let cy = triangle[2][1] - point[1];
    let determinant = (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx);
    let orientation = triangle_area_2d(triangle);
    if orientation > 0.0 {
        determinant > -1.0e-12
    } else {
        determinant < 1.0e-12
    }
}

pub(super) fn triangle_edges_2d(point_indices: [usize; 3]) -> [[usize; 2]; 3] {
    [
        [point_indices[0], point_indices[1]],
        [point_indices[1], point_indices[2]],
        [point_indices[2], point_indices[0]],
    ]
}

pub(super) fn triangle_area_2d(points: [[f64; 2]; 3]) -> f64 {
    0.5 * ((points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
        - (points[1][1] - points[0][1]) * (points[2][0] - points[0][0]))
}

pub(super) fn triangle_centroid_2d(points: [[f64; 2]; 3]) -> [f64; 2] {
    [
        (points[0][0] + points[1][0] + points[2][0]) / 3.0,
        (points[0][1] + points[1][1] + points[2][1]) / 3.0,
    ]
}

pub(super) fn point_in_triangle_2d(point: [f64; 2], triangle: [[f64; 2]; 3]) -> bool {
    let area = triangle_area_2d(triangle);
    if area.abs() <= f64::EPSILON {
        return false;
    }
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let edge_areas = [
        triangle_area_2d([triangle[0], triangle[1], point]) * sign,
        triangle_area_2d([triangle[1], triangle[2], point]) * sign,
        triangle_area_2d([triangle[2], triangle[0], point]) * sign,
    ];
    edge_areas.iter().all(|value| *value >= -1.0e-12)
}

pub(super) fn boundary_loop_polygons(
    frame: &CadFaceEvaluationFrame,
    segment_loops: &[Vec<FaceCurveSegment>],
    nodes: &[SurfaceNode],
) -> Vec<Vec<[f64; 2]>> {
    segment_loops
        .iter()
        .map(|segments| {
            boundary_loop_polygon(&boundary_triangulation_points(frame, segments, nodes))
        })
        .filter(|polygon| polygon.len() >= 3 && polygon_area_2d(polygon).abs() > f64::EPSILON)
        .collect()
}

fn boundary_loop_polygon(points: &[FaceTriangulationPoint]) -> Vec<[f64; 2]> {
    let mut polygon = Vec::<[f64; 2]>::new();
    for point in points {
        if polygon
            .last()
            .is_some_and(|last| distance2_2d(*last, point.uv) <= 1.0e-24)
        {
            continue;
        }
        polygon.push(point.uv);
    }
    if polygon.len() > 1
        && distance2_2d(
            polygon[0],
            *polygon.last().expect("polygon should be non-empty"),
        ) <= 1.0e-24
    {
        polygon.pop();
    }
    polygon
}

pub(super) fn point_in_trimmed_domain_2d(point: [f64; 2], polygons: &[Vec<[f64; 2]>]) -> bool {
    let Some(outer_index) = outer_boundary_polygon_index(polygons) else {
        return false;
    };
    if !point_in_polygon_2d(point, &polygons[outer_index]) {
        return false;
    }
    polygons
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != outer_index)
        .all(|(_, hole)| !point_in_polygon_2d(point, hole))
}

fn outer_boundary_polygon_index(polygons: &[Vec<[f64; 2]>]) -> Option<usize> {
    polygons
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            polygon_area_2d(left)
                .abs()
                .total_cmp(&polygon_area_2d(right).abs())
        })
        .map(|(index, _)| index)
}

fn polygon_area_2d(polygon: &[[f64; 2]]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    let mut previous = polygon[polygon.len() - 1];
    for current in polygon {
        area += previous[0] * current[1] - current[0] * previous[1];
        previous = *current;
    }
    0.5 * area
}

pub(super) fn point_in_polygon_2d(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for current in polygon {
        if point_on_segment_2d(point, previous, *current) {
            return true;
        }
        let denominator = previous[1] - current[1];
        let crosses = denominator.abs() > f64::EPSILON
            && ((current[1] > point[1]) != (previous[1] > point[1]))
            && point[0]
                < (previous[0] - current[0]) * (point[1] - current[1]) / denominator + current[0];
        if crosses {
            inside = !inside;
        }
        previous = *current;
    }
    inside
}

pub(super) fn point_on_segment_2d(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> bool {
    cross_2d(start, end, point).abs() <= 1.0e-10
        && point[0] >= start[0].min(end[0]) - 1.0e-10
        && point[0] <= start[0].max(end[0]) + 1.0e-10
        && point[1] >= start[1].min(end[1]) - 1.0e-10
        && point[1] <= start[1].max(end[1]) + 1.0e-10
}

fn cross_2d(origin: [f64; 2], left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])
}

pub(super) fn distance2_2d(left: [f64; 2], right: [f64; 2]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    dx * dx + dy * dy
}

pub(super) fn finite_point2(point: [f64; 2]) -> bool {
    point.iter().all(|value| value.is_finite())
}

pub(super) fn finite_point3(point: [f64; 3]) -> bool {
    point.iter().all(|value| value.is_finite())
}

pub(super) fn point_in_triangle_3d(point: [f64; 3], triangle: [[f64; 3]; 3]) -> bool {
    let v0 = sub(triangle[2], triangle[0]);
    let v1 = sub(triangle[1], triangle[0]);
    let v2 = sub(point, triangle[0]);
    let dot00 = dot(v0, v0);
    let dot01 = dot(v0, v1);
    let dot02 = dot(v0, v2);
    let dot11 = dot(v1, v1);
    let dot12 = dot(v1, v2);
    let denominator = dot00 * dot11 - dot01 * dot01;
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return false;
    }
    let inv_denominator = 1.0 / denominator;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denominator;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denominator;
    let tolerance = 1.0e-10;
    u >= -tolerance && v >= -tolerance && u + v <= 1.0 + tolerance
}

pub(super) fn sorted_node_pair(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

pub(super) fn sorted_index_pair(left: usize, right: usize) -> [usize; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}
