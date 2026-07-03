use serde::{Deserialize, Serialize};

use crate::tolerance::MeshingTolerance;

pub type Point3 = [f64; 3];
pub type Triangle3 = [Point3; 3];
pub type Tetrahedron3 = [Point3; 4];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RayTriangleHit {
    pub distance: f64,
    pub barycentric_u: f64,
    pub barycentric_v: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointInClosedSurface {
    Inside,
    Outside,
    OnBoundary,
}

pub fn sub(left: Point3, right: Point3) -> Point3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub fn add(left: Point3, right: Point3) -> Point3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub fn scale(value: Point3, factor: f64) -> Point3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

pub fn dot(left: Point3, right: Point3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub fn cross(left: Point3, right: Point3) -> Point3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub fn norm_squared(value: Point3) -> f64 {
    dot(value, value)
}

pub fn norm(value: Point3) -> f64 {
    norm_squared(value).sqrt()
}

pub fn distance_squared(left: Point3, right: Point3) -> f64 {
    norm_squared(sub(left, right))
}

pub fn distance(left: Point3, right: Point3) -> f64 {
    distance_squared(left, right).sqrt()
}

pub fn triangle_centroid(points: Triangle3) -> Point3 {
    scale(add(add(points[0], points[1]), points[2]), 1.0 / 3.0)
}

pub fn tetrahedron_centroid(points: Tetrahedron3) -> Point3 {
    scale(
        add(add(points[0], points[1]), add(points[2], points[3])),
        0.25,
    )
}

pub fn triangle_area(points: Triangle3) -> f64 {
    0.5 * norm(cross(sub(points[1], points[0]), sub(points[2], points[0])))
}

pub fn tetrahedron_signed_volume(points: Tetrahedron3) -> f64 {
    dot(
        sub(points[1], points[0]),
        cross(sub(points[2], points[0]), sub(points[3], points[0])),
    ) / 6.0
}

pub fn tetrahedron_volume(points: Tetrahedron3) -> f64 {
    tetrahedron_signed_volume(points).abs()
}

pub fn orient_tetrahedron_node_ids(
    mut node_ids: [u32; 4],
    points: Tetrahedron3,
) -> ([u32; 4], f64) {
    let mut signed_volume = tetrahedron_signed_volume(points);
    if signed_volume < 0.0 {
        node_ids.swap(1, 2);
        signed_volume = -signed_volume;
    }
    (node_ids, signed_volume)
}

pub fn tetrahedron_edge_aspect_ratio(points: Tetrahedron3) -> f64 {
    let mut min_edge = f64::INFINITY;
    let mut max_edge = 0.0_f64;
    for left_index in 0..4 {
        for right_index in (left_index + 1)..4 {
            let length = distance(points[left_index], points[right_index]);
            min_edge = min_edge.min(length);
            max_edge = max_edge.max(length);
        }
    }
    max_edge / min_edge.max(f64::EPSILON)
}

pub fn tetrahedron_scaled_jacobian(points: Tetrahedron3) -> f64 {
    let corners = [
        (points[0], points[1], points[2], points[3]),
        (points[1], points[0], points[3], points[2]),
        (points[2], points[0], points[1], points[3]),
        (points[3], points[0], points[2], points[1]),
    ];
    corners
        .into_iter()
        .map(|(origin, first, second, third)| {
            let first = sub(first, origin);
            let second = sub(second, origin);
            let third = sub(third, origin);
            let denominator = norm(first) * norm(second) * norm(third);
            if denominator <= f64::EPSILON {
                return 0.0;
            }
            (2.0_f64.sqrt() * dot(first, cross(second, third)) / denominator).abs()
        })
        .fold(f64::INFINITY, f64::min)
}

pub fn ray_triangle_intersection(
    origin: Point3,
    direction: Point3,
    triangle: Triangle3,
    tolerance: MeshingTolerance,
) -> Option<RayTriangleHit> {
    let edge_1 = sub(triangle[1], triangle[0]);
    let edge_2 = sub(triangle[2], triangle[0]);
    let scale_m = distance(triangle[0], triangle[1])
        .max(distance(triangle[1], triangle[2]))
        .max(distance(triangle[2], triangle[0]))
        .max(norm(direction));
    let epsilon = tolerance.length_epsilon(scale_m);
    let h = cross(direction, edge_2);
    let determinant = dot(edge_1, h);
    if determinant.abs() <= epsilon {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let s = sub(origin, triangle[0]);
    let u = inverse_determinant * dot(s, h);
    if u < -epsilon || u > 1.0 + epsilon {
        return None;
    }
    let q = cross(s, edge_1);
    let v = inverse_determinant * dot(direction, q);
    if v < -epsilon || u + v > 1.0 + epsilon {
        return None;
    }
    let distance = inverse_determinant * dot(edge_2, q);
    distance.is_finite().then_some(RayTriangleHit {
        distance,
        barycentric_u: u,
        barycentric_v: v,
    })
}

pub fn point_in_closed_triangle_surface(
    point: Point3,
    triangles: &[Triangle3],
    tolerance: MeshingTolerance,
) -> PointInClosedSurface {
    if triangles
        .iter()
        .any(|triangle| point_triangle_distance(point, *triangle) <= tolerance.absolute_m)
    {
        return PointInClosedSurface::OnBoundary;
    }
    let epsilon = tolerance.absolute_m;
    let probes = [
        ([1.0, 0.0, 0.0], [-0.37, 0.19, 0.11]),
        ([0.0, 1.0, 0.0], [0.13, -0.41, 0.23]),
        ([0.0, 0.0, 1.0], [0.17, 0.29, -0.43]),
    ];
    let inside_votes = probes
        .into_iter()
        .filter(|(direction, jitter)| {
            ray_has_odd_surface_intersections(
                add(point, scale(*jitter, epsilon)),
                *direction,
                triangles,
                tolerance,
            )
        })
        .count();
    if inside_votes >= 2 {
        PointInClosedSurface::Inside
    } else {
        PointInClosedSurface::Outside
    }
}

pub fn ray_has_odd_surface_intersections(
    origin: Point3,
    direction: Point3,
    triangles: &[Triangle3],
    tolerance: MeshingTolerance,
) -> bool {
    let mut intersections = Vec::<f64>::new();
    for triangle in triangles {
        let Some(hit) = ray_triangle_intersection(origin, direction, *triangle, tolerance) else {
            continue;
        };
        if hit.distance > tolerance.absolute_m {
            intersections.push(hit.distance);
        }
    }
    intersections.sort_by(f64::total_cmp);
    intersections.dedup_by(|left, right| (*left - *right).abs() <= tolerance.absolute_m);
    intersections.len() % 2 == 1
}

pub fn tetrahedron_circumsphere_contains_point(
    tetrahedron_points: Tetrahedron3,
    point: Point3,
    tolerance: MeshingTolerance,
) -> bool {
    let Some((center, radius_squared)) = tetrahedron_circumsphere(tetrahedron_points, tolerance)
    else {
        return false;
    };
    distance_squared(center, point) <= radius_squared * (1.0 + tolerance.relative)
}

pub fn tetrahedron_circumsphere(
    points: Tetrahedron3,
    tolerance: MeshingTolerance,
) -> Option<(Point3, f64)> {
    let a = [
        [
            2.0 * (points[1][0] - points[0][0]),
            2.0 * (points[1][1] - points[0][1]),
            2.0 * (points[1][2] - points[0][2]),
        ],
        [
            2.0 * (points[2][0] - points[0][0]),
            2.0 * (points[2][1] - points[0][1]),
            2.0 * (points[2][2] - points[0][2]),
        ],
        [
            2.0 * (points[3][0] - points[0][0]),
            2.0 * (points[3][1] - points[0][1]),
            2.0 * (points[3][2] - points[0][2]),
        ],
    ];
    let b = [
        dot(points[1], points[1]) - dot(points[0], points[0]),
        dot(points[2], points[2]) - dot(points[0], points[0]),
        dot(points[3], points[3]) - dot(points[0], points[0]),
    ];
    let center = solve_3x3(a, b, tolerance)?;
    Some((center, distance_squared(center, points[0])))
}

pub fn point_triangle_distance(point: Point3, triangle: Triangle3) -> f64 {
    distance(point, closest_point_on_triangle(point, triangle))
}

pub fn closest_point_on_triangle(point: Point3, triangle: Triangle3) -> Point3 {
    let a = triangle[0];
    let b = triangle[1];
    let c = triangle[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }

    let normal = cross(ab, ac);
    let normal_dot = dot(normal, normal);
    if normal_dot <= f64::EPSILON {
        return [a, b, c]
            .into_iter()
            .min_by(|left, right| distance(point, *left).total_cmp(&distance(point, *right)))
            .unwrap_or(a);
    }
    sub(point, scale(normal, dot(ap, normal) / normal_dot))
}

pub fn solve_3x3(
    mut a: [[f64; 3]; 3],
    mut b: [f64; 3],
    tolerance: MeshingTolerance,
) -> Option<Point3> {
    for pivot in 0..3 {
        let mut pivot_row = pivot;
        for row in (pivot + 1)..3 {
            if a[row][pivot].abs() > a[pivot_row][pivot].abs() {
                pivot_row = row;
            }
        }
        if a[pivot_row][pivot].abs() <= tolerance.absolute_m {
            return None;
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }
        let pivot_value = a[pivot][pivot];
        for column in pivot..3 {
            a[pivot][column] /= pivot_value;
        }
        b[pivot] /= pivot_value;
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for column in pivot..3 {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetrahedron_signed_volume_reports_orientation() {
        let tetrahedron = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        assert!((tetrahedron_signed_volume(tetrahedron) - 1.0 / 6.0).abs() < 1.0e-12);
        let (node_ids, volume) = orient_tetrahedron_node_ids(
            [1, 3, 2, 4],
            [
                tetrahedron[0],
                tetrahedron[2],
                tetrahedron[1],
                tetrahedron[3],
            ],
        );
        assert_eq!(node_ids, [1, 2, 3, 4]);
        assert!((volume - 1.0 / 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn tetrahedron_scaled_jacobian_reports_shape_quality() {
        let orthogonal = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let sliver = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0e-4, 1.0e-4, 1.0e-5],
        ];

        assert!(tetrahedron_scaled_jacobian(orthogonal) > 0.5);
        assert!(tetrahedron_scaled_jacobian(sliver) < 0.01);
    }

    #[test]
    fn ray_triangle_intersection_returns_distance_and_barycentric_coordinates() {
        let hit = ray_triangle_intersection(
            [-1.0, 0.25, 0.25],
            [1.0, 0.0, 0.0],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            MeshingTolerance::default(),
        )
        .expect("ray should hit triangle");

        assert!((hit.distance - 1.0).abs() < 1.0e-12);
        assert!((hit.barycentric_u - 0.25).abs() < 1.0e-12);
        assert!((hit.barycentric_v - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn point_in_closed_surface_classifies_cube_points() {
        let triangles = cube_triangles();
        let tolerance = MeshingTolerance::from_bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        assert_eq!(
            point_in_closed_triangle_surface([0.5, 0.5, 0.5], &triangles, tolerance),
            PointInClosedSurface::Inside
        );
        assert_eq!(
            point_in_closed_triangle_surface([1.5, 0.5, 0.5], &triangles, tolerance),
            PointInClosedSurface::Outside
        );
        assert_eq!(
            point_in_closed_triangle_surface([1.0, 0.5, 0.5], &triangles, tolerance),
            PointInClosedSurface::OnBoundary
        );
    }

    #[test]
    fn circumsphere_contains_regular_tetrahedron_center() {
        let tetrahedron = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        assert!(tetrahedron_circumsphere_contains_point(
            tetrahedron,
            [0.5, 0.5, 0.5],
            MeshingTolerance::default()
        ));
        assert!(!tetrahedron_circumsphere_contains_point(
            tetrahedron,
            [2.0, 2.0, 2.0],
            MeshingTolerance::default()
        ));
    }

    fn cube_triangles() -> Vec<Triangle3> {
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        [
            [p[0], p[2], p[1]],
            [p[0], p[3], p[2]],
            [p[4], p[5], p[6]],
            [p[4], p[6], p[7]],
            [p[0], p[1], p[5]],
            [p[0], p[5], p[4]],
            [p[1], p[2], p[6]],
            [p[1], p[6], p[5]],
            [p[2], p[3], p[7]],
            [p[2], p[7], p[6]],
            [p[3], p[0], p[4]],
            [p[3], p[4], p[7]],
        ]
        .to_vec()
    }
}
