use crate::quality::tolerance::MeshingTolerance;

use super::{
    solve::solve_3x3,
    types::{Point3, Tetrahedron3},
    vector::{distance_squared, dot},
};

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
