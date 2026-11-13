use crate::quality::tolerance::MeshingTolerance;

use super::{
    solve::solve_3x3,
    types::{Point3, Tetrahedron3},
    vector::{add, distance_squared, dot, sub},
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
    // Solve relative to the first point. Expanding squared absolute
    // coordinates loses the cell-scale signal when a small mesh is translated
    // far from the origin.
    let edges = [
        sub(points[1], points[0]),
        sub(points[2], points[0]),
        sub(points[3], points[0]),
    ];
    let a = edges.map(|edge| edge.map(|coordinate| 2.0 * coordinate));
    let b = edges.map(|edge| dot(edge, edge));
    let center_offset = solve_3x3(a, b, tolerance)?;
    let center = add(points[0], center_offset);
    Some((center, dot(center_offset, center_offset)))
}
