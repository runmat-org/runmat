use crate::quality::tolerance::MeshingTolerance;

use super::{
    distance::point_triangle_distance,
    orient2d, orient3d,
    ray::ray_triangle_intersection,
    types::{Point3, PointInClosedSurface, Triangle3},
    vector::{add, scale},
    PredicateSign,
};

pub fn point_in_closed_triangle_surface(
    point: Point3,
    triangles: &[Triangle3],
    tolerance: MeshingTolerance,
) -> PointInClosedSurface {
    let on_boundary = if tolerance.absolute_m.canonical_value() == 0.0 {
        triangles
            .iter()
            .any(|triangle| point_on_triangle_exact(point, *triangle))
    } else {
        triangles.iter().any(|triangle| {
            point_triangle_distance(point, *triangle) <= tolerance.absolute_m.canonical_value()
        })
    };
    if on_boundary {
        return PointInClosedSurface::OnBoundary;
    }
    let epsilon = tolerance.absolute_m.canonical_value();
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

fn point_on_triangle_exact(point: Point3, triangle: Triangle3) -> bool {
    if orient3d([triangle[0], triangle[1], triangle[2], point]) != Ok(PredicateSign::Zero) {
        return false;
    }
    let Some(axes) = super::triangle_projection_axes(triangle).ok().flatten() else {
        return false;
    };
    let projected = triangle.map(|vertex| [vertex[axes[0]], vertex[axes[1]]]);
    let query = [point[axes[0]], point[axes[1]]];
    let signs = [
        orient2d([projected[0], projected[1], query]),
        orient2d([projected[1], projected[2], query]),
        orient2d([projected[2], projected[0], query]),
    ];
    if signs.iter().any(Result::is_err) {
        return false;
    }
    let signs = signs.map(Result::unwrap);
    !(signs.contains(&PredicateSign::Positive) && signs.contains(&PredicateSign::Negative))
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
        if hit.distance > tolerance.absolute_m.canonical_value() {
            intersections.push(hit.distance);
        }
    }
    intersections.sort_by(f64::total_cmp);
    intersections
        .dedup_by(|left, right| (*left - *right).abs() <= tolerance.absolute_m.canonical_value());
    intersections.len() % 2 == 1
}
