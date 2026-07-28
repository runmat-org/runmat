use crate::quality::tolerance::MeshingTolerance;

use super::{
    distance::point_triangle_distance,
    ray::ray_triangle_intersection,
    types::{Point3, PointInClosedSurface, Triangle3},
    vector::{add, scale},
};

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
