use crate::quality::tolerance::MeshingTolerance;

use super::{
    types::{Point3, RayTriangleHit, Triangle3},
    vector::{cross, distance, dot, norm, sub},
};

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
