use super::{
    types::{Point3, Tetrahedron3, Triangle3},
    vector::{add, cross, distance, dot, norm, scale, sub},
};

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
