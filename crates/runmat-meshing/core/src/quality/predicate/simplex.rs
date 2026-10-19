use super::{
    types::{Point3, Tetrahedron3, Triangle3},
    vector::{add, cross, distance, norm, scale, sub},
};
use robust::Coord3D;

pub fn triangle_centroid(points: Triangle3) -> Point3 {
    scale(add(add(points[0], points[1]), points[2]), 1.0 / 3.0)
}

pub fn tetrahedron_centroid(points: Tetrahedron3) -> Point3 {
    std::array::from_fn(|axis| {
        points[0][axis]
            + ((points[1][axis] - points[0][axis])
                + (points[2][axis] - points[0][axis])
                + (points[3][axis] - points[0][axis]))
                * 0.25
    })
}

pub fn triangle_area(points: Triangle3) -> f64 {
    0.5 * norm(cross(sub(points[1], points[0]), sub(points[2], points[0])))
}

pub fn tetrahedron_signed_volume(points: Tetrahedron3) -> f64 {
    -robust::orient3d(
        Coord3D {
            x: points[0][0],
            y: points[0][1],
            z: points[0][2],
        },
        Coord3D {
            x: points[1][0],
            y: points[1][1],
            z: points[1][2],
        },
        Coord3D {
            x: points[2][0],
            y: points[2][1],
            z: points[2][2],
        },
        Coord3D {
            x: points[3][0],
            y: points[3][1],
            z: points[3][2],
        },
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
            let Some(first) = normalized(sub(first, origin)) else {
                return 0.0;
            };
            let Some(second) = normalized(sub(second, origin)) else {
                return 0.0;
            };
            let Some(third) = normalized(sub(third, origin)) else {
                return 0.0;
            };
            // Normalize before evaluating the dimensionless determinant so the
            // cubic edge product cannot overflow or underflow. The adaptive
            // determinant retains the sign and magnitude of nearly coplanar unit
            // vectors far beyond a direct dot/cross evaluation.
            let determinant = -robust::orient3d(
                Coord3D {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                coord(first),
                coord(second),
                coord(third),
            );
            (2.0_f64.sqrt() * determinant).abs()
        })
        .fold(f64::INFINITY, f64::min)
}

fn normalized(vector: Point3) -> Option<Point3> {
    let scale = vector.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let scaled = vector.map(|value| value / scale);
    let length = scaled
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    (length.is_finite() && length > 0.0).then(|| scaled.map(|value| value / length))
}

fn coord(point: Point3) -> Coord3D<f64> {
    Coord3D {
        x: point[0],
        y: point[1],
        z: point[2],
    }
}
