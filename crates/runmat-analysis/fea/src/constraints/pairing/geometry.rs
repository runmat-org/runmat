//! Existing geometric formulas for deterministic surface projection and periodic transforms.
use super::super::vector::{add, cross, dot, magnitude, scale, subtract};
use runmat_analysis_core::PeriodicTransform;

pub(super) fn transform_position(transform: &PeriodicTransform, point: [f64; 3]) -> [f64; 3] {
    match transform {
        PeriodicTransform::Identity => point,
        PeriodicTransform::Translation(value) => add(point, value.canonical_value()),
        PeriodicTransform::Rotation {
            origin,
            axis,
            angle,
        } => add(
            origin.canonical_value(),
            matrix_vector(
                rotation_matrix(*axis, angle.canonical_value()),
                subtract(point, origin.canonical_value()),
            ),
        ),
    }
}

pub(super) fn rotation_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let inverse_length = magnitude(axis).recip();
    let [x, y, z] = scale(axis, inverse_length);
    let cosine = angle.cos();
    let sine = angle.sin();
    let one_minus_cosine = 1.0 - cosine;
    [
        [
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ],
        [
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ],
        [
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ],
    ]
}

fn matrix_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| dot(row, vector))
}

pub(super) fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

// Ericson's closest-point regions, returning a point and barycentric coordinates.
pub(super) fn closest_triangle_point(
    point: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    if magnitude(cross(ab, ac)) <= 1.0e-18 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let ap = subtract(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = subtract(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = subtract(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(subtract(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denominator = (va + vb + vc).recip();
    let v = vb * denominator;
    let w = vc * denominator;
    (add(a, add(scale(ab, v), scale(ac, w))), [1.0 - v - w, v, w])
}
