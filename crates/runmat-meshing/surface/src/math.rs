use serde::{Deserialize, Serialize};

pub type Point3 = [f64; 3];
pub type Triangle3 = [Point3; 3];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshingTolerance {
    pub absolute_m: f64,
    pub relative: f64,
}

impl Default for MeshingTolerance {
    fn default() -> Self {
        Self {
            absolute_m: 1.0e-12,
            relative: 1.0e-10,
        }
    }
}

impl MeshingTolerance {
    pub fn from_bounds(bounds_min_m: [f64; 3], bounds_max_m: [f64; 3]) -> Self {
        let span = (0..3)
            .map(|axis| bounds_max_m[axis] - bounds_min_m[axis])
            .filter(|span| span.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        Self {
            absolute_m: (span * 1.0e-10).max(1.0e-12),
            relative: 1.0e-10,
        }
    }

    pub fn length_epsilon(self, scale_m: f64) -> f64 {
        let scale_m = if scale_m.is_finite() {
            scale_m.abs()
        } else {
            1.0
        };
        self.absolute_m.max(scale_m * self.relative)
    }
}

pub fn add(left: Point3, right: Point3) -> Point3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub fn sub(left: Point3, right: Point3) -> Point3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
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

pub fn triangle_area(points: Triangle3) -> f64 {
    0.5 * norm(cross(sub(points[1], points[0]), sub(points[2], points[0])))
}

pub fn point_triangle_distance(point: Point3, triangle: Triangle3) -> f64 {
    distance(point, closest_point_on_triangle(point, triangle))
}

fn closest_point_on_triangle(point: Point3, triangle: Triangle3) -> Point3 {
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
        return add(a, scale(ab, d1 / (d1 - d3)));
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = sub(c, b);
        return add(b, scale(bc, (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }

    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    add(a, add(scale(ab, v), scale(ac, w)))
}
