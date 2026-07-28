use super::{
    types::{Point3, Triangle3},
    vector::{add, cross, distance, dot, scale, sub},
};

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
