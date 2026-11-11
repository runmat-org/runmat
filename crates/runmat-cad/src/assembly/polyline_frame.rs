//! The frame of an edge drawn as a polyline: what a mate connector anchored to a published edge
//! sits on, the same frame the assembly's mate picks read from the display.
//!
//! - A straight edge: its midpoint, Z along the edge from its first point to its last.
//! - A circle or arc: its centre (a least-squares fit in the edge's plane), Z along the plane's
//!   normal (Newell's method), and the radius.
//!
//! X is [`perpendicular`] to Z in both cases.

pub type V3 = [f64; 3];

/// An edge's frame: origin, Z and X, and a circle's radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolylineFrame {
    pub origin: V3,
    pub z: V3,
    pub x: V3,
    pub radius: Option<f64>,
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> V3 {
    let l = norm(a);
    if l > 1e-15 {
        scale(a, 1.0 / l)
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// A unit vector perpendicular to `z`, along the world axis that leans least on it.
pub fn perpendicular(z: V3) -> V3 {
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let least = axes
        .into_iter()
        .min_by(|a, b| dot(*a, z).abs().total_cmp(&dot(*b, z).abs()))
        .unwrap_or([1.0, 0.0, 0.0]);
    unit(sub(least, scale(z, dot(least, z))))
}

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(m);
    if d.abs() < 1e-300 {
        return None;
    }
    let mut out = [0.0; 3];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut mk = m;
        for i in 0..3 {
            mk[i][k] = r[i];
        }
        *slot = det(mk) / d;
    }
    Some(out)
}

/// The circle through `points` (all roughly in the plane through `centre` normal to `z`): its
/// centre and radius, by an algebraic least-squares fit in the plane.
pub fn fit_circle(points: &[V3], centre: V3, z: V3) -> Option<(V3, f64)> {
    let u = perpendicular(z);
    let v = cross(z, u);
    // x² + y² + D x + E y + F = 0, least squares.
    let (mut m, mut r) = ([[0.0_f64; 3]; 3], [0.0_f64; 3]);
    for p in points {
        let d = sub(*p, centre);
        let (x, y) = (dot(d, u), dot(d, v));
        let row = [x, y, 1.0];
        let rhs = -(x * x + y * y);
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += row[i] * row[j];
            }
            r[i] += row[i] * rhs;
        }
    }
    let [d, e, f] = solve3(m, r)?;
    let (cx, cy) = (-d / 2.0, -e / 2.0);
    let radius2 = cx * cx + cy * cy - f;
    (radius2 > 0.0).then(|| (add(centre, add(scale(u, cx), scale(v, cy))), radius2.sqrt()))
}

/// The frame of the edge drawn by `points`, or `None` for fewer than two points or a curve
/// that is neither straight nor round.
pub fn polyline_frame(p: &[V3]) -> Option<PolylineFrame> {
    let (first, last) = (*p.first()?, *p.last()?);
    if p.len() < 2 {
        return None;
    }
    let chord = sub(last, first);
    let straight = p.len() == 2
        || p.iter().all(|q| norm(cross(sub(*q, first), unit(chord))) <= 1e-6 * norm(chord).max(1e-9));
    if straight && norm(chord) > 0.0 {
        let z = unit(chord);
        return Some(PolylineFrame { origin: scale(add(first, last), 0.5), z, x: perpendicular(z), radius: None });
    }
    let mean = scale(p.iter().fold([0.0; 3], |s, q| add(s, *q)), 1.0 / p.len() as f64);
    let mut n = [0.0; 3];
    for (i, a) in p.iter().enumerate() {
        let b = p[(i + 1) % p.len()];
        n = add(n, cross(sub(*a, mean), sub(b, mean)));
    }
    let z = unit(n);
    let (centre, radius) = fit_circle(p, mean, z)?;
    Some(PolylineFrame { origin: centre, z, x: perpendicular(z), radius: Some(radius) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_and_round_edges_give_their_frames() {
        let line = polyline_frame(&[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]).unwrap();
        assert_eq!(line.origin, [0.01, 0.0, 0.0]);
        assert_eq!(line.z, [1.0, 0.0, 0.0]);
        assert!(line.radius.is_none());
        let circle: Vec<V3> = (0..=24)
            .map(|k| {
                let a = k as f64 / 24.0 * std::f64::consts::TAU;
                [0.005 + 0.003 * a.cos(), 0.002 + 0.003 * a.sin(), 0.01]
            })
            .collect();
        let f = polyline_frame(&circle).unwrap();
        assert!(norm(sub(f.origin, [0.005, 0.002, 0.01])) < 1e-9, "{f:?}");
        assert!((f.z[2].abs() - 1.0).abs() < 1e-9);
        assert!((f.radius.unwrap() - 0.003).abs() < 1e-9);
    }
}
