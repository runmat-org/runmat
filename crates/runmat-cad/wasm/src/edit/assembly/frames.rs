//! Frames on a component's faces and edges, from its compiled display geometry, in the
//! definition's own coordinates (metres): what a mate connector anchors to.
//!
//! - A planar face: its area-weighted centroid, Z along the outward normal.
//! - A cylindrical (or conical) face: a point on its axis level with the face's middle, Z along
//!   the axis, and the radius.
//! - A circular edge: its centre, Z along the circle's normal, and the radius.
//! - A straight edge: its midpoint, Z along the edge.

use serde_json::Value;

pub type V3 = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameKind {
    /// A plane through `origin` with normal `z`.
    Plane,
    /// A line through `origin` along `z` (a cylinder's or circle's axis, a straight edge).
    Axis,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub kind: FrameKind,
    pub origin: V3,
    pub z: V3,
    pub x: V3,
    /// A cylinder's or circle's radius.
    pub radius: Option<f64>,
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn unit(a: V3) -> V3 {
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

fn points(values: &Value) -> Vec<V3> {
    let flat: Vec<f64> = values
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// The circle through `points` (all roughly in the plane through `centre` normal to `z`): its
/// centre and radius, by an algebraic least-squares fit in the plane.
fn fit_circle(points: &[V3], centre: V3, z: V3) -> Option<(V3, f64)> {
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

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
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

/// The frame of display face `face` (positions, triangles).
pub fn face_frame(face: &Value) -> Option<Frame> {
    let p = points(&face["positions"]);
    let t: Vec<usize> = face["triangles"]
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    let mut normals = Vec::new();
    let (mut area_normal, mut centroid, mut area) = ([0.0; 3], [0.0; 3], 0.0);
    for tri in t.chunks_exact(3) {
        let (a, b, c) = (*p.get(tri[0])?, *p.get(tri[1])?, *p.get(tri[2])?);
        let n = cross(sub(b, a), sub(c, a));
        let w = norm(n) / 2.0;
        if w <= 0.0 {
            continue;
        }
        normals.push((unit(n), w));
        area_normal = add(area_normal, scale(n, 0.5));
        centroid = add(centroid, scale(add(add(a, b), c), w / 3.0));
        area += w;
    }
    if area <= 0.0 {
        return None;
    }
    let centroid = scale(centroid, 1.0 / area);
    let mean = unit(area_normal);
    // Planar when every triangle faces the same way (within about a degree).
    if normals.iter().all(|(n, _)| dot(*n, mean) > 0.9998) {
        return Some(Frame {
            kind: FrameKind::Plane,
            origin: centroid,
            z: mean,
            x: perpendicular(mean),
            radius: None,
        });
    }
    // Curved: the axis is normal to the triangle normals; the two most different give it.
    let mut best = ([0.0; 3], 0.0);
    for (i, (a, _)) in normals.iter().enumerate() {
        for (b, _) in normals.iter().skip(i + 1) {
            let c = cross(*a, *b);
            let l = norm(c);
            if l > best.1 {
                best = (c, l);
            }
        }
    }
    let axis = unit(best.0);
    let (centre, radius) = fit_circle(&p, centroid, axis)?;
    Some(Frame {
        kind: FrameKind::Axis,
        origin: centre,
        z: axis,
        x: perpendicular(axis),
        radius: Some(radius),
    })
}

/// The frame of display edge `edge` (a polyline of points).
pub fn edge_frame(edge: &Value) -> Option<Frame> {
    let p = points(&edge["points"]);
    let (first, last) = (*p.first()?, *p.last()?);
    let chord = sub(last, first);
    let straight = p.len() == 2
        || p.iter().all(|q| {
            let d = sub(*q, first);
            norm(cross(d, unit(chord))) <= 1e-6 * norm(chord).max(1e-9)
        });
    if straight && norm(chord) > 0.0 {
        let z = unit(chord);
        return Some(Frame {
            kind: FrameKind::Axis,
            origin: scale(add(first, last), 0.5),
            z,
            x: perpendicular(z),
            radius: None,
        });
    }
    // A circle or arc: its plane's normal by Newell's method, then the fitted centre.
    let mean = scale(p.iter().fold([0.0; 3], |s, q| add(s, *q)), 1.0 / p.len() as f64);
    let mut n = [0.0; 3];
    for (i, a) in p.iter().enumerate() {
        let b = p[(i + 1) % p.len()];
        n = add(n, cross(sub(*a, mean), sub(b, mean)));
    }
    let z = unit(n);
    let (centre, radius) = fit_circle(&p, mean, z)?;
    Some(Frame {
        kind: FrameKind::Axis,
        origin: centre,
        z,
        x: perpendicular(z),
        radius: Some(radius),
    })
}

/// `frame` in world coordinates under a display transform (row-major rotation, then translation).
pub fn to_world(frame: Frame, t: &[f64]) -> Frame {
    if t.len() < 12 {
        return frame;
    }
    let rotate = |v: V3| {
        [
            t[0] * v[0] + t[1] * v[1] + t[2] * v[2],
            t[3] * v[0] + t[4] * v[1] + t[5] * v[2],
            t[6] * v[0] + t[7] * v[1] + t[8] * v[2],
        ]
    };
    Frame {
        origin: add(rotate(frame.origin), [t[9], t[10], t[11]]),
        z: rotate(frame.z),
        x: rotate(frame.x),
        ..frame
    }
}
