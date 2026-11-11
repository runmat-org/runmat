//! Frames on a component's faces and edges, from its compiled display geometry, in the
//! definition's own coordinates (metres): what a mate connector anchors to.
//!
//! - A planar face: its area-weighted centroid, Z along the outward normal.
//! - A cylindrical (or conical) face: a point on its axis level with the face's middle, Z along
//!   the axis, and the radius.
//! - A circular edge: its centre, Z along the circle's normal, and the radius.
//! - A straight edge: its midpoint, Z along the edge.
//! - A vertex: its point, with the world axes.
//! - A datum plane, axis, point or frame (a mate connector too), from the lines the display draws
//!   it with: a plane's square (X along its first side, Z its normal), an axis's line, a point's
//!   cross, a frame's three axis lines.

use serde_json::Value;

pub type V3 = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameKind {
    /// A plane through `origin` with normal `z`.
    Plane,
    /// A line through `origin` along `z` (a cylinder's or circle's axis, a straight edge).
    Axis,
    /// A point at `origin` (a vertex, a datum point).
    Point,
    /// A whole frame (a mate connector or coordinate system).
    Frame,
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

pub use runmat_cad::assembly::polyline_frame::perpendicular;
use runmat_cad::assembly::polyline_frame::{fit_circle, polyline_frame};

fn points(values: &Value) -> Vec<V3> {
    let flat: Vec<f64> = values
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
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

/// The frame of display edge `edge` (a polyline of points): see
/// [`polyline_frame`](runmat_cad::assembly::polyline_frame::polyline_frame).
pub fn edge_frame(edge: &Value) -> Option<Frame> {
    let f = polyline_frame(&points(&edge["points"]))?;
    Some(Frame {
        kind: FrameKind::Axis,
        origin: f.origin,
        z: f.z,
        x: f.x,
        radius: f.radius,
    })
}

/// The frame of display vertex `vertex`.
pub fn vertex_frame(vertex: &Value) -> Option<Frame> {
    let p = points(&vertex["point"]);
    Some(Frame {
        kind: FrameKind::Point,
        origin: *p.first()?,
        z: [0.0, 0.0, 1.0],
        x: [1.0, 0.0, 0.0],
        radius: None,
    })
}

/// The frame of display reference `reference` (`{ id, kind, lines }`): a datum plane, axis,
/// point, frame or mate connector.
pub fn datum_frame(reference: &Value) -> Option<Frame> {
    let lines: Vec<Vec<V3>> = reference["lines"].as_array()?.iter().map(points).collect();
    let first = lines.first()?;
    let frame = |kind, origin, z: V3, x: V3| Some(Frame { kind, origin, z: unit(z), x: unit(x), radius: None });
    match reference["kind"].as_str()? {
        "plane" if first.len() >= 4 => {
            let x = sub(first[1], first[0]);
            let y = sub(first[3], first[0]);
            let centre = scale(add(first[0], first[2]), 0.5);
            frame(FrameKind::Plane, centre, cross(x, y), x)
        }
        "axis" if first.len() >= 2 => {
            let z = sub(first[1], first[0]);
            frame(FrameKind::Axis, scale(add(first[0], first[1]), 0.5), z, perpendicular(unit(z)))
        }
        "point" if first.len() >= 2 => frame(FrameKind::Point, scale(add(first[0], first[1]), 0.5), [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        "frame" | "connector" if lines.len() >= 3 && lines.iter().all(|l| l.len() >= 2) => {
            let origin = first[0];
            frame(FrameKind::Frame, origin, sub(lines[2][1], origin), sub(first[1], origin))
        }
        _ => None,
    }
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
