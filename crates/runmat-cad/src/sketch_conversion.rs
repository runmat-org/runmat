//! Converted sketch geometry (Use edges): model edges and faces carried onto a sketch's plane
//! along its normal, then read as the simplest sketch curve through them.
//!
//! Parts and assemblies share this module. An assembly projects the edges a component
//! publishes (`crate::assembly::projection`); a part reads its own bodies as they stand when the
//! sketch regenerates (`crate::regen`), and the Desktop's Use edges command writes the first
//! position from the same functions.
//!
//! - **Project**: an edge's polyline, projected.
//! - **Intersection**: where a face crosses the sketch plane. The face's triangles are cut by
//!   the plane (the zero set of the signed distance, interpolated along each triangle edge),
//!   and the cut points on a cylinder are moved onto the exact cylinder within the plane.
//! - **Silhouette**: where a curved face turns edge-on to the sketch normal, the zero set of
//!   `n · z` over the face's vertex normals (the smooth-surface contour). On a cylinder the
//!   contour is the exact pair of generator lines.
//! - **Offset on surface**: an edge moved across a planar face by a distance, into the face
//!   for a positive distance; a straight edge moves along the face's in-plane normal to it, a
//!   circular edge changes radius. Neighbouring offsets of one face meet at their corners.
//!
//! A polyline is read ([`fit`]) as a line, then a circle (closed) or an arc, then a spline, to
//! [`FIT`].

use crate::model::datum::Frame;
use crate::model::sketch::{PointExpr, SketchGeometry};
use crate::refs::history::{EdgeCurve, SurfaceKind};
use crate::regen::kernel::FaceMesh;
use crate::value::ValueExpr;

/// How far (millimetres) a projected point may sit from the curve read through it.
pub const FIT: f64 = 0.01;

/// The most points a projected spline keeps.
const SPLINE_POINTS: usize = 24;

/// Ends closer than this (metres) join into one chain.
pub const JOIN: f64 = 1e-6;

/// A point in model coordinates (metres), in `frame`'s plane (millimetres).
pub fn in_plane(frame: &Frame, p: [f64; 3]) -> [f64; 2] {
    let d = sub(p, frame.origin);
    [dot(d, frame.x) * 1000.0, dot(d, frame.y) * 1000.0]
}

fn mm(v: f64) -> ValueExpr {
    ValueExpr::mm((v * 1e4).round() / 1e4)
}

fn at(p: [f64; 2]) -> PointExpr {
    PointExpr::Coords {
        x: mm(p[0]),
        y: mm(p[1]),
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The circle through three points, if they aren't in a line: its center and radius.
fn circle_through(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let sq = |p: [f64; 2]| p[0] * p[0] + p[1] * p[1];
    let x = (sq(a) * (b[1] - c[1]) + sq(b) * (c[1] - a[1]) + sq(c) * (a[1] - b[1])) / d;
    let y = (sq(a) * (c[0] - b[0]) + sq(b) * (a[0] - c[0]) + sq(c) * (b[0] - a[0])) / d;
    Some(([x, y], dist([x, y], a)))
}

/// A sketch curve read from a polyline, in sketch millimetres.
#[derive(Clone, Debug, PartialEq)]
pub enum Curve {
    Line {
        start: [f64; 2],
        end: [f64; 2],
    },
    /// Counter-clockwise from `start` to `end`.
    Arc {
        center: [f64; 2],
        start: [f64; 2],
        end: [f64; 2],
    },
    Circle {
        center: [f64; 2],
        radius: f64,
    },
    Spline {
        points: Vec<[f64; 2]>,
        closed: bool,
    },
}

impl Curve {
    /// The curve as sketch geometry.
    pub fn geometry(&self) -> SketchGeometry {
        match self {
            Curve::Line { start, end } => SketchGeometry::Line {
                start: at(*start),
                end: at(*end),
            },
            Curve::Arc { center, start, end } => SketchGeometry::Arc {
                center: at(*center),
                start: at(*start),
                end: at(*end),
            },
            Curve::Circle { center, radius } => SketchGeometry::Circle {
                center: at(*center),
                radius: mm(*radius),
            },
            Curve::Spline { points, closed } => SketchGeometry::Spline {
                points: points.iter().copied().map(at).collect(),
                closed: *closed,
            },
        }
    }

    /// The open curve's two ends.
    fn ends(&self) -> Option<[[f64; 2]; 2]> {
        match self {
            Curve::Line { start, end } | Curve::Arc { start, end, .. } => Some([*start, *end]),
            Curve::Spline {
                points,
                closed: false,
            } => Some([*points.first()?, *points.last()?]),
            _ => None,
        }
    }

    fn set_end(&mut self, which: usize, p: [f64; 2]) {
        match self {
            Curve::Line { start, end } | Curve::Arc { start, end, .. } => {
                *(if which == 0 { start } else { end }) = p
            }
            _ => {}
        }
    }
}

/// The simplest curve through a projected polyline (millimetres): a line, a circle (closed), an
/// arc, else a spline. `None` when it projects to a point.
pub fn fit_curve(points: &[[f64; 2]]) -> Option<Curve> {
    let (&first, &last) = (points.first()?, points.last()?);
    let closed = points.len() > 2 && dist(first, last) < FIT;
    let far = points
        .iter()
        .copied()
        .max_by(|a, b| dist(first, *a).total_cmp(&dist(first, *b)))?;
    if dist(first, far) < FIT {
        return None;
    }
    // A line: every point on the chord.
    if !closed {
        let (dx, dy) = (last[0] - first[0], last[1] - first[1]);
        let len = dx.hypot(dy);
        if len > FIT
            && points
                .iter()
                .all(|p| ((p[0] - first[0]) * dy - (p[1] - first[1]) * dx).abs() / len < FIT)
        {
            return Some(Curve::Line {
                start: first,
                end: last,
            });
        }
    }
    // A circle or an arc: every point at one distance from the center.
    let mid = points[points.len() / 3];
    let third = points[(2 * points.len()) / 3];
    if let Some((center, radius)) = circle_through(first, mid, third) {
        if points
            .iter()
            .all(|p| (dist(*p, center) - radius).abs() < FIT.max(radius * 1e-4))
        {
            if closed {
                return Some(Curve::Circle { center, radius });
            }
            // Counter-clockwise from start to end: a polyline that turns left runs counter-clockwise.
            let cross = (mid[0] - first[0]) * (last[1] - first[1])
                - (mid[1] - first[1]) * (last[0] - first[0]);
            let (start, end) = if cross > 0.0 {
                (first, last)
            } else {
                (last, first)
            };
            return Some(Curve::Arc { center, start, end });
        }
    }
    let step = (points.len() as f64 / SPLINE_POINTS as f64).ceil().max(1.0) as usize;
    let mut kept: Vec<[f64; 2]> = points.iter().step_by(step).copied().collect();
    if closed {
        if kept.len() > 1 && dist(*kept.last()?, first) < FIT {
            kept.pop();
        }
    } else if kept.last() != Some(&last) {
        kept.push(last);
    }
    Some(Curve::Spline {
        points: kept,
        closed,
    })
}

/// [`fit_curve`] as sketch geometry.
pub fn fit(points: &[[f64; 2]]) -> Option<SketchGeometry> {
    fit_curve(points).map(|c| c.geometry())
}

// ------------------------------------------------------------------------------ vectors

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        a
    }
}
/// `v` without its component along unit `d`.
fn reject(v: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    sub(v, scale(d, dot(v, d)))
}

// ------------------------------------------------------------------------------ chains

fn near(a: [f64; 3], b: [f64; 3]) -> bool {
    norm(sub(a, b)) < JOIN
}

/// Polylines joined end to end, each turned as it needs to be.
pub fn join(mut pieces: Vec<Vec<[f64; 3]>>) -> Vec<Vec<[f64; 3]>> {
    let mut chains = Vec::new();
    while let Some(mut chain) = pieces.pop() {
        loop {
            let (head, tail) = (chain[0], chain[chain.len() - 1]);
            let Some(at) = pieces.iter().position(|p| {
                let (s, e) = (p[0], p[p.len() - 1]);
                near(s, tail) || near(e, tail) || near(s, head) || near(e, head)
            }) else {
                break;
            };
            let mut p = pieces.remove(at);
            let (s, e) = (p[0], p[p.len() - 1]);
            if near(s, tail) {
                chain.extend(p.drain(1..));
            } else if near(e, tail) {
                p.reverse();
                chain.extend(p.drain(1..));
            } else {
                if near(s, head) {
                    p.reverse();
                }
                p.pop();
                p.extend(chain);
                chain = p;
            }
        }
        chains.push(chain);
    }
    chains
}

/// Chains in a fixed order on the sketch plane, by where their centroid sits (x, then y, to
/// the micrometre), so `chain = 1` names the same piece on every regeneration.
fn ordered(mut chains: Vec<Vec<[f64; 3]>>, frame: &Frame) -> Vec<Vec<[f64; 3]>> {
    let key = |c: &Vec<[f64; 3]>| {
        let n = c.len().max(1) as f64;
        let s = c.iter().fold([0.0; 3], |a, p| add(a, *p));
        let p = in_plane(frame, scale(s, 1.0 / n));
        (
            (p[0] * 1000.0).round() as i64,
            (p[1] * 1000.0).round() as i64,
        )
    };
    chains.sort_by_key(key);
    chains
}

/// The zero set of `values` (one per mesh position) over a face's triangles, as chains.
fn contour(mesh: &FaceMesh, values: &[f64]) -> Vec<Vec<[f64; 3]>> {
    let mut segments = Vec::new();
    for t in &mesh.triangles {
        let mut cut = Vec::with_capacity(2);
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (va, vb) = (values[a as usize], values[b as usize]);
            if (va >= 0.0) != (vb >= 0.0) {
                let (pa, pb) = (mesh.positions[a as usize], mesh.positions[b as usize]);
                cut.push(add(pa, scale(sub(pb, pa), va / (va - vb))));
            }
        }
        if let [p, q] = cut.as_slice() {
            if !near(*p, *q) {
                segments.push(vec![*p, *q]);
            }
        }
    }
    join(segments)
}

/// The outward normal at each mesh position: the kernel's, else the area-weighted normals of
/// the triangles that share it.
fn vertex_normals(mesh: &FaceMesh) -> Vec<[f64; 3]> {
    if mesh.normals.len() == mesh.positions.len() {
        return mesh.normals.clone();
    }
    let mut out = vec![[0.0; 3]; mesh.positions.len()];
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
        let n = cross(sub(b, a), sub(c, a));
        for i in t {
            out[*i as usize] = add(out[*i as usize], n);
        }
    }
    out.into_iter().map(normalize).collect()
}

/// A face as the conversions read it: its triangles and, where the kernel knows it, its surface.
pub struct FaceSurface<'a> {
    pub mesh: &'a FaceMesh,
    pub surface: Option<SurfaceKind>,
}

/// Where `face` crosses the sketch plane, as chains in model coordinates (metres), in a fixed
/// order.
pub fn plane_section(face: &FaceSurface<'_>, frame: &Frame) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let z = normalize(frame.z);
    let values: Vec<f64> = face
        .mesh
        .positions
        .iter()
        .map(|p| dot(sub(*p, frame.origin), z))
        .collect();
    if values.iter().all(|v| v.abs() < 1e-9) {
        return Err("the face lies in the sketch plane; use Project on its edges".into());
    }
    let mut chains = contour(face.mesh, &values);
    if let Some(SurfaceKind::Cylinder {
        axis_point,
        axis_direction,
        radius,
    }) = face.surface
    {
        let d = normalize(axis_direction);
        for p in chains.iter_mut().flatten() {
            *p = onto_cylinder_in_plane(*p, axis_point, d, radius, z);
        }
    }
    if chains.is_empty() {
        return Err("the face doesn't cross the sketch plane".into());
    }
    Ok(ordered(chains, frame))
}

/// `q` (on the sketch plane, normal `z`) moved within the plane, along its radial direction
/// from the axis, onto the cylinder of `radius` about the axis through `p` along unit `d`.
fn onto_cylinder_in_plane(
    q: [f64; 3],
    p: [f64; 3],
    d: [f64; 3],
    radius: f64,
    z: [f64; 3],
) -> [f64; 3] {
    let w0 = reject(sub(q, p), d);
    let u = normalize(reject(w0, z));
    let w1 = reject(u, d);
    // |w0 + t w1|² = r²
    let (a, b, c) = (
        dot(w1, w1),
        2.0 * dot(w0, w1),
        dot(w0, w0) - radius * radius,
    );
    if a < 1e-18 {
        return q;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return q;
    }
    let s = disc.sqrt();
    let (t1, t2) = ((-b + s) / (2.0 * a), (-b - s) / (2.0 * a));
    let t = if t1.abs() < t2.abs() { t1 } else { t2 };
    add(q, scale(u, t))
}

/// Where `face` turns edge-on seen along the sketch normal, as chains in model coordinates
/// (metres), in a fixed order.
pub fn silhouette(face: &FaceSurface<'_>, frame: &Frame) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let z = normalize(frame.z);
    if let Some(SurfaceKind::Plane { .. }) = face.surface {
        return Err("a planar face has no silhouette; use Project on its edges".into());
    }
    let values: Vec<f64> = vertex_normals(face.mesh)
        .iter()
        .map(|n| dot(*n, z))
        .collect();
    let mut chains = contour(face.mesh, &values);
    if let Some(SurfaceKind::Cylinder {
        axis_point,
        axis_direction,
        radius,
    }) = face.surface
    {
        // A cylinder's outline is the pair of generator lines whose normals are square to z.
        let d = normalize(axis_direction);
        let s = normalize(cross(d, z));
        if norm(cross(d, z)) > 1e-9 {
            for p in chains.iter_mut().flatten() {
                let along = dot(sub(*p, axis_point), d);
                let side = if dot(sub(*p, axis_point), s) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                *p = add(add(axis_point, scale(d, along)), scale(s, side * radius));
            }
        }
    }
    if chains.is_empty() {
        return Err(
            "the face doesn't turn edge-on seen square to the sketch, so it has no silhouette \
             here"
                .into(),
        );
    }
    Ok(ordered(chains, frame))
}

/// Whether `p` (projected onto the face's plane) lies on one of the face's triangles.
fn on_face(mesh: &FaceMesh, normal: [f64; 3], p: [f64; 3]) -> bool {
    let n = normalize(normal);
    let u = normalize(if n[0].abs() < 0.9 {
        cross(n, [1.0, 0.0, 0.0])
    } else {
        cross(n, [0.0, 1.0, 0.0])
    });
    let v = cross(n, u);
    let flat = |q: [f64; 3]| [dot(q, u), dot(q, v)];
    let q = flat(p);
    mesh.triangles.iter().any(|t| {
        let [a, b, c] = t.map(|i| flat(mesh.positions[i as usize]));
        let side = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
            (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        };
        let (d1, d2, d3) = (side(a, b, q), side(b, c, q), side(c, a, q));
        let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(neg && pos)
    })
}

/// Edge `points` (metres, along curve `curve`) moved `distance` across planar `face`: into
/// the face for a positive distance.
pub fn offset_on_face(
    points: &[[f64; 3]],
    curve: EdgeCurve,
    face: &FaceSurface<'_>,
    distance: f64,
) -> Result<Vec<[f64; 3]>, String> {
    let Some(SurfaceKind::Plane { normal }) = face.surface else {
        return Err(
            "Offset on surface moves an edge across a planar face; pick a planar face beside it"
                .into(),
        );
    };
    if points.len() < 2 {
        return Err("the edge isn't drawn".into());
    }
    let n = normalize(normal);
    let probe = 1e-5_f64.max(distance.abs() * 1e-3);
    // The direction across the face at each point, square to the edge, still unsigned.
    let across: Vec<[f64; 3]> = match curve {
        EdgeCurve::Circle { center, normal, .. } => {
            let a = normalize(normal);
            points
                .iter()
                .map(|p| normalize(reject(sub(*p, center), a)))
                .collect()
        }
        _ => (0..points.len())
            .map(|i| {
                let a = points[i.saturating_sub(1)];
                let b = points[(i + 1).min(points.len() - 1)];
                normalize(cross(n, sub(b, a)))
            })
            .collect(),
    };
    let mid = points.len() / 2;
    let (m, s) = if points.len() == 2 {
        (scale(add(points[0], points[1]), 0.5), across[0])
    } else {
        (points[mid], across[mid])
    };
    let sign = if on_face(face.mesh, n, add(m, scale(s, probe))) {
        1.0
    } else if on_face(face.mesh, n, add(m, scale(s, -probe))) {
        -1.0
    } else {
        return Err("the edge isn't on the picked face".into());
    };
    Ok(points
        .iter()
        .zip(&across)
        .map(|(p, a)| add(*p, scale(*a, sign * distance)))
        .collect())
}

/// Where the carriers of two curves (a line's line, an arc's circle) cross, the crossing
/// nearest `near`.
fn carrier_crossing(a: &Curve, b: &Curve, near: [f64; 2]) -> Option<[f64; 2]> {
    enum C {
        L([f64; 2], [f64; 2]),
        R([f64; 2], f64),
    }
    let carrier = |c: &Curve| match c {
        Curve::Line { start, end } => Some(C::L(*start, *end)),
        Curve::Arc { center, start, .. } => Some(C::R(*center, dist(*center, *start))),
        _ => None,
    };
    let line_circle = |p: [f64; 2], q: [f64; 2], c: [f64; 2], r: f64| -> Vec<[f64; 2]> {
        let d = [q[0] - p[0], q[1] - p[1]];
        let f = [p[0] - c[0], p[1] - c[1]];
        let (a, b, k) = (
            d[0] * d[0] + d[1] * d[1],
            2.0 * (f[0] * d[0] + f[1] * d[1]),
            f[0] * f[0] + f[1] * f[1] - r * r,
        );
        let disc = b * b - 4.0 * a * k;
        if a < 1e-18 || disc < 0.0 {
            return Vec::new();
        }
        [
            (-b + disc.sqrt()) / (2.0 * a),
            (-b - disc.sqrt()) / (2.0 * a),
        ]
        .map(|t| [p[0] + t * d[0], p[1] + t * d[1]])
        .to_vec()
    };
    let points: Vec<[f64; 2]> = match (carrier(a)?, carrier(b)?) {
        (C::L(p, q), C::L(r, s)) => {
            let (d1, d2) = ([q[0] - p[0], q[1] - p[1]], [s[0] - r[0], s[1] - r[1]]);
            let den = d1[0] * d2[1] - d1[1] * d2[0];
            if den.abs() < 1e-12 {
                return None;
            }
            let t = ((r[0] - p[0]) * d2[1] - (r[1] - p[1]) * d2[0]) / den;
            vec![[p[0] + t * d1[0], p[1] + t * d1[1]]]
        }
        (C::L(p, q), C::R(c, r)) | (C::R(c, r), C::L(p, q)) => line_circle(p, q, c, r),
        (C::R(c1, r1), C::R(c2, r2)) => {
            let d = dist(c1, c2);
            if d < 1e-12 || d > r1 + r2 || d < (r1 - r2).abs() {
                return None;
            }
            let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
            let h = (r1 * r1 - a * a).max(0.0).sqrt();
            let m = [
                c1[0] + a * (c2[0] - c1[0]) / d,
                c1[1] + a * (c2[1] - c1[1]) / d,
            ];
            let o = [-(c2[1] - c1[1]) / d * h, (c2[0] - c1[0]) / d * h];
            vec![[m[0] + o[0], m[1] + o[1]], [m[0] - o[0], m[1] - o[1]]]
        }
    };
    points
        .into_iter()
        .min_by(|p, q| dist(*p, near).total_cmp(&dist(*q, near)))
}

/// Offsets of one face's edges, each with the ends of the edge it came from (sketch
/// millimetres), made to meet where their edges met: each pair of ends whose edges shared a
/// corner moves to where the two offset curves cross, nearest the corner.
pub fn join_offset_corners(offsets: &mut [(Curve, [[f64; 2]; 2])]) {
    for i in 0..offsets.len() {
        for j in i + 1..offsets.len() {
            for si in 0..2 {
                for sj in 0..2 {
                    let corner = offsets[i].1[si];
                    if dist(corner, offsets[j].1[sj]) > FIT {
                        continue;
                    }
                    let (Some(ei), Some(ej)) = (offsets[i].0.ends(), offsets[j].0.ends()) else {
                        continue;
                    };
                    let nearest = |ends: [[f64; 2]; 2]| {
                        if dist(ends[0], corner) <= dist(ends[1], corner) {
                            0
                        } else {
                            1
                        }
                    };
                    let (ki, kj) = (nearest(ei), nearest(ej));
                    if dist(ei[ki], ej[kj]) < FIT {
                        continue;
                    }
                    let mid = [(ei[ki][0] + ej[kj][0]) / 2.0, (ei[ki][1] + ej[kj][1]) / 2.0];
                    if let Some(p) = carrier_crossing(&offsets[i].0, &offsets[j].0, mid) {
                        offsets[i].0.set_end(ki, p);
                        offsets[j].0.set_end(kj, p);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straight_polyline_reads_as_a_line_and_a_round_one_as_an_arc_or_circle() {
        let line = fit(&[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]).unwrap();
        assert!(matches!(line, SketchGeometry::Line { .. }), "{line:?}");
        let ring: Vec<[f64; 2]> = (0..=36)
            .map(|i| {
                let t = i as f64 / 36.0 * std::f64::consts::TAU;
                [10.0 + 4.0 * t.cos(), 5.0 + 4.0 * t.sin()]
            })
            .collect();
        match fit(&ring).unwrap() {
            SketchGeometry::Circle { radius, .. } => assert_eq!(radius, ValueExpr::mm(4.0)),
            other => panic!("{other:?}"),
        }
        let half: Vec<[f64; 2]> = ring[..19].to_vec();
        assert!(matches!(fit(&half).unwrap(), SketchGeometry::Arc { .. }));
        let wiggle = [[0.0, 0.0], [3.0, 2.0], [6.0, -1.0], [9.0, 3.0], [12.0, 0.0]];
        assert!(matches!(
            fit(&wiggle).unwrap(),
            SketchGeometry::Spline { closed: false, .. }
        ));
        assert!(fit(&[[1.0, 1.0], [1.0, 1.0]]).is_none());
    }

    /// A cylinder of radius 5 mm about z, 20 mm tall, as a ring of 72 quads with exact normals.
    fn cylinder() -> FaceMesh {
        let n = 72;
        let mut mesh = FaceMesh::default();
        for i in 0..n {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            let (c, s) = (t.cos(), t.sin());
            mesh.positions.push([0.005 * c, 0.005 * s, 0.0]);
            mesh.positions.push([0.005 * c, 0.005 * s, 0.02]);
            mesh.normals.push([c, s, 0.0]);
            mesh.normals.push([c, s, 0.0]);
        }
        for i in 0..n as u32 {
            let (a, b) = (2 * i, 2 * ((i + 1) % n as u32));
            mesh.triangles.push([a, b, b + 1]);
            mesh.triangles.push([a, b + 1, a + 1]);
        }
        mesh
    }

    fn cylinder_surface() -> Option<SurfaceKind> {
        Some(SurfaceKind::Cylinder {
            axis_point: [0.0; 3],
            axis_direction: [0.0, 0.0, 1.0],
            radius: 0.005,
        })
    }

    /// A plane square to the cylinder cuts it in its exact circle; a plane along the axis sees
    /// its outline as the two generator lines 10 mm apart.
    #[test]
    fn a_cylinder_cuts_to_its_circle_and_outlines_as_two_lines() {
        let mesh = cylinder();
        let face = FaceSurface {
            mesh: &mesh,
            surface: cylinder_surface(),
        };
        let across = Frame {
            origin: [0.0, 0.0, 0.007],
            x: [1.0, 0.0, 0.0],
            y: [0.0, 1.0, 0.0],
            z: [0.0, 0.0, 1.0],
        };
        let chains = plane_section(&face, &across).unwrap();
        assert_eq!(chains.len(), 1);
        let local: Vec<[f64; 2]> = chains[0].iter().map(|p| in_plane(&across, *p)).collect();
        match fit_curve(&local).unwrap() {
            Curve::Circle { center, radius } => {
                assert!(dist(center, [0.0, 0.0]) < 1e-6, "{center:?}");
                assert!((radius - 5.0).abs() < 1e-6, "{radius}");
            }
            other => panic!("{other:?}"),
        }
        // Seen from the side (sketch on the zx plane), the outline is x = ±5 mm.
        let side = Frame {
            origin: [0.0; 3],
            x: [0.0, 0.0, 1.0],
            y: [1.0, 0.0, 0.0],
            z: [0.0, 1.0, 0.0],
        };
        let outline = silhouette(&face, &side).unwrap();
        assert_eq!(outline.len(), 2);
        for (chain, y) in outline.iter().zip([-5.0, 5.0]) {
            let local: Vec<[f64; 2]> = chain.iter().map(|p| in_plane(&side, *p)).collect();
            match fit_curve(&local).unwrap() {
                Curve::Line { start, end } => {
                    assert!(
                        (start[1] - y).abs() < 1e-6 && (end[1] - y).abs() < 1e-6,
                        "{start:?} {end:?}"
                    );
                    assert!((dist(start, end) - 20.0).abs() < 1e-6);
                }
                other => panic!("{other:?}"),
            }
        }
        let flat = FaceSurface {
            mesh: &mesh,
            surface: Some(SurfaceKind::Plane {
                normal: [0.0, 0.0, 1.0],
            }),
        };
        assert!(silhouette(&flat, &side).is_err());
    }

    /// The four edges of a 40 x 20 mm face, offset 3 mm into it, meet at their corners: a
    /// 34 x 14 mm rectangle.
    #[test]
    fn offsets_of_a_face_s_edges_meet_at_the_corners() {
        let mesh = FaceMesh {
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.04, 0.02, 0.0],
                [0.0, 0.02, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            normals: Vec::new(),
        };
        let face = FaceSurface {
            mesh: &mesh,
            surface: Some(SurfaceKind::Plane {
                normal: [0.0, 0.0, 1.0],
            }),
        };
        let frame = Frame {
            origin: [0.0; 3],
            x: [1.0, 0.0, 0.0],
            y: [0.0, 1.0, 0.0],
            z: [0.0, 0.0, 1.0],
        };
        let corners = [
            [0.0, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.04, 0.02, 0.0],
            [0.0, 0.02, 0.0],
        ];
        let mut offsets: Vec<(Curve, [[f64; 2]; 2])> = (0..4)
            .map(|i| {
                let edge = [corners[i], corners[(i + 1) % 4]];
                let moved = offset_on_face(&edge, EdgeCurve::Line, &face, 0.003).unwrap();
                let local: Vec<[f64; 2]> = moved.iter().map(|p| in_plane(&frame, *p)).collect();
                let ends = [in_plane(&frame, edge[0]), in_plane(&frame, edge[1])];
                (fit_curve(&local).unwrap(), ends)
            })
            .collect();
        join_offset_corners(&mut offsets);
        let expect = [
            ([3.0, 3.0], [37.0, 3.0]),
            ([37.0, 3.0], [37.0, 17.0]),
            ([37.0, 17.0], [3.0, 17.0]),
            ([3.0, 17.0], [3.0, 3.0]),
        ];
        for ((curve, _), (s, e)) in offsets.iter().zip(expect) {
            match curve {
                Curve::Line { start, end } => {
                    assert!(
                        dist(*start, s) < 1e-9 && dist(*end, e) < 1e-9,
                        "{start:?} {end:?}"
                    )
                }
                other => panic!("{other:?}"),
            }
        }
        // Off the face: refused, naming why.
        let away = [[0.0, 0.05, 0.0], [0.04, 0.05, 0.0]];
        assert!(offset_on_face(&away, EdgeCurve::Line, &face, 0.003).is_err());
    }
}
