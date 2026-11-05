//! Sampled tools: curved and off-axis shapes on the fake kernel's grid.
//!
//! A sampled tool is an implicit solid in its own local frame: `inside(p)` decides cell
//! occupancy at cell centres, and a facet the tool bounds is labelled by the tool surface the
//! boundary actually crosses. The crossing is found by bisection between the two cell centres
//! the facet separates, and the surface is the one the crossing point lies on. Labels are
//! therefore decided by the tool's own construction (its profile curves, caps, path segments
//! and sections), the way a B-rep kernel knows which face it built, never by the grid's order.
//!
//! Surfaces that land on a grid plane (an axis-aligned cap, a hole bottom) are exact planar
//! faces. Every other surface is a faceted face: a connected staircase of facets with one
//! label, joined into one face, with its edges chained into one edge per run between two faces.
//!
//! Shapes: prisms of line, arc and circle profiles on any frame; revolves (any angle, caps
//! when not full); translational sweeps along polyline paths (one side face per curve per path
//! segment, ordinals from the path); ruled lofts through parallel sections (one side face per
//! curve per span); holes (simple, counterbore, countersink) as revolves of their section; and
//! images of whole bodies under any rigid motion or reflection.

use super::exact::Exact;
use super::{FakeSolid, Rect};
use crate::refs::history::{HistorySource, OperandEntity, SurfaceKind};
use crate::refs::name::InstanceKey;
use crate::regen::kernel::{KernelError, ProfileSegment, SegmentKind, Transform};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

/// Uniform grid subdivisions a sampled tool adds across its extent on each axis.
pub(super) const RES: usize = 10;

#[derive(Clone, Debug)]
pub(super) enum Curve2 {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    /// From `a0` by `sweep` radians (counter-clockwise when positive).
    Arc {
        c: [f64; 2],
        r: f64,
        a0: f64,
        sweep: f64,
    },
    Circle {
        c: [f64; 2],
        r: f64,
    },
    /// A free-form curve (ellipse, spline) as a fine polyline from start to end.
    Poly {
        pts: Vec<[f64; 2]>,
    },
}

/// Polyline points per free-form profile curve.
const POLY_SAMPLES: usize = 64;

fn d2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    if l2 < 1e-30 {
        return d2(p, a);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0);
    d2(p, [a[0] + t * dx, a[1] + t * dy])
}

impl Curve2 {
    pub(super) fn from_segment(s: &ProfileSegment) -> Curve2 {
        match &s.kind {
            SegmentKind::Line => Curve2::Line {
                a: s.start,
                b: s.end,
            },
            SegmentKind::Circle { center, radius } => Curve2::Circle {
                c: *center,
                r: *radius,
            },
            SegmentKind::Arc { center, clockwise } => {
                let r = d2(s.start, *center);
                let a0 = (s.start[1] - center[1]).atan2(s.start[0] - center[0]);
                let a1 = (s.end[1] - center[1]).atan2(s.end[0] - center[0]);
                let mut sweep = a1 - a0;
                if *clockwise {
                    while sweep >= -1e-12 {
                        sweep -= TAU;
                    }
                } else {
                    while sweep <= 1e-12 {
                        sweep += TAU;
                    }
                }
                Curve2::Arc {
                    c: *center,
                    r,
                    a0,
                    sweep,
                }
            }
            SegmentKind::Ellipse { .. } | SegmentKind::Spline { .. } => Curve2::Poly {
                pts: s.polyline(POLY_SAMPLES),
            },
        }
    }

    pub(super) fn samples(&self) -> usize {
        match self {
            Curve2::Line { .. } => 1,
            Curve2::Arc { sweep, .. } => ((sweep.abs() / (PI / 24.0)).ceil() as usize).max(2),
            Curve2::Circle { .. } => 48,
            Curve2::Poly { pts } => pts.len() - 1,
        }
    }

    /// `n + 1` points from start to end (`n` from [`Curve2::samples`] or more).
    pub(super) fn polyline(&self, n: usize) -> Vec<[f64; 2]> {
        let n = n.max(1);
        if let Curve2::Poly { pts } = self {
            // Even steps along the polyline's own points.
            let m = pts.len() - 1;
            return (0..=n)
                .map(|i| {
                    let x = i as f64 * m as f64 / n as f64;
                    let k = (x.floor() as usize).min(m.saturating_sub(1));
                    let f = x - k as f64;
                    let (a, b) = (pts[k], pts[(k + 1).min(m)]);
                    [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]
                })
                .collect();
        }
        (0..=n)
            .map(|i| {
                let t = i as f64 / n as f64;
                match self {
                    Curve2::Line { a, b } => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])],
                    Curve2::Arc { c, r, a0, sweep } => {
                        let a = a0 + t * sweep;
                        [c[0] + r * a.cos(), c[1] + r * a.sin()]
                    }
                    Curve2::Circle { c, r } => {
                        let a = t * TAU;
                        [c[0] + r * a.cos(), c[1] + r * a.sin()]
                    }
                    Curve2::Poly { .. } => unreachable!("handled above"),
                }
            })
            .collect()
    }

    fn distance(&self, p: [f64; 2]) -> f64 {
        match self {
            Curve2::Line { a, b } => seg_dist(p, *a, *b),
            Curve2::Circle { c, r } => (d2(p, *c) - r).abs(),
            Curve2::Poly { pts } => pts
                .windows(2)
                .map(|w| seg_dist(p, w[0], w[1]))
                .fold(f64::INFINITY, f64::min),
            Curve2::Arc { c, r, a0, sweep } => {
                // As the counter-clockwise arc over the same points.
                let (a0, sweep) = if *sweep < 0.0 {
                    (a0 + sweep, &-sweep)
                } else {
                    (*a0, sweep)
                };
                let ang = (p[1] - c[1]).atan2(p[0] - c[0]);
                let mut rel = ang - a0;
                while rel < 0.0 {
                    rel += TAU;
                }
                while rel >= TAU {
                    rel -= TAU;
                }
                if rel <= *sweep {
                    (d2(p, *c) - r).abs()
                } else {
                    let s = [c[0] + r * a0.cos(), c[1] + r * a0.sin()];
                    let e = [c[0] + r * (a0 + sweep).cos(), c[1] + r * (a0 + sweep).sin()];
                    d2(p, s).min(d2(p, e))
                }
            }
        }
    }

    /// The direction of a straight curve, for exact-plane detection.
    fn line(&self) -> Option<([f64; 2], [f64; 2])> {
        match self {
            Curve2::Line { a, b } => Some((*a, *b)),
            _ => None,
        }
    }
}

/// Even-odd point-in-region over closed polylines.
fn inside_polys(polys: &[Vec<[f64; 2]>], q: [f64; 2]) -> bool {
    let mut inside = false;
    for l in polys {
        let n = l.len();
        for i in 0..n {
            let (a, b) = (l[i], l[(i + 1) % n]);
            if (a[1] > q[1]) != (b[1] > q[1]) {
                let x = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
                if q[0] < x {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// One closed polyline per loop from its curves' polylines (each curve sampled `n[i]` times).
fn loop_polys(loops: &[Vec<(Curve2, usize)>], counts: &[Vec<usize>]) -> Vec<Vec<[f64; 2]>> {
    loops
        .iter()
        .zip(counts)
        .map(|(l, ns)| {
            let mut pts = Vec::new();
            for ((c, _), n) in l.iter().zip(ns) {
                let p = c.polyline(*n);
                pts.extend_from_slice(&p[..p.len() - 1]);
            }
            pts
        })
        .collect()
}

/// A labelled tool surface.
#[derive(Clone, Debug)]
pub(super) struct Surf {
    pub label: HistorySource,
    /// Informative geometry in world coordinates.
    pub hint: SurfaceKind,
    /// The exact grid plane the surface lies on, when it lies on one.
    pub plane: Option<(usize, f64)>,
}

#[derive(Clone, Debug)]
pub(super) enum Shape {
    Prism {
        loops: Vec<Vec<(Curve2, usize)>>,
        polys: Vec<Vec<[f64; 2]>>,
        z0: f64,
        z1: f64,
        caps: [usize; 2],
    },
    Revolve {
        loops: Vec<Vec<(Curve2, usize)>>,
        polys: Vec<Vec<[f64; 2]>>,
        ap: [f64; 2],
        ad: [f64; 2],
        e1: [f64; 2],
        /// Right-handed about `ad` when positive.
        sense: f64,
        angle: f64,
        caps: Option<[usize; 2]>,
    },
    Sweep {
        loops: Vec<Vec<Curve2>>,
        polys: Vec<Vec<[f64; 2]>>,
        /// Path vertex offsets from the first vertex, local.
        offsets: Vec<[f64; 3]>,
        /// Side surface per loop, curve and path segment.
        sides: Vec<Vec<Vec<usize>>>,
        caps: [usize; 2],
    },
    Loft {
        z: Vec<f64>,
        /// Per section, loop, curve: the curve's polyline (equal counts across sections).
        curves: Vec<Vec<Vec<Vec<[f64; 2]>>>>,
        /// Side surface per loop, curve and span.
        sides: Vec<Vec<Vec<usize>>>,
        caps: [usize; 2],
    },
    /// A whole body under a rigid motion or reflection; surface `i` is face `i`.
    Image { solid: Arc<FakeSolid> },
    /// A prism whose caps follow end conditions (per profile point), with tapered sides and
    /// an optional thin wall.
    Extrusion(Box<Extrusion>),
}

/// Where an extrusion cap lies above a profile point (local z).
#[derive(Clone, Debug)]
pub(super) enum ZBound {
    At(f64),
    /// The nearest hit, along local z from `from` (`sense` ±1; 0: either way), of a ray
    /// against world facet rects; then `offset` further along `sense`.
    Hits {
        rects: Arc<Vec<(usize, f64, Rect)>>,
        from: f64,
        sense: f64,
        offset: f64,
    },
    /// `distance` past another bound (a blind cap measured from a face start).
    Past {
        base: Box<ZBound>,
        distance: f64,
    },
}

impl ZBound {
    fn eval(&self, q: [f64; 2], to_world: &Transform) -> Option<f64> {
        match self {
            ZBound::At(z) => Some(*z),
            ZBound::Hits {
                rects,
                from,
                sense,
                offset,
            } => {
                let o = to_world.apply([q[0], q[1], *from]);
                let d = to_world.apply_vector([0.0, 0.0, 1.0]);
                let mut best: Option<f64> = None;
                for (axis, off, r) in rects.iter() {
                    let da = d[*axis];
                    if da.abs() < 1e-12 {
                        continue;
                    }
                    let t = (off - o[*axis]) / da;
                    if *sense != 0.0 && t * sense <= 1e-9 {
                        continue;
                    }
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    if !(0..3)
                        .all(|a| a == *axis || (p[a] >= r.lo[a] - 1e-9 && p[a] <= r.hi[a] + 1e-9))
                    {
                        continue;
                    }
                    if best.map(|b| t.abs() < b.abs()).unwrap_or(true) {
                        best = Some(t);
                    }
                }
                best.map(|t| {
                    let s = if *sense == 0.0 { 1.0 } else { *sense };
                    from + t + s * offset
                })
            }
            ZBound::Past { base, distance } => base.eval(q, to_world).map(|z| z + distance),
        }
    }
    fn fixed(&self) -> Option<f64> {
        match self {
            ZBound::At(z) => Some(*z),
            _ => None,
        }
    }
}

/// A thin wall: inside, outside, and the cap thickness with its two inner cap surfaces
/// (surface indices, or the labels they are built from).
pub(super) type ThinBand<S> = (f64, f64, Option<(f64, [S; 2])>);

#[derive(Clone, Debug)]
pub(super) struct Extrusion {
    /// Per loop and curve: the curve, its outer side surface and (thin) its inner one.
    loops: Vec<Vec<(Curve2, usize, Option<usize>)>>,
    polys: Vec<Vec<[f64; 2]>>,
    start: ZBound,
    end: ZBound,
    caps: [usize; 2],
    /// +1 when the end cap lies towards +z of the start.
    end_sign: f64,
    /// Draft: hinge height, tangent on the end side, on the start side.
    draft: Option<(f64, f64, f64)>,
    /// Thin wall: inside, outside, and cap thickness with the inner cap surfaces.
    thin: Option<ThinBand<usize>>,
}

impl Extrusion {
    fn taper(&self, z: f64) -> f64 {
        match self.draft {
            None => 0.0,
            Some((h, te, ts)) => {
                let t = if (z - h) * self.end_sign >= 0.0 {
                    te
                } else {
                    ts
                };
                (z - h).abs() * t
            }
        }
    }
    fn signed(&self, q: [f64; 2]) -> (f64, bool) {
        let inside = inside_polys(&self.polys, q);
        let d = self
            .loops
            .iter()
            .flatten()
            .map(|(c, _, _)| c.distance(q))
            .fold(f64::INFINITY, f64::min);
        (if inside { d } else { -d }, inside)
    }
    fn band(&self) -> (f64, f64) {
        match self.thin {
            Some((i, o, _)) => (o, i),
            None => (0.0, f64::INFINITY),
        }
    }
    fn caps_at(&self, q: [f64; 2], to_world: &Transform) -> Option<(f64, f64)> {
        Some((self.start.eval(q, to_world)?, self.end.eval(q, to_world)?))
    }
    fn inside(&self, p: [f64; 3], to_world: &Transform) -> bool {
        let q = [p[0], p[1]];
        let Some((zs, ze)) = self.caps_at(q, to_world) else {
            return false;
        };
        let (lo, hi) = (zs.min(ze), zs.max(ze));
        if p[2] <= lo || p[2] >= hi {
            return false;
        }
        let e = self.signed(q).0 - self.taper(p[2]);
        let (outside, inside) = self.band();
        if e > -outside && e < inside {
            return true;
        }
        if let Some((_, _, Some((cap, _)))) = self.thin {
            return e > -outside && (p[2] - lo < cap || hi - p[2] < cap);
        }
        false
    }
    fn distances(&self, p: [f64; 3], to_world: &Transform, out: &mut [f64]) {
        let q = [p[0], p[1]];
        let Some((zs, ze)) = self.caps_at(q, to_world) else {
            return;
        };
        let (lo, hi) = (zs.min(ze), zs.max(ze));
        let zr = (lo - p[2]).max(p[2] - hi).max(0.0);
        let tp = self.taper(p[2]);
        let (_, inside_profile) = self.signed(q);
        let (outside, inside) = self.band();
        for (c, o, inn) in self.loops.iter().flatten() {
            let d = c.distance(q);
            let sd = if inside_profile { d } else { -d };
            out[*o] = out[*o].min((sd - tp + outside).abs() + zr);
            if let Some(k) = inn {
                out[*k] = out[*k].min((sd - tp - inside).abs() + zr);
            }
        }
        let e = self.signed(q).0 - tp;
        let off_band = (-outside - e)
            .max(if inside.is_finite() { e - inside } else { 0.0 })
            .max(0.0);
        let cap_band = (-outside - e).max(0.0);
        out[self.caps[0]] = out[self.caps[0]].min((p[2] - zs).abs() + cap_band.min(off_band));
        out[self.caps[1]] = out[self.caps[1]].min((p[2] - ze).abs() + cap_band.min(off_band));
        if let Some((_, _, Some((cap, [a, b])))) = self.thin {
            let s = if ze >= zs { 1.0 } else { -1.0 };
            let hollow = (inside - e).max(0.0);
            out[a] = out[a].min((p[2] - (zs + s * cap)).abs() + hollow);
            out[b] = out[b].min((p[2] - (ze - s * cap)).abs() + hollow);
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Sampled {
    /// World to shape-local.
    pub to_local: Transform,
    pub shape: Shape,
    pub surfaces: Vec<Surf>,
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    pub exact: Vec<(usize, f64)>,
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// The world axis a vector lies along, if any.
fn axis_along(v: [f64; 3]) -> Option<usize> {
    let n = norm3(v);
    if n < 1e-12 {
        return None;
    }
    (0..3).find(|&a| (v[a].abs() / n - 1.0).abs() < 1e-9)
}

fn rect_dist(p: [f64; 3], r: &Rect) -> f64 {
    let mut s = 0.0;
    for a in 0..3 {
        let d = if p[a] < r.lo[a] {
            r.lo[a] - p[a]
        } else if p[a] > r.hi[a] {
            p[a] - r.hi[a]
        } else {
            0.0
        };
        s += d * d;
    }
    s.sqrt()
}

/// How far an angle lies outside `[0, angle]` (radians, both measured the same way).
fn ang_out(theta: f64, angle: f64) -> f64 {
    if theta <= angle {
        0.0
    } else {
        (theta - angle).min(TAU - theta)
    }
}

impl Sampled {
    pub(super) fn inside(&self, w: [f64; 3]) -> bool {
        let p = self.to_local.apply(w);
        match &self.shape {
            Shape::Prism { polys, z0, z1, .. } => {
                p[2] > *z0 && p[2] < *z1 && inside_polys(polys, [p[0], p[1]])
            }
            Shape::Revolve { .. } => {
                let (q, theta, _, angle) = self.revolve_coords(p);
                (angle >= TAU - 1e-9 || theta <= angle) && self.revolve_in(q)
            }
            Shape::Sweep { polys, offsets, .. } => (0..offsets.len() - 1).any(|j| {
                let (s, q) = sweep_coords(offsets, j, p);
                (0.0..=1.0).contains(&s) && inside_polys(polys, q)
            }),
            Shape::Loft { z, curves, .. } => {
                if p[2] <= z[0] || p[2] >= z[z.len() - 1] {
                    return false;
                }
                let polys = loft_polys(z, curves, p[2]);
                inside_polys(&polys, [p[0], p[1]])
            }
            Shape::Image { solid } => solid.contains(p),
            Shape::Extrusion(x) => x.inside(p, &self.to_local.inverse()),
        }
    }

    /// A profile curve lying on the revolve axis sweeps nothing.
    fn on_axis(&self, c: &Curve2) -> bool {
        let Shape::Revolve { ap, ad, .. } = &self.shape else {
            return false;
        };
        let off = |p: [f64; 2]| {
            let rel = [p[0] - ap[0], p[1] - ap[1]];
            (rel[0] * ad[1] - rel[1] * ad[0]).abs()
        };
        match c.line() {
            Some((a, b)) => off(a) < 1e-9 && off(b) < 1e-9,
            None => false,
        }
    }

    fn revolve_in(&self, q: [f64; 2]) -> bool {
        match &self.shape {
            Shape::Revolve { polys, .. } => inside_polys(polys, q),
            _ => false,
        }
    }

    /// Profile-plane point, angle from the profile half-plane (0..2π), radius, sweep angle.
    fn revolve_coords(&self, p: [f64; 3]) -> ([f64; 2], f64, f64, f64) {
        let Shape::Revolve {
            ap,
            ad,
            e1,
            sense,
            angle,
            ..
        } = &self.shape
        else {
            unreachable!()
        };
        let ap3 = [ap[0], ap[1], 0.0];
        let ad3 = [ad[0], ad[1], 0.0];
        let e13 = [e1[0], e1[1], 0.0];
        // e2 = ad × e1, flipped for a negative sense.
        let e23 = [
            sense * (ad3[1] * e13[2] - ad3[2] * e13[1]),
            sense * (ad3[2] * e13[0] - ad3[0] * e13[2]),
            sense * (ad3[0] * e13[1] - ad3[1] * e13[0]),
        ];
        let d = sub3(p, ap3);
        let h = dot3(d, ad3);
        let rv = [d[0] - h * ad3[0], d[1] - h * ad3[1], d[2] - h * ad3[2]];
        let r = norm3(rv);
        let mut theta = dot3(rv, e23).atan2(dot3(rv, e13));
        if theta < 0.0 {
            theta += TAU;
        }
        let q = [ap[0] + h * ad[0] + r * e1[0], ap[1] + h * ad[1] + r * e1[1]];
        (q, theta, r, *angle)
    }

    /// Distance from local `p` to each surface; the least is the surface `p` lies on.
    fn surface_distances(&self, p: [f64; 3]) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; self.surfaces.len()];
        match &self.shape {
            Shape::Prism {
                loops,
                polys,
                z0,
                z1,
                caps,
            } => {
                let q = [p[0], p[1]];
                let zr = (z0 - p[2]).max(p[2] - z1).max(0.0);
                let mut edge = f64::INFINITY;
                for l in loops {
                    for (c, si) in l {
                        let d = c.distance(q);
                        edge = edge.min(d);
                        out[*si] = out[*si].min(d + zr);
                    }
                }
                let outside = if inside_polys(polys, q) { 0.0 } else { edge };
                out[caps[0]] = out[caps[0]].min((p[2] - z0).abs() + outside);
                out[caps[1]] = out[caps[1]].min((p[2] - z1).abs() + outside);
            }
            Shape::Revolve { loops, caps, .. } => {
                let (q, theta, r, angle) = self.revolve_coords(p);
                let full = angle >= TAU - 1e-9;
                let ar = if full { 0.0 } else { r * ang_out(theta, angle) };
                let mut edge = f64::INFINITY;
                for l in loops {
                    for (c, si) in l {
                        if self.on_axis(c) {
                            continue;
                        }
                        let d = c.distance(q);
                        edge = edge.min(d);
                        out[*si] = out[*si].min(d + ar);
                    }
                }
                if let Some([s, e]) = caps {
                    let outside = if self.revolve_in(q) { 0.0 } else { edge };
                    let to_start = theta.min(TAU - theta);
                    let to_end = (theta - angle).abs().min(TAU - (theta - angle).abs());
                    out[*s] = out[*s].min(r * to_start.min(PI / 2.0).sin() + outside);
                    out[*e] = out[*e].min(r * to_end.min(PI / 2.0).sin() + outside);
                }
            }
            Shape::Sweep {
                loops,
                polys,
                offsets,
                sides,
                caps,
            } => {
                let n = offsets.len() - 1;
                for j in 0..n {
                    let (s, q) = sweep_coords(offsets, j, p);
                    let len = norm3(sub3(offsets[j + 1], offsets[j]));
                    let sr = (-s).max(s - 1.0).max(0.0) * len;
                    let mut edge = f64::INFINITY;
                    for (li, l) in loops.iter().enumerate() {
                        for (ci, c) in l.iter().enumerate() {
                            let d = c.distance(q);
                            edge = edge.min(d);
                            let si = sides[li][ci][j];
                            out[si] = out[si].min(d + sr);
                        }
                    }
                    let outside = if inside_polys(polys, q) { 0.0 } else { edge };
                    if j == 0 {
                        out[caps[0]] = out[caps[0]].min(s.abs() * len + outside);
                    }
                    if j == n - 1 {
                        out[caps[1]] = out[caps[1]].min((s - 1.0).abs() * len + outside);
                    }
                }
            }
            Shape::Loft {
                z,
                curves,
                sides,
                caps,
            } => {
                let zz = p[2].clamp(z[0], z[z.len() - 1]);
                let k = span_of(z, zz);
                let polys = loft_polys(z, curves, zz);
                let q = [p[0], p[1]];
                let mut edge = f64::INFINITY;
                let t = (zz - z[k]) / (z[k + 1] - z[k]);
                for (li, l) in curves[k].iter().enumerate() {
                    for (ci, c0) in l.iter().enumerate() {
                        let c1 = &curves[k + 1][li][ci];
                        let pts: Vec<[f64; 2]> = c0
                            .iter()
                            .zip(c1)
                            .map(|(a, b)| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
                            .collect();
                        let mut d = f64::INFINITY;
                        for w in pts.windows(2) {
                            d = d.min(seg_dist(q, w[0], w[1]));
                        }
                        edge = edge.min(d);
                        let zr = (p[2] - zz).abs();
                        for (span, si) in sides[li][ci].iter().enumerate() {
                            let gap = if span == k {
                                0.0
                            } else if span < k {
                                zz - z[span + 1]
                            } else {
                                z[span] - zz
                            };
                            out[*si] = out[*si].min(d + zr + gap.max(0.0));
                        }
                    }
                }
                let outside = if inside_polys(&polys, q) { 0.0 } else { edge };
                out[caps[0]] = out[caps[0]].min((p[2] - z[0]).abs() + outside);
                out[caps[1]] = out[caps[1]].min((p[2] - z[z.len() - 1]).abs() + outside);
            }
            Shape::Image { solid } => {
                for (i, f) in solid.faces.iter().enumerate() {
                    for r in &f.rects {
                        out[i] = out[i].min(rect_dist(p, r));
                    }
                }
            }
            Shape::Extrusion(x) => x.distances(p, &self.to_local.inverse(), &mut out),
        }
        out
    }

    /// The label of the surface between `a` and `b` (world cell centres on either side of a
    /// facet), when this tool's boundary runs between them.
    pub(super) fn crossing(&self, a: [f64; 3], b: [f64; 3]) -> Option<usize> {
        let (ia, ib) = (self.inside(a), self.inside(b));
        if ia == ib {
            return None;
        }
        let (mut lo, mut hi) = if ia { (a, b) } else { (b, a) };
        for _ in 0..30 {
            let m = [
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                (lo[2] + hi[2]) / 2.0,
            ];
            if self.inside(m) {
                lo = m;
            } else {
                hi = m;
            }
        }
        let x = self.to_local.apply([
            (lo[0] + hi[0]) / 2.0,
            (lo[1] + hi[1]) / 2.0,
            (lo[2] + hi[2]) / 2.0,
        ]);
        let d = self.surface_distances(x);
        let mut best = None;
        let mut bd = f64::INFINITY;
        for (i, v) in d.iter().enumerate() {
            if *v < bd - 1e-12 {
                bd = *v;
                best = Some(i);
            }
        }
        best
    }

    pub(super) fn breaks(&self, out: &mut [Vec<f64>; 3]) {
        for a in 0..3 {
            out[a].push(self.lo[a]);
            out[a].push(self.hi[a]);
            for k in 1..RES {
                out[a].push(self.lo[a] + (self.hi[a] - self.lo[a]) * k as f64 / RES as f64);
            }
        }
        for (a, c) in &self.exact {
            out[*a].push(*c);
        }
        if let Shape::Extrusion(x) = &self.shape {
            if x.draft.is_some() {
                self.draft_breaks(x, out);
            }
        }
    }

    /// Breaks along a drafted extrusion's tapered sides. Each of `RES` slices of the
    /// extrusion's length gets breaks where the straight sides sit at the slice's middle, so
    /// cell centres sample the taper by the midpoint rule however small the taper is beside
    /// the uniform breaks.
    fn draft_breaks(&self, x: &Extrusion, out: &mut [Vec<f64>; 3]) {
        let world = self.to_local.inverse();
        let (mut zlo, mut zhi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..8 {
            let c = [
                if i & 1 == 0 { self.lo[0] } else { self.hi[0] },
                if i & 2 == 0 { self.lo[1] } else { self.hi[1] },
                if i & 4 == 0 { self.lo[2] } else { self.hi[2] },
            ];
            let z = self.to_local.apply(c)[2];
            zlo = zlo.min(z);
            zhi = zhi.max(z);
        }
        let (outside, inside) = x.band();
        let bands: Vec<f64> = [-outside, inside]
            .into_iter()
            .filter(|b| b.is_finite())
            .collect();
        let mut push = |p: [f64; 3]| {
            let w = world.apply(p);
            for a in 0..3 {
                out[a].push(w[a]);
            }
        };
        for (c, _, _) in x.loops.iter().flatten() {
            let Some((a, b)) = c.line() else {
                continue;
            };
            let d = [b[0] - a[0], b[1] - a[1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if l <= 1e-300 {
                continue;
            }
            // The normal into the profile region.
            let mut n = [-d[1] / l, d[0] / l];
            let m = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let probe = [m[0] + n[0] * l * 1e-6, m[1] + n[1] * l * 1e-6];
            if !inside_polys(&x.polys, probe) {
                n = [-n[0], -n[1]];
            }
            for k in 0..=RES {
                let z = zlo + (zhi - zlo) * k as f64 / RES as f64;
                push([a[0], a[1], z]);
                if k == RES {
                    break;
                }
                let zm = zlo + (zhi - zlo) * (k as f64 + 0.5) / RES as f64;
                let t = x.taper(zm);
                for band in &bands {
                    let o = t + band;
                    // At the slice's own start height: no break inside the slice.
                    for q in [a, b] {
                        push([q[0] + n[0] * o, q[1] + n[1] * o, z]);
                    }
                }
            }
        }
    }

    pub(super) fn map_labels(&mut self, fix: &mut dyn FnMut(&mut HistorySource)) {
        for s in &mut self.surfaces {
            fix(&mut s.label);
        }
    }

    pub(super) fn relabel(&mut self, seed: u32, instance: &InstanceKey) {
        for s in &mut self.surfaces {
            if let HistorySource::Generated {
                seed: sd,
                instance: inst,
                ..
            } = &mut s.label
            {
                *sd = Some(seed);
                *inst = Some(instance.clone());
            }
        }
    }

    /// The same tool moved by `t` (applied after the tool's own placement).
    pub(super) fn transformed(&self, t: &Transform) -> Sampled {
        let mut out = self.clone();
        out.to_local = self.to_local.compose(&t.inverse());
        let (lo, hi) = moved_box(self.lo, self.hi, t);
        out.lo = lo;
        out.hi = hi;
        out.exact = Vec::new();
        for s in &mut out.surfaces {
            s.hint = move_hint(&s.hint, t);
            s.plane = None;
        }
        out.recompute_exact();
        out
    }

    /// Exact planes: surfaces whose plane is normal to a world axis. Found from the hint's
    /// normal and a point on the surface.
    fn recompute_exact(&mut self) {
        let world = self.to_local.inverse();
        let mut exact = Vec::new();
        let anchors = self.plane_anchors();
        for (i, s) in self.surfaces.iter_mut().enumerate() {
            if let (SurfaceKind::Plane { normal }, Some(anchor)) = (s.hint, anchors[i]) {
                if let Some(a) = axis_along(normal) {
                    let w = world.apply(anchor);
                    s.plane = Some((a, w[a]));
                    exact.push((a, w[a]));
                    continue;
                }
            }
            s.plane = None;
        }
        self.exact = exact;
    }

    /// The exact surface of surface `i`, in world coordinates, where the shape has a chart
    /// for it (see [`super::exact`]). Lofts and drafted or thin extrusion walls have none.
    pub(super) fn exact_of(&self, i: usize) -> Option<Exact> {
        let world = self.to_local.inverse();
        let plane_at = |o: [f64; 3], e1: [f64; 3], e2: [f64; 3]| {
            Some(Exact::plane(
                world.apply(o),
                world.apply_vector(e1),
                world.apply_vector(e2),
            ))
        };
        let (x, y) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        match &self.shape {
            Shape::Prism {
                loops,
                z0,
                z1,
                caps,
                ..
            } => {
                if i == caps[0] {
                    return plane_at([0.0, 0.0, *z0], x, y);
                }
                if i == caps[1] {
                    return plane_at([0.0, 0.0, *z1], x, y);
                }
                let (c, _) = loops.iter().flatten().find(|(_, si)| *si == i)?;
                Exact::extruded(world, c.clone(), [0.0; 3], [0.0, 0.0, 1.0])
            }
            Shape::Revolve {
                loops,
                ap,
                ad,
                e1,
                sense,
                angle,
                caps,
                ..
            } => {
                if let Some([s, e]) = caps {
                    if i == *s || i == *e {
                        let th = if i == *s { 0.0 } else { *angle };
                        let ad3 = [ad[0], ad[1], 0.0];
                        let e13 = [e1[0], e1[1], 0.0];
                        let e23 = {
                            let c = [
                                ad3[1] * e13[2] - ad3[2] * e13[1],
                                ad3[2] * e13[0] - ad3[0] * e13[2],
                                ad3[0] * e13[1] - ad3[1] * e13[0],
                            ];
                            [c[0] * sense, c[1] * sense, c[2] * sense]
                        };
                        let radial = [
                            e13[0] * th.cos() + e23[0] * th.sin(),
                            e13[1] * th.cos() + e23[1] * th.sin(),
                            e13[2] * th.cos() + e23[2] * th.sin(),
                        ];
                        return plane_at([ap[0], ap[1], 0.0], ad3, radial);
                    }
                }
                let (c, _) = loops
                    .iter()
                    .flatten()
                    .find(|(c, si)| *si == i && !self.on_axis(c))?;
                Exact::revolved(world, c.clone(), *ap, *ad, *e1, *sense)
            }
            Shape::Sweep {
                loops,
                offsets,
                sides,
                caps,
                ..
            } => {
                if i == caps[0] {
                    return plane_at([0.0; 3], x, y);
                }
                if i == caps[1] {
                    return plane_at(offsets[offsets.len() - 1], x, y);
                }
                for (li, l) in loops.iter().enumerate() {
                    for (ci, c) in l.iter().enumerate() {
                        if let Some(j) = sides[li][ci].iter().position(|s| *s == i) {
                            // One side surface per path segment; a curve reused across
                            // segments of one direction is still one extrusion.
                            let segs: Vec<usize> = sides[li][ci]
                                .iter()
                                .enumerate()
                                .filter(|(_, s)| **s == i)
                                .map(|(k, _)| k)
                                .collect();
                            let d = sub3(offsets[j + 1], offsets[j]);
                            let same = segs.iter().all(|&k| {
                                let e = sub3(offsets[k + 1], offsets[k]);
                                let (a, b) = (norm3(d), norm3(e));
                                a > 1e-300
                                    && b > 1e-300
                                    && (dot3(d, e) / (a * b) - 1.0).abs() < 1e-9
                            });
                            if !same {
                                return None;
                            }
                            return Exact::extruded(world, c.clone(), offsets[j], d);
                        }
                    }
                }
                None
            }
            Shape::Loft { .. } => None,
            Shape::Image { solid } => {
                let f = solid.faces.get(i)?;
                let local = match (&f.exact, f.plane) {
                    (Some(e), _) => (**e).clone(),
                    (None, Some(p)) => super::mesh::grid_plane(&p),
                    _ => return None,
                };
                Some(local.transformed(&world))
            }
            Shape::Extrusion(x) => {
                if x.draft.is_some() {
                    return None;
                }
                if i == x.caps[0] {
                    return x
                        .start
                        .fixed()
                        .and_then(|z| plane_at([0.0, 0.0, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
                }
                if i == x.caps[1] {
                    return x
                        .end
                        .fixed()
                        .and_then(|z| plane_at([0.0, 0.0, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
                }
                let outside = x.thin.map(|t| t.1).unwrap_or(0.0);
                let (c, _, _) = x.loops.iter().flatten().find(|(_, o, _)| *o == i)?;
                if outside.abs() > 1e-12 {
                    return None;
                }
                Exact::extruded(world, c.clone(), [0.0; 3], [0.0, 0.0, 1.0])
            }
        }
    }

    /// A local point on each planar surface (where the shape knows one).
    fn plane_anchors(&self) -> Vec<Option<[f64; 3]>> {
        let mut out = vec![None; self.surfaces.len()];
        match &self.shape {
            Shape::Prism {
                loops,
                z0,
                z1,
                caps,
                ..
            } => {
                for l in loops {
                    for (c, si) in l {
                        if let Some((a, _)) = c.line() {
                            out[*si] = Some([a[0], a[1], *z0]);
                        }
                    }
                }
                out[caps[0]] = Some([0.0, 0.0, *z0]);
                out[caps[1]] = Some([0.0, 0.0, *z1]);
            }
            Shape::Revolve {
                loops, ap, caps, ..
            } => {
                // A profile line perpendicular to the axis sweeps a plane through its points.
                for l in loops {
                    for (c, si) in l {
                        if let Some((a, _)) = c.line() {
                            out[*si] = Some([a[0], a[1], 0.0]);
                        }
                    }
                }
                if let Some([s, e]) = caps {
                    let a = [ap[0], ap[1], 0.0];
                    out[*s] = Some(a);
                    out[*e] = Some(a);
                }
            }
            Shape::Sweep {
                offsets,
                caps,
                loops,
                sides,
                ..
            } => {
                out[caps[0]] = Some([0.0, 0.0, 0.0]);
                out[caps[1]] = Some(offsets[offsets.len() - 1]);
                for (li, l) in loops.iter().enumerate() {
                    for (ci, c) in l.iter().enumerate() {
                        if let Some((a, _)) = c.line() {
                            for (j, si) in sides[li][ci].iter().enumerate() {
                                let o = offsets[j];
                                out[*si] = Some([a[0] + o[0], a[1] + o[1], o[2]]);
                            }
                        }
                    }
                }
            }
            Shape::Loft { z, caps, .. } => {
                out[caps[0]] = Some([0.0, 0.0, z[0]]);
                out[caps[1]] = Some([0.0, 0.0, z[z.len() - 1]]);
            }
            Shape::Image { solid } => {
                for (i, f) in solid.faces.iter().enumerate() {
                    if let Some(r) = f.rects.first() {
                        out[i] = Some(r.lo);
                    }
                }
            }
            Shape::Extrusion(x) => {
                let hinge = x.draft.map(|d| d.0);
                for (c, o, inn) in x.loops.iter().flatten() {
                    if let Some((a, _)) = c.line() {
                        let z = hinge.or(x.start.fixed()).unwrap_or(0.0);
                        out[*o] = Some([a[0], a[1], z]);
                        if let (Some(k), Some((i, _, _)), None) = (inn, x.thin, x.draft) {
                            if let Some(b) = inner_line(c, i) {
                                out[*k] = Some([b[0], b[1], z]);
                            }
                        }
                    }
                }
                if let Some(z) = x.start.fixed() {
                    out[x.caps[0]] = Some([0.0, 0.0, z]);
                }
                if let Some(z) = x.end.fixed() {
                    out[x.caps[1]] = Some([0.0, 0.0, z]);
                }
            }
        }
        out
    }
}

fn sweep_coords(offsets: &[[f64; 3]], j: usize, p: [f64; 3]) -> (f64, [f64; 2]) {
    let o = offsets[j];
    let d = sub3(offsets[j + 1], o);
    let rel = sub3(p, o);
    let s = if d[2].abs() > 1e-12 {
        rel[2] / d[2]
    } else {
        -1.0
    };
    (s, [rel[0] - s * d[0], rel[1] - s * d[1]])
}

fn span_of(z: &[f64], v: f64) -> usize {
    let mut k = 0;
    while k + 2 < z.len() && v > z[k + 1] {
        k += 1;
    }
    k
}

fn loft_polys(z: &[f64], curves: &[Vec<Vec<Vec<[f64; 2]>>>], v: f64) -> Vec<Vec<[f64; 2]>> {
    let k = span_of(z, v);
    let t = ((v - z[k]) / (z[k + 1] - z[k])).clamp(0.0, 1.0);
    curves[k]
        .iter()
        .zip(&curves[k + 1])
        .map(|(l0, l1)| {
            let mut pts = Vec::new();
            for (c0, c1) in l0.iter().zip(l1) {
                for (a, b) in c0.iter().zip(c1).take(c0.len() - 1) {
                    pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
            }
            pts
        })
        .collect()
}

fn moved_box(lo: [f64; 3], hi: [f64; 3], t: &Transform) -> ([f64; 3], [f64; 3]) {
    let mut nlo = [f64::INFINITY; 3];
    let mut nhi = [f64::NEG_INFINITY; 3];
    for i in 0..8 {
        let c = [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ];
        let w = t.apply(c);
        for a in 0..3 {
            nlo[a] = nlo[a].min(w[a]);
            nhi[a] = nhi[a].max(w[a]);
        }
    }
    (nlo, nhi)
}

pub(super) fn move_hint(h: &SurfaceKind, t: &Transform) -> SurfaceKind {
    match *h {
        SurfaceKind::Plane { normal } => SurfaceKind::Plane {
            normal: t.apply_vector(normal),
        },
        SurfaceKind::Cylinder {
            axis_point,
            axis_direction,
            radius,
        } => SurfaceKind::Cylinder {
            axis_point: t.apply(axis_point),
            axis_direction: t.apply_vector(axis_direction),
            radius,
        },
        SurfaceKind::Other => SurfaceKind::Other,
    }
}

/// The box around world points.
fn bbox(points: impl IntoIterator<Item = [f64; 3]>) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

fn unsupported(what: &str) -> KernelError {
    KernelError::NotSupported {
        what: format!("fake kernel: {}", what),
    }
}

fn plane_hint(local_normal: [f64; 3], to_world: &Transform) -> SurfaceKind {
    SurfaceKind::Plane {
        normal: to_world.apply_vector(local_normal),
    }
}

/// Side-face hint of a straight profile curve in the profile plane, swept along local z.
fn side_hint(c: &Curve2, to_world: &Transform) -> SurfaceKind {
    match c {
        Curve2::Line { a, b } => {
            let d = [b[0] - a[0], b[1] - a[1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
            // Outward for a counter-clockwise outer loop.
            plane_hint([d[1] / l, -d[0] / l, 0.0], to_world)
        }
        Curve2::Arc { c, r, .. } | Curve2::Circle { c, r } => SurfaceKind::Cylinder {
            axis_point: to_world.apply([c[0], c[1], 0.0]),
            axis_direction: to_world.apply_vector([0.0, 0.0, 1.0]),
            radius: *r,
        },
        Curve2::Poly { .. } => SurfaceKind::Other,
    }
}

/// A prism of any profile on any frame: `to_world` places the profile plane (local z = 0);
/// the prism spans local `z0..z1` with `labels[0]` at `z0`.
pub(super) fn prism(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    z0: f64,
    z1: f64,
    cap_labels: [HistorySource; 2],
) -> Sampled {
    let mut surfaces = Vec::new();
    let mut l2 = Vec::new();
    for l in loops {
        let mut v = Vec::new();
        for (c, lab) in l {
            v.push((c.clone(), surfaces.len()));
            surfaces.push(Surf {
                label: lab.clone(),
                hint: side_hint(c, &to_world),
                plane: None,
            });
        }
        l2.push(v);
    }
    let (z0, z1, caps_l) = if z0 <= z1 {
        (z0, z1, cap_labels)
    } else {
        let [a, b] = cap_labels;
        (z1, z0, [b, a])
    };
    let [c0, c1] = caps_l;
    let caps = [surfaces.len(), surfaces.len() + 1];
    surfaces.push(Surf {
        label: c0,
        hint: plane_hint([0.0, 0.0, -1.0], &to_world),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: plane_hint([0.0, 0.0, 1.0], &to_world),
        plane: None,
    });
    let counts: Vec<Vec<usize>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, _)| c.samples()).collect())
        .collect();
    let polys = loop_polys(&l2, &counts);
    let pts: Vec<[f64; 3]> = polys
        .iter()
        .flatten()
        .flat_map(|q| [[q[0], q[1], z0], [q[0], q[1], z1]])
        .map(|p| to_world.apply(p))
        .collect();
    let (lo, hi) = bbox(pts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Prism {
            loops: l2,
            polys,
            z0,
            z1,
            caps,
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    s
}

/// A revolve of profile loops (local z = 0 plane of `to_world`) about the local in-plane axis
/// through `ap` along `ad`, by `angle` radians (negative: the other way).
pub(super) fn revolve(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    ap: [f64; 2],
    ad: [f64; 2],
    angle: f64,
    cap_labels: Option<[HistorySource; 2]>,
) -> Result<Sampled, KernelError> {
    let l = (ad[0] * ad[0] + ad[1] * ad[1]).sqrt();
    if l < 1e-12 {
        return Err(unsupported("revolve axis out of the profile plane"));
    }
    let ad = [ad[0] / l, ad[1] / l];
    let mut surfaces = Vec::new();
    let mut l2 = Vec::new();
    let mut centroid = [0.0, 0.0];
    let mut count = 0.0;
    for lp in loops {
        let mut v = Vec::new();
        for (c, lab) in lp {
            for q in c.polyline(c.samples()) {
                centroid[0] += q[0];
                centroid[1] += q[1];
                count += 1.0;
            }
            v.push((c.clone(), surfaces.len()));
            let hint = match c.line() {
                Some((a, b)) => {
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let along = (d[0] * ad[0] + d[1] * ad[1]).abs() / (d2(a, b).max(1e-300));
                    let r = {
                        let rel = [a[0] - ap[0], a[1] - ap[1]];
                        let h = rel[0] * ad[0] + rel[1] * ad[1];
                        d2(rel, [h * ad[0], h * ad[1]])
                    };
                    if (along - 1.0).abs() < 1e-9 {
                        SurfaceKind::Cylinder {
                            axis_point: to_world.apply([ap[0], ap[1], 0.0]),
                            axis_direction: to_world.apply_vector([ad[0], ad[1], 0.0]),
                            radius: r,
                        }
                    } else if along < 1e-9 {
                        plane_hint([ad[0], ad[1], 0.0], &to_world)
                    } else {
                        SurfaceKind::Other
                    }
                }
                None => SurfaceKind::Other,
            };
            surfaces.push(Surf {
                label: lab.clone(),
                hint,
                plane: None,
            });
        }
        l2.push(v);
    }
    if count == 0.0 {
        return Err(unsupported("revolve of an empty profile"));
    }
    centroid = [centroid[0] / count, centroid[1] / count];
    let rel = [centroid[0] - ap[0], centroid[1] - ap[1]];
    let h = rel[0] * ad[0] + rel[1] * ad[1];
    let perp = [rel[0] - h * ad[0], rel[1] - h * ad[1]];
    let pl = (perp[0] * perp[0] + perp[1] * perp[1]).sqrt();
    if pl < 1e-12 {
        return Err(unsupported("revolve profile centred on its axis"));
    }
    let e1 = [perp[0] / pl, perp[1] / pl];
    let sense = if angle < 0.0 { -1.0 } else { 1.0 };
    let angle = angle.abs().min(TAU);
    let full = angle >= TAU - 1e-9;
    let caps = match (full, cap_labels) {
        (false, Some([s, e])) => {
            let i = surfaces.len();
            // The start cap lies in the profile plane, facing against the sweep.
            surfaces.push(Surf {
                label: s,
                hint: plane_hint([0.0, 0.0, -sense], &to_world),
                plane: None,
            });
            // The end cap is the profile plane turned through the angle.
            let end_normal = Transform::rotation_about(
                to_world.apply([ap[0], ap[1], 0.0]),
                to_world.apply_vector([ad[0], ad[1], 0.0]),
                sense * angle,
            )
            .apply_vector(to_world.apply_vector([0.0, 0.0, sense]));
            surfaces.push(Surf {
                label: e,
                hint: SurfaceKind::Plane { normal: end_normal },
                plane: None,
            });
            Some([i, i + 1])
        }
        _ => None,
    };
    let counts: Vec<Vec<usize>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, _)| c.samples()).collect())
        .collect();
    let polys = loop_polys(&l2, &counts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Revolve {
            loops: l2,
            polys: polys.clone(),
            ap,
            ad,
            e1,
            sense,
            angle,
            caps,
        },
        surfaces,
        lo: [0.0; 3],
        hi: [0.0; 3],
        exact: Vec::new(),
    };
    // Bounding box: profile samples turned through the sweep.
    let steps = 48;
    let mut pts = Vec::new();
    let ad3 = [ad[0], ad[1], 0.0];
    let e13 = [e1[0], e1[1], 0.0];
    let e23 = [
        sense * (ad3[1] * e13[2] - ad3[2] * e13[1]),
        sense * (ad3[2] * e13[0] - ad3[0] * e13[2]),
        sense * (ad3[0] * e13[1] - ad3[1] * e13[0]),
    ];
    for q in polys.iter().flatten() {
        let rel = [q[0] - ap[0], q[1] - ap[1]];
        let h = rel[0] * ad[0] + rel[1] * ad[1];
        let r = d2(rel, [h * ad[0], h * ad[1]]);
        for k in 0..=steps {
            let a = angle * k as f64 / steps as f64;
            let (sn, cs) = a.sin_cos();
            let p = [
                ap[0] + h * ad3[0] + r * (cs * e13[0] + sn * e23[0]),
                ap[1] + h * ad3[1] + r * (cs * e13[1] + sn * e23[1]),
                r * (cs * e13[2] + sn * e23[2]),
            ];
            pts.push(to_world.apply(p));
        }
    }
    let (mut lo, mut hi) = bbox(pts);
    let pad = 1e-6 * (0..3).map(|a| hi[a] - lo[a]).fold(1.0, f64::max);
    for a in 0..3 {
        lo[a] -= pad;
        hi[a] += pad;
    }
    s.lo = lo;
    s.hi = hi;
    s.recompute_exact();
    Ok(s)
}

/// A translational sweep of profile loops (local z = 0) along a polyline path given in local
/// coordinates. Side faces are numbered by path segment when the path has corners.
pub(super) fn sweep(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    path: &[[f64; 3]],
    path_curves: &[u32],
    cap_labels: [HistorySource; 2],
) -> Result<Sampled, KernelError> {
    if path.len() < 2 {
        return Err(KernelError::Failed {
            message: "a sweep path needs at least two points".into(),
        });
    }
    let offsets: Vec<[f64; 3]> = path.iter().map(|p| sub3(*p, path[0])).collect();
    let n = offsets.len() - 1;
    for j in 0..n {
        if (offsets[j + 1][2] - offsets[j][2]).abs() < 1e-12 {
            return Err(unsupported("sweep path segments in the profile plane"));
        }
    }
    // Path curve per segment: given, or one per segment.
    let curve_of: Vec<u32> = if path_curves.len() == n {
        path_curves.to_vec()
    } else {
        (0..n as u32).collect()
    };
    let distinct = {
        let mut v = curve_of.clone();
        v.sort();
        v.dedup();
        v.len()
    };
    let mut surfaces: Vec<Surf> = Vec::new();
    let mut curves = Vec::new();
    let mut sides = Vec::new();
    for l in loops {
        let mut cl = Vec::new();
        let mut sl = Vec::new();
        for (c, lab) in l {
            cl.push(c.clone());
            let mut per = Vec::new();
            let mut by_curve: std::collections::BTreeMap<u32, usize> =
                std::collections::BTreeMap::new();
            for j in 0..n {
                if let Some(si) = by_curve.get(&curve_of[j]) {
                    // A further sample of the same path curve: the same face.
                    per.push(*si);
                    if let SurfaceKind::Plane { .. } = surfaces[*si].hint {
                        let seg = sub3(offsets[j + 1], offsets[j]);
                        let prev = sub3(offsets[j], offsets[j.saturating_sub(1)]);
                        if norm3(seg) > 0.0 && j > 0 && {
                            let c = dot3(seg, prev) / (norm3(seg) * norm3(prev).max(1e-300));
                            (c - 1.0).abs() > 1e-9
                        } {
                            surfaces[*si].hint = SurfaceKind::Other;
                        }
                    }
                    continue;
                }
                let mut label = lab.clone();
                if distinct > 1 {
                    if let HistorySource::Generated { ordinal, .. } = &mut label {
                        *ordinal = Some(curve_of[j]);
                    }
                }
                by_curve.insert(curve_of[j], surfaces.len());
                per.push(surfaces.len());
                // A straight curve swept along a straight segment is planar.
                let hint = match c.line() {
                    Some((a, b)) => {
                        let l = [b[0] - a[0], b[1] - a[1], 0.0];
                        let d = sub3(offsets[j + 1], offsets[j]);
                        let nrm = [
                            l[1] * d[2] - l[2] * d[1],
                            l[2] * d[0] - l[0] * d[2],
                            l[0] * d[1] - l[1] * d[0],
                        ];
                        let m = norm3(nrm).max(1e-300);
                        plane_hint([nrm[0] / m, nrm[1] / m, nrm[2] / m], &to_world)
                    }
                    None => SurfaceKind::Other,
                };
                surfaces.push(Surf {
                    label,
                    hint,
                    plane: None,
                });
            }
            sl.push(per);
        }
        curves.push(cl);
        sides.push(sl);
    }
    let [c0, c1] = cap_labels;
    let caps = [surfaces.len(), surfaces.len() + 1];
    surfaces.push(Surf {
        label: c0,
        hint: plane_hint([0.0, 0.0, -1.0], &to_world),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: plane_hint([0.0, 0.0, 1.0], &to_world),
        plane: None,
    });
    let indexed: Vec<Vec<(Curve2, usize)>> = curves
        .iter()
        .map(|l| l.iter().map(|c| (c.clone(), 0)).collect())
        .collect();
    let counts: Vec<Vec<usize>> = curves
        .iter()
        .map(|l| l.iter().map(|c| c.samples()).collect())
        .collect();
    let polys = loop_polys(&indexed, &counts);
    let pts: Vec<[f64; 3]> = offsets
        .iter()
        .flat_map(|o| {
            polys
                .iter()
                .flatten()
                .map(move |q| [q[0] + o[0], q[1] + o[1], o[2]])
                .collect::<Vec<_>>()
        })
        .map(|p| to_world.apply(p))
        .collect();
    let (lo, hi) = bbox(pts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Sweep {
            loops: curves,
            polys,
            offsets,
            sides,
            caps,
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    Ok(s)
}

/// A ruled loft through sections at local heights `z` (strictly increasing after sorting),
/// each with the same loops and curve counts. Side faces are tagged by the first section's
/// curves and numbered by span when there are more than two sections.
pub(super) fn loft(
    to_world: Transform,
    sections: Vec<(f64, Vec<Vec<Curve2>>)>,
    labels: &[Vec<HistorySource>],
    cap_labels: [HistorySource; 2],
) -> Result<Sampled, KernelError> {
    if sections.len() < 2 {
        return Err(KernelError::Failed {
            message: "a loft needs at least two sections".into(),
        });
    }
    let shape0: Vec<usize> = sections[0].1.iter().map(|l| l.len()).collect();
    for (_, s) in &sections {
        let sh: Vec<usize> = s.iter().map(|l| l.len()).collect();
        if sh != shape0 {
            return Err(unsupported(
                "loft sections with different loop or curve counts",
            ));
        }
    }
    let mut order: Vec<usize> = (0..sections.len()).collect();
    let increasing = sections[sections.len() - 1].0 > sections[0].0;
    if !increasing {
        order.reverse();
    }
    let z: Vec<f64> = order.iter().map(|i| sections[*i].0).collect();
    if z.windows(2).any(|w| w[1] - w[0] < 1e-9) {
        return Err(unsupported(
            "loft sections that are not stacked along the normal",
        ));
    }
    let spans = z.len() - 1;
    // Equal sample counts per curve across sections.
    let mut counts: Vec<Vec<usize>> = shape0.iter().map(|n| vec![1; *n]).collect();
    for (_, s) in &sections {
        for (li, l) in s.iter().enumerate() {
            for (ci, c) in l.iter().enumerate() {
                counts[li][ci] = counts[li][ci].max(c.samples());
            }
        }
    }
    let curves: Vec<Vec<Vec<Vec<[f64; 2]>>>> = order
        .iter()
        .map(|i| {
            sections[*i]
                .1
                .iter()
                .enumerate()
                .map(|(li, l)| {
                    l.iter()
                        .enumerate()
                        .map(|(ci, c)| c.polyline(counts[li][ci]))
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut surfaces = Vec::new();
    let mut sides = Vec::new();
    for (li, l) in labels.iter().enumerate() {
        let mut sl = Vec::new();
        for lab in l.iter().take(shape0[li]) {
            let mut per = Vec::new();
            for k in 0..spans {
                let span = if increasing { k } else { spans - 1 - k };
                let mut label = lab.clone();
                if spans > 1 {
                    if let HistorySource::Generated { ordinal, .. } = &mut label {
                        *ordinal = Some(span as u32);
                    }
                }
                per.push(surfaces.len());
                surfaces.push(Surf {
                    label,
                    hint: SurfaceKind::Other,
                    plane: None,
                });
            }
            sl.push(per);
        }
        sides.push(sl);
    }
    let [mut c0, mut c1] = cap_labels;
    if !increasing {
        std::mem::swap(&mut c0, &mut c1);
    }
    let caps = [surfaces.len(), surfaces.len() + 1];
    surfaces.push(Surf {
        label: c0,
        hint: plane_hint([0.0, 0.0, -1.0], &to_world),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: plane_hint([0.0, 0.0, 1.0], &to_world),
        plane: None,
    });
    let pts: Vec<[f64; 3]> = curves
        .iter()
        .zip(&z)
        .flat_map(|(sec, zz)| {
            sec.iter()
                .flatten()
                .flatten()
                .map(|q| [q[0], q[1], *zz])
                .collect::<Vec<_>>()
        })
        .map(|p| to_world.apply(p))
        .collect();
    let (lo, hi) = bbox(pts);
    let mut s = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Loft {
            z,
            curves,
            sides,
            caps,
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    Ok(s)
}

/// The image of a whole solid under `t` (any rigid motion or reflection). Faces become
/// [`HistorySource::Copy`] of the operand's faces at `instance`, or continue them
/// ([`HistorySource::Input`]) when the body itself moves.
pub(super) fn image(
    solid: &FakeSolid,
    operand: u32,
    t: &Transform,
    instance: Option<&InstanceKey>,
) -> Result<Sampled, KernelError> {
    if !solid.decorations.is_empty() {
        return Err(unsupported(
            "moving blended bodies by anything but right-angle motions",
        ));
    }
    let surfaces = solid
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| Surf {
            label: {
                let from = OperandEntity {
                    operand,
                    entity: i as u32,
                };
                match instance {
                    Some(k) => HistorySource::Copy {
                        from,
                        instance: k.clone(),
                    },
                    None => HistorySource::Input { from },
                }
            },
            hint: move_hint(&f.hint(), t),
            plane: None,
        })
        .collect();
    let (lo, hi) = solid
        .bounds()
        .map(|(l, h)| moved_box(l, h, t))
        .ok_or_else(|| unsupported("moving an empty body"))?;
    let mut s = Sampled {
        to_local: t.inverse(),
        shape: Shape::Image {
            solid: Arc::new(solid.clone()),
        },
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    s.recompute_exact();
    Ok(s)
}

/// The thin wall's inner side of a straight curve: the line moved `inside` into the profile.
fn inner_line(c: &Curve2, inside: f64) -> Option<[f64; 2]> {
    let (a, b) = c.line()?;
    let d = [b[0] - a[0], b[1] - a[1]];
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
    // Inward is left of a counter-clockwise outer loop.
    Some([a[0] - d[1] / l * inside, a[1] + d[0] / l * inside])
}

/// An extrusion: a prism whose caps follow end conditions, with draft and thin options.
/// `labels` are the side labels per loop and curve; `inner` the thin inner-wall labels.
#[allow(clippy::too_many_arguments)]
pub(super) fn extrusion(
    to_world: Transform,
    loops: &[Vec<(Curve2, HistorySource)>],
    inner: Option<&[Vec<HistorySource>]>,
    start: ZBound,
    end: ZBound,
    cap_labels: [HistorySource; 2],
    draft: Option<(f64, f64, f64)>,
    thin: Option<ThinBand<HistorySource>>,
) -> Result<Sampled, KernelError> {
    let mut surfaces = Vec::new();
    let mut l2 = Vec::new();
    let end_sign = match (start.fixed(), end.fixed()) {
        (Some(a), Some(b)) => (b - a).signum(),
        _ => 1.0,
    };
    for (li, l) in loops.iter().enumerate() {
        let mut v = Vec::new();
        for (ci, (c, lab)) in l.iter().enumerate() {
            let o = surfaces.len();
            let hint = match (draft, c) {
                (None, _) => side_hint(c, &to_world),
                (Some((h, te, ts)), Curve2::Line { a, b }) => {
                    let one_side = start
                        .fixed()
                        .zip(end.fixed())
                        .map(|(zs, ze)| (zs - h) * (ze - h) >= -1e-15);
                    match one_side {
                        Some(true) => {
                            let d = [b[0] - a[0], b[1] - a[1]];
                            let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
                            let t = if end
                                .fixed()
                                .map(|z| (z - h) * end_sign >= 0.0)
                                .unwrap_or(true)
                            {
                                te
                            } else {
                                -ts
                            };
                            plane_hint([d[1] / l, -d[0] / l, t * end_sign], &to_world)
                        }
                        _ => SurfaceKind::Other,
                    }
                }
                (Some(_), _) => SurfaceKind::Other,
            };
            surfaces.push(Surf {
                label: lab.clone(),
                hint,
                plane: None,
            });
            let inn = match (inner, thin.as_ref()) {
                (Some(labels), Some(_)) => {
                    let k = surfaces.len();
                    let mut hint = side_hint(c, &to_world);
                    if let SurfaceKind::Plane { normal } = hint {
                        hint = SurfaceKind::Plane {
                            normal: [-normal[0], -normal[1], -normal[2]],
                        };
                    }
                    if draft.is_some() {
                        hint = SurfaceKind::Other;
                    }
                    surfaces.push(Surf {
                        label: labels[li][ci].clone(),
                        hint,
                        plane: None,
                    });
                    Some(k)
                }
                _ => None,
            };
            v.push((c.clone(), o, inn));
        }
        l2.push(v);
    }
    let caps = [surfaces.len(), surfaces.len() + 1];
    let [c0, c1] = cap_labels;
    let cap_hint = |b: &ZBound, n: f64| match b {
        ZBound::At(_) => plane_hint([0.0, 0.0, n], &to_world),
        _ => SurfaceKind::Other,
    };
    surfaces.push(Surf {
        label: c0,
        hint: cap_hint(&start, -end_sign),
        plane: None,
    });
    surfaces.push(Surf {
        label: c1,
        hint: cap_hint(&end, end_sign),
        plane: None,
    });
    let thin_spec = match thin {
        None => None,
        Some((i, o, None)) => Some((i, o, None)),
        Some((i, o, Some((cap, [a, b])))) => {
            let k = surfaces.len();
            surfaces.push(Surf {
                label: a,
                hint: plane_hint([0.0, 0.0, end_sign], &to_world),
                plane: None,
            });
            surfaces.push(Surf {
                label: b,
                hint: plane_hint([0.0, 0.0, -end_sign], &to_world),
                plane: None,
            });
            Some((i, o, Some((cap, [k, k + 1]))))
        }
    };
    let counts: Vec<Vec<usize>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, _, _)| c.samples()).collect())
        .collect();
    let curve_loops: Vec<Vec<(Curve2, usize)>> = l2
        .iter()
        .map(|l| l.iter().map(|(c, o, _)| (c.clone(), *o)).collect())
        .collect();
    let polys = loop_polys(&curve_loops, &counts);
    let x = Extrusion {
        loops: l2,
        polys,
        start,
        end,
        caps,
        end_sign,
        draft,
        thin: thin_spec,
    };
    // The extent: caps over sample points of the profile, sides grown by any outward wall.
    let mut plo = [f64::INFINITY; 2];
    let mut phi = [f64::NEG_INFINITY; 2];
    for q in x.polys.iter().flatten() {
        for a in 0..2 {
            plo[a] = plo[a].min(q[a]);
            phi[a] = phi[a].max(q[a]);
        }
    }
    let mut samples: Vec<[f64; 2]> = x.polys.iter().flatten().copied().collect();
    for i in 0..=8 {
        for j in 0..=8 {
            let q = [
                plo[0] + (phi[0] - plo[0]) * i as f64 / 8.0,
                plo[1] + (phi[1] - plo[1]) * j as f64 / 8.0,
            ];
            if inside_polys(&x.polys, q) {
                samples.push(q);
            }
        }
    }
    let (mut zlo, mut zhi) = (f64::INFINITY, f64::NEG_INFINITY);
    for q in &samples {
        // Boundary points sit exactly on the profile; nudge them inside for the ray.
        let Some((a, b)) = x.caps_at(*q, &to_world) else {
            if inside_polys(&x.polys, *q) {
                return Err(KernelError::Failed {
                    message: "no face the extrude can end at intercepts the whole profile".into(),
                });
            }
            continue;
        };
        zlo = zlo.min(a.min(b));
        zhi = zhi.max(a.max(b));
    }
    if !zlo.is_finite() || zhi - zlo <= 1e-12 {
        return Err(KernelError::Failed {
            message: "the extrude has no length".into(),
        });
    }
    let mut grow = x.thin.map(|t| t.1).unwrap_or(0.0);
    if let Some((h, te, ts)) = x.draft {
        let reach = (zhi - h).abs().max((zlo - h).abs());
        grow += reach * (-te).max(-ts).max(0.0);
    }
    let pts: Vec<[f64; 3]> = [
        [plo[0] - grow, plo[1] - grow],
        [phi[0] + grow, plo[1] - grow],
        [plo[0] - grow, phi[1] + grow],
        [phi[0] + grow, phi[1] + grow],
    ]
    .iter()
    .flat_map(|q| [[q[0], q[1], zlo], [q[0], q[1], zhi]])
    .map(|p| to_world.apply(p))
    .collect();
    let (lo, hi) = bbox(pts);
    let mut out = Sampled {
        to_local: to_world.inverse(),
        shape: Shape::Extrusion(Box::new(x)),
        surfaces,
        lo,
        hi,
        exact: Vec::new(),
    };
    out.recompute_exact();
    Ok(out)
}
