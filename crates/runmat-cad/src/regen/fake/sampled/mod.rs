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
//! segment, ordinals from the path); lofts through parallel sections (one side face per curve
//! through every section, sampled as ruled strips); holes (simple, counterbore, countersink) as revolves of their section; and
//! images of whole bodies under any rigid motion or reflection.

mod occupancy;
mod breaks;
mod surfaces;
mod shapes;
mod builders;
mod sweep_loft;
mod images;
pub(super) use builders::*;
pub(super) use images::*;
pub(super) use shapes::*;
pub(super) use sweep_loft::*;

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

impl Sampled {}
