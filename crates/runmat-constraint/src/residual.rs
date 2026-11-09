//! Residual functions.
//!
//! Each constraint is lowered once per solve to a [`Compiled`] record: the ordered list of
//! variables it reads (its local layout) and an [`Op`] that evaluates its residual rows from
//! those variables. Ops are generic over [`Real`], so the same code yields values (`f64`) and
//! exact Jacobian blocks ([`Dual`]).
//!
//! Orientation-dependent constraints (tangent side, internal versus external contact, the side
//! of a point–line distance, the curvature sign of a G2 joint) take their branch from the
//! starting geometry, so a solve never flips a sketch to the mirror configuration.
//!
//! # Wide constraints
//!
//! A residual's derivatives are taken by forward-mode AD seeded only on the layout entries it
//! really reads at the current point ([`Op::support`]), in chunks of at most 16 seeds. A point
//! on a spline with hundreds of control points reads the `degree + 1` control points of its
//! active knot span, so its Jacobian block costs the same as for a 4-point Bézier.

use crate::ad::{Dual, Real};
use crate::ids::{ConstraintId, EntityId, VarId};
use crate::system::{
    AngleMode, Constraint, ConstraintKindRecord, CurveEnd, DistanceMode, EntityKind, Frame,
    InternalKind, Mate, System, TangentKind, Target,
};
use std::f64::consts::PI;

/// How a round curve's centre and radius are read from its layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Round {
    /// `[cx, cy, r]`
    Circle,
    /// `[cx, cy, sx, sy, ex, ey]`; the radius is `|s - c|`.
    Arc,
}

impl Round {
    fn len(self) -> usize {
        match self {
            Round::Circle => 3,
            Round::Arc => 6,
        }
    }
    fn radius<S: Real>(self, x: &[S]) -> S {
        match self {
            Round::Circle => x[2],
            Round::Arc => (x[2] - x[0]).hypot(x[3] - x[1]),
        }
    }
}

/// How the tangent direction at an open-curve end is read from its end layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EndDir {
    /// `[p, q]`: direction `q - p` (a line, or a spline's end control polygon leg).
    Segment,
    /// `[c, p]`: perpendicular to the radius `p - c` (an arc end).
    ArcRadius,
    /// `[ellipse(5), p]`: the counter-clockwise ellipse tangent at `p` (an elliptical arc end).
    Ellipse,
}

impl EndDir {
    fn len(self) -> usize {
        match self {
            EndDir::Segment | EndDir::ArcRadius => 4,
            EndDir::Ellipse => 7,
        }
    }
    fn dir<S: Real>(self, x: &[S]) -> (S, S) {
        match self {
            EndDir::Segment => (x[2] - x[0], x[3] - x[1]),
            EndDir::ArcRadius => (-(x[3] - x[1]), x[2] - x[0]),
            EndDir::Ellipse => ell_tangent_at(&x[0..5], (x[5], x[6])),
        }
    }
}

/// The unit of a residual row, for unit-aware scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unit {
    /// A length (scaled by the sub-problem's characteristic length).
    Length,
    /// An angle, a ratio or another dimensionless quantity (not scaled).
    Ratio,
}

/// A lowered residual.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Op {
    /// `[p, q]` → `p - q` (2 rows).
    Coincident,
    /// `[p, q]` → `py - qy`.
    SameY,
    /// `[p, q]` → `px - qx`.
    SameX,
    /// `[l1, l2]` → `sin` of the angle between them.
    Parallel,
    /// `[l1, l2]` → `cos` of the angle between them.
    Perpendicular,
    /// `[line, round]` → `side · dist(c, line) - r`.
    TangentLineRound { round: Round, side: f64 },
    /// `[roundA, roundB]` → `|ca - cb| - (ra + rb)` or `|ca - cb| - σ(ra - rb)`.
    TangentRounds {
        a: Round,
        b: Round,
        internal: bool,
        sigma: f64,
    },
    /// `[line, ellipse(5)]` → `side · dist(c, line) - h(n)`, `h` the ellipse's support
    /// function along the line normal.
    TangentLineEllipse { side: f64 },
    /// `[round, ellipse(5), θ]` → the ellipse point at eccentric angle `θ` is on the circle
    /// and the circle's radius there is normal to the ellipse (2 rows).
    TangentRoundEllipse { round: Round },
    /// `[ellipseA(5), ellipseB(5), θa, θb]` → the two contact points coincide and their
    /// tangents are parallel (3 rows).
    TangentEllipses,
    /// `[endA, endB]` → `sin` of the angle between the end tangents.
    TangentEnds { a: EndDir, b: EndDir },
    /// `[Q0, Q1, Q2, arc(6)]` (then `[w0, w1, w2]` when rational) → G1 (`cos` between
    /// tangent and radius) and `κ r - σ`.
    CurvatureG2 {
        e1: f64,
        e2: f64,
        f: f64,
        degree: f64,
        sigma: f64,
        rational: bool,
    },
    /// `[l1, l2]` → `|l1| - |l2|`.
    EqualLength,
    /// `[roundA, roundB]` → `ra - rb`.
    EqualRadius { a: Round, b: Round },
    /// `[aA, bA, aB, bB]` → equal semi-axes (2 rows).
    EqualEllipse,
    /// `[a, b]` → an ellipse dimension minus its value.
    EllipseValue {
        measure: crate::system::EllipseMeasure,
        v: f64,
    },
    /// `[roundA, roundB]` → centres coincide and radii are equal (3 rows).
    Coradial { a: Round, b: Round },
    /// `[p, line]` → `p - mid(line)` (2 rows).
    Midpoint,
    /// `[a, b, line]` → mid(a, b) on the line and `(b - a) ⟂ line` (2 rows).
    Symmetric,
    /// `[l1, l2]` → both ends of `l2` on `l1` (2 rows).
    Collinear,
    /// `[vars…]` → `x_i - target_i`.
    FixValues { targets: Vec<f64> },
    /// `[c, s, e]` → centre and start fixed, end held along its circle's tangent (5 rows).
    FixArc {
        targets: [f64; 6],
        tangent: [f64; 2],
    },
    /// `[body(7)]` → position and orientation held (6 rows).
    FixBody { targets: [f64; 7] },
    /// `[ellipse(5), s, e]` → ellipse held, each end held along its tangent (7 rows).
    FixEllipticalArc {
        targets: [f64; 9],
        tangents: [f64; 4],
    },
    /// `[p, line]` → signed distance from the line.
    PointOnLine,
    /// `[p, round]` → `|p - c| - r`.
    PointOnRound { round: Round },
    /// `[p, ellipse(5)]` → scaled implicit ellipse equation.
    PointOnEllipse,
    /// `[p, t, control points…, weights…]` → `p - C(t)` (2 rows). Control points (and
    /// weights, when rational) are the wrapped sequence for a periodic spline.
    PointOnSpline {
        degree: usize,
        knots: Vec<f64>,
        rational: bool,
        periodic: bool,
    },
    /// `[p, q]` → `|p - q| - d`.
    DistancePP { d: f64 },
    /// `[p, line]` → `side · dist - d`.
    DistancePL { d: f64, side: f64 },
    /// `[l1, l2]` → wrapped `angle(l1, l2) - θ`.
    Angle { theta: f64 },
    /// `[round]` → `scale · r - v`.
    RadiusValue { round: Round, scale: f64, v: f64 },
    /// `[line]` → `|line| - v`.
    Length { v: f64 },
    /// `[var]` → `x - v`.
    VarValue { v: f64 },
    /// Internal: `[arc(6)]` → `|s - c| - |e - c|`.
    ArcRadius,
    /// Internal: `[q(4)]` → `(|q|² - 1) / 2`.
    QuatNorm,
    /// Internal: `[ellipse(5), s, e]` → both ends on the ellipse (2 rows).
    EllipticalArcEnds,
    /// `[bodyA(7), bodyB(7)]` → mate rows.
    Mate { kind: MateOp, la: Frame, lb: Frame },
}

/// Mate residual kinds (connector frames are applied in [`Op::Mate`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MateOp {
    Fasten,
    Revolute,
    Slider,
    Cylindrical,
    Planar,
    Ball,
    Parallel,
    TangentSphere {
        r: f64,
    },
    TangentCylinder {
        r: f64,
    },
    DistancePoint {
        d: f64,
    },
    DistancePlane {
        d: f64,
    },
    AngleAboutZ {
        theta: f64,
    },
    AngleBetweenZ {
        theta: f64,
    },
    PinSlot,
    Screw {
        lead: f64,
    },
    /// `x0`/`z0`: each connector's world X and Z where the solve starts.
    Gear {
        ratio: f64,
        xa0: [f64; 3],
        za0: [f64; 3],
        xb0: [f64; 3],
        zb0: [f64; 3],
    },
    /// `ob0`/`xb0`: the rack connector's world origin and X where the solve starts.
    RackPinion {
        travel: f64,
        xa0: [f64; 3],
        za0: [f64; 3],
        ob0: [f64; 3],
        xb0: [f64; 3],
    },
    /// The cam's station radii, evenly spaced from its X.
    Cam {
        radii: std::sync::Arc<[f64]>,
    },
    /// Each connector's world origin and Z where the solve starts.
    LinearCoupler {
        ratio: f64,
        oa0: [f64; 3],
        za0: [f64; 3],
        ob0: [f64; 3],
        zb0: [f64; 3],
    },
    UniversalJoint {
        joint_point: bool,
    },
    /// The plane's connector frame; its body's 7 variables follow A's and B's.
    Symmetric {
        lc: Frame,
    },
    /// The path's points in A's connector frame.
    Path {
        points: std::sync::Arc<[[f64; 3]]>,
    },
}

impl Op {
    /// The number of residual rows.
    pub(crate) fn rows(&self) -> usize {
        match self {
            Op::Coincident
            | Op::Midpoint
            | Op::Symmetric
            | Op::Collinear
            | Op::PointOnSpline { .. } => 2,
            Op::CurvatureG2 { .. }
            | Op::TangentRoundEllipse { .. }
            | Op::EqualEllipse
            | Op::EllipticalArcEnds => 2,
            Op::Coradial { .. } | Op::TangentEllipses => 3,
            Op::FixEllipticalArc { .. } => 7,
            Op::FixValues { targets } => targets.len(),
            Op::FixArc { .. } => 5,
            Op::FixBody { .. } => 6,
            Op::Mate { kind, .. } => match kind {
                MateOp::Fasten => 6,
                MateOp::Revolute | MateOp::Slider | MateOp::Screw { .. } => 5,
                MateOp::Cylindrical | MateOp::PinSlot => 4,
                MateOp::UniversalJoint { joint_point: true } => 4,
                MateOp::Planar | MateOp::Ball | MateOp::Symmetric { .. } => 3,
                MateOp::Parallel | MateOp::TangentCylinder { .. } | MateOp::Path { .. } => 2,
                _ => 1,
            },
            _ => 1,
        }
    }

    /// The same op with its dimension target replaced (used to measure with a zero target and
    /// to activate a range bound).
    pub(crate) fn with_target(&self, t: f64) -> Op {
        let mut op = self.clone();
        match &mut op {
            Op::DistancePP { d } | Op::DistancePL { d, .. } => *d = t,
            Op::Angle { theta } => *theta = t,
            Op::RadiusValue { v, .. }
            | Op::Length { v }
            | Op::VarValue { v }
            | Op::EllipseValue { v, .. } => *v = t,
            Op::Mate { kind, .. } => match kind {
                MateOp::DistancePoint { d } | MateOp::DistancePlane { d } => *d = t,
                MateOp::AngleAboutZ { theta } | MateOp::AngleBetweenZ { theta } => *theta = t,
                _ => {}
            },
            _ => {}
        }
        op
    }

    /// Evaluates the residual rows into `out` (length [`Op::rows`]).
    pub(crate) fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
        match self {
            Op::Coincident => {
                out[0] = x[0] - x[2];
                out[1] = x[1] - x[3];
            }
            Op::SameY => out[0] = x[1] - x[3],
            Op::SameX => out[0] = x[0] - x[2],
            Op::Parallel => {
                let (a, b) = (seg(&x[0..4]), seg(&x[4..8]));
                out[0] = cross(a, b) / (norm(a) * norm(b));
            }
            Op::Perpendicular => {
                let (a, b) = (seg(&x[0..4]), seg(&x[4..8]));
                out[0] = dot(a, b) / (norm(a) * norm(b));
            }
            Op::TangentLineRound { round, side } => {
                let r = round.radius(&x[4..]);
                out[0] = signed_dist(&x[0..4], (x[4], x[5])) * *side - r;
            }
            Op::TangentRounds {
                a,
                b,
                internal,
                sigma,
            } => {
                let ra = a.radius(&x[..a.len()]);
                let xb = &x[a.len()..];
                let rb = b.radius(xb);
                let d = (x[0] - xb[0]).hypot(x[1] - xb[1]);
                out[0] = if *internal {
                    d - (ra - rb) * *sigma
                } else {
                    d - (ra + rb)
                };
            }
            Op::TangentLineEllipse { side } => {
                let l = &x[0..4];
                let e = &x[4..9];
                let d = seg(l);
                let nd = norm(d);
                let n = (-d.1 / nd, d.0 / nd);
                let (u, v) = ell_axes(e);
                let h = ((e[2] * dot(n, u)).sq() + (e[3] * dot(n, v)).sq()).sqrt();
                out[0] = signed_dist(l, (e[0], e[1])) * *side - h;
            }
            Op::TangentRoundEllipse { round } => {
                let k = round.len();
                let r = round.radius(&x[..k]);
                let e = &x[k..k + 5];
                let t = x[k + 5];
                let p = ell_point(e, t);
                let tan = ell_tangent(e, t);
                let q = (p.0 - x[0], p.1 - x[1]);
                let nq = norm(q);
                out[0] = nq - r;
                out[1] = dot(q, tan) / (nq * norm(tan));
            }
            Op::TangentEllipses => {
                let (ea, eb) = (&x[0..5], &x[5..10]);
                let (ta, tb) = (x[10], x[11]);
                let (pa, pb) = (ell_point(ea, ta), ell_point(eb, tb));
                out[0] = pa.0 - pb.0;
                out[1] = pa.1 - pb.1;
                let (da, db) = (ell_tangent(ea, ta), ell_tangent(eb, tb));
                out[2] = cross(da, db) / (norm(da) * norm(db));
            }
            Op::TangentEnds { a, b } => {
                let k = a.len();
                let (ta, tb) = (a.dir(&x[0..k]), b.dir(&x[k..k + b.len()]));
                out[0] = cross(ta, tb) / (norm(ta) * norm(tb));
            }
            Op::CurvatureG2 {
                e1,
                e2,
                f,
                degree,
                sigma,
                rational,
            } => {
                let p = *degree;
                let q0 = (x[0], x[1]);
                let d01 = (x[2] - x[0], x[3] - x[1]);
                let d12 = (x[4] - x[2], x[5] - x[3]);
                let c1 = (d01.0 * (p / e1), d01.1 * (p / e1));
                let k = p * (p - 1.0) / f;
                let c2 = (
                    (d12.0 / *e2 - d01.0 / *e1) * k,
                    (d12.1 / *e2 - d01.1 / *e1) * k,
                );
                let arc = &x[6..12];
                let rad = (q0.0 - arc[0], q0.1 - arc[1]);
                out[0] = dot(c1, rad) / (norm(c1) * norm(rad));
                let n1 = norm(c1);
                let mut kappa = cross(c1, c2) / (n1 * n1 * n1);
                if *rational {
                    // The end of a clamped NURBS: κ scales by w0 w2 / w1² (Farin).
                    let w = &x[12..15];
                    kappa = kappa * w[0] * w[2] / w[1].sq();
                }
                out[1] = kappa * Round::Arc.radius(arc) - *sigma;
            }
            Op::EqualLength => out[0] = norm(seg(&x[0..4])) - norm(seg(&x[4..8])),
            Op::EqualRadius { a, b } => out[0] = a.radius(&x[..a.len()]) - b.radius(&x[a.len()..]),
            Op::EqualEllipse => {
                out[0] = x[0] - x[2];
                out[1] = x[1] - x[3];
            }
            Op::EllipseValue { measure, v } => {
                use crate::system::EllipseMeasure as M;
                let (a, b) = (x[0], x[1]);
                out[0] = match measure {
                    M::SemiMajor => a,
                    M::SemiMinor => b,
                    M::MajorDiameter => a * 2.0,
                    M::MinorDiameter => b * 2.0,
                    M::FocalDistance => {
                        let d = a.sq() - b.sq();
                        let d = if d.val() < 0.0 { -d } else { d };
                        d.sqrt() * 2.0
                    }
                } - *v;
            }
            Op::Coradial { a, b } => {
                let xb = &x[a.len()..];
                out[0] = x[0] - xb[0];
                out[1] = x[1] - xb[1];
                out[2] = a.radius(&x[..a.len()]) - b.radius(xb);
            }
            Op::Midpoint => {
                out[0] = x[0] - (x[2] + x[4]) * 0.5;
                out[1] = x[1] - (x[3] + x[5]) * 0.5;
            }
            Op::Symmetric => {
                let m = ((x[0] + x[2]) * 0.5, (x[1] + x[3]) * 0.5);
                let l = &x[4..8];
                out[0] = signed_dist(l, m);
                let d = seg(l);
                out[1] = dot((x[2] - x[0], x[3] - x[1]), d) / norm(d);
            }
            Op::Collinear => {
                let l = &x[0..4];
                out[0] = signed_dist(l, (x[4], x[5]));
                out[1] = signed_dist(l, (x[6], x[7]));
            }
            Op::FixValues { targets } => {
                for (i, t) in targets.iter().enumerate() {
                    out[i] = x[i] - *t;
                }
            }
            Op::FixArc { targets, tangent } => {
                for i in 0..4 {
                    out[i] = x[i] - targets[i];
                }
                out[4] = (x[4] - targets[4]) * tangent[0] + (x[5] - targets[5]) * tangent[1];
            }
            Op::FixBody { targets } => {
                for i in 0..3 {
                    out[i] = x[i] - targets[i];
                }
                let q = unit_quat(&x[3..7]);
                let q0 = [
                    S::cst(targets[3]),
                    S::cst(targets[4]),
                    S::cst(targets[5]),
                    S::cst(targets[6]),
                ];
                let v = rel_vec(&q0, &q);
                out[3] = v[0];
                out[4] = v[1];
                out[5] = v[2];
            }
            Op::FixEllipticalArc { targets, tangents } => {
                for i in 0..5 {
                    out[i] = x[i] - targets[i];
                }
                out[5] = (x[5] - targets[5]) * tangents[0] + (x[6] - targets[6]) * tangents[1];
                out[6] = (x[7] - targets[7]) * tangents[2] + (x[8] - targets[8]) * tangents[3];
            }
            Op::PointOnLine => out[0] = signed_dist(&x[2..6], (x[0], x[1])),
            Op::PointOnRound { round } => {
                let c = &x[2..];
                out[0] = (x[0] - c[0]).hypot(x[1] - c[1]) - round.radius(c);
            }
            Op::PointOnEllipse => out[0] = ell_implicit(&x[2..7], (x[0], x[1])),
            Op::PointOnSpline {
                degree,
                knots,
                rational,
                periodic,
            } => {
                let m = spline_count(x.len(), *rational);
                let cps = &x[3..3 + 2 * m];
                let w = rational.then(|| &x[3 + 2 * m..3 + 3 * m]);
                let (cx, cy) = spline_eval(*degree, knots, cps, w, x[2], *periodic);
                out[0] = x[0] - cx;
                out[1] = x[1] - cy;
            }
            Op::DistancePP { d } => out[0] = (x[0] - x[2]).hypot(x[1] - x[3]) - *d,
            Op::DistancePL { d, side } => out[0] = signed_dist(&x[2..6], (x[0], x[1])) * *side - *d,
            Op::Angle { theta } => {
                let (a, b) = (seg(&x[0..4]), seg(&x[4..8]));
                out[0] = wrap(cross(a, b).atan2(dot(a, b)) - *theta);
            }
            Op::RadiusValue { round, scale, v } => out[0] = round.radius(x) * *scale - *v,
            Op::Length { v } => out[0] = norm(seg(&x[0..4])) - *v,
            Op::VarValue { v } => out[0] = x[0] - *v,
            Op::ArcRadius => {
                out[0] = (x[2] - x[0]).hypot(x[3] - x[1]) - (x[4] - x[0]).hypot(x[5] - x[1]);
            }
            Op::QuatNorm => {
                out[0] = (x[0].sq() + x[1].sq() + x[2].sq() + x[3].sq() - 1.0) * 0.5;
            }
            Op::EllipticalArcEnds => {
                let e = &x[0..5];
                out[0] = ell_implicit(e, (x[5], x[6]));
                out[1] = ell_implicit(e, (x[7], x[8]));
            }
            Op::Mate { kind, la, lb } => eval_mate(kind, la, lb, x, out),
        }
    }

    /// Residual values at `x`.
    pub(crate) fn values(&self, x: &[f64], out: &mut [f64]) {
        self.eval(x, out);
    }

    /// The layout entries the residual reads at `x`, ascending, when that is a strict subset
    /// known in advance (a spline point reads only its active knot span).
    pub(crate) fn support(&self, x: &[f64]) -> Option<Vec<usize>> {
        match self {
            Op::PointOnSpline {
                degree,
                knots,
                rational,
                periodic,
            } => {
                let m = spline_count(x.len(), *rational);
                let s = spline_span(*degree, knots, m, x[2], *periodic);
                let first = s - degree;
                let mut out = vec![0, 1, 2];
                for j in first..=s {
                    out.extend([3 + 2 * j, 4 + 2 * j]);
                }
                if *rational {
                    out.extend((first..=s).map(|j| 3 + 2 * m + j));
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// Residual values and the dense `rows × x.len()` Jacobian block (row-major).
    pub(crate) fn jacobian(&self, x: &[f64], vals: &mut [f64], jac: &mut [f64]) {
        let n = x.len();
        if let Op::FixValues { targets } = self {
            // Linear: the identity block.
            jac.iter_mut().for_each(|v| *v = 0.0);
            for (i, t) in targets.iter().enumerate() {
                vals[i] = x[i] - t;
                jac[i * n + i] = 1.0;
            }
            return;
        }
        if let Some(support) = self.support(x) {
            return jac_support(self, x, &support, vals, jac);
        }
        match n {
            0..=4 => jac_n::<4>(self, x, vals, jac),
            5..=8 => jac_n::<8>(self, x, vals, jac),
            9..=16 => jac_n::<16>(self, x, vals, jac),
            17..=32 => jac_n::<32>(self, x, vals, jac),
            33..=64 => jac_n::<64>(self, x, vals, jac),
            _ => {
                let all: Vec<usize> = (0..n).collect();
                jac_support(self, x, &all, vals, jac)
            }
        }
    }
}

/// Chunked forward-mode AD seeded only on `support`: `ceil(|support| / 16)` evaluations, each
/// carrying at most 16 derivative lanes. Columns outside the support are zero.
fn jac_support(op: &Op, x: &[f64], support: &[usize], vals: &mut [f64], jac: &mut [f64]) {
    jac.iter_mut().for_each(|v| *v = 0.0);
    if support.is_empty() {
        op.values(x, vals);
        return;
    }
    for chunk in support.chunks(16) {
        match chunk.len() {
            0..=4 => jac_chunk::<4>(op, x, chunk, vals, jac),
            5..=8 => jac_chunk::<8>(op, x, chunk, vals, jac),
            _ => jac_chunk::<16>(op, x, chunk, vals, jac),
        }
    }
}

fn jac_chunk<const N: usize>(
    op: &Op,
    x: &[f64],
    chunk: &[usize],
    vals: &mut [f64],
    jac: &mut [f64],
) {
    let n = x.len();
    let mut xs: Vec<Dual<N>> = x.iter().map(|&v| Dual::cst(v)).collect();
    for (k, &i) in chunk.iter().enumerate() {
        xs[i] = Dual::var(x[i], k);
    }
    let rows = op.rows();
    let mut out = vec![Dual::<N>::cst(0.0); rows];
    op.eval(&xs, &mut out);
    for r in 0..rows {
        vals[r] = out[r].v;
        for (k, &i) in chunk.iter().enumerate() {
            jac[r * n + i] = out[r].d[k];
        }
    }
}

fn jac_n<const N: usize>(op: &Op, x: &[f64], vals: &mut [f64], jac: &mut [f64]) {
    let n = x.len();
    assert!(n <= N, "constraint reads {n} variables, more than {N}");
    let xs: Vec<Dual<N>> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| Dual::var(v, i))
        .collect();
    let rows = op.rows();
    let mut out = vec![Dual::<N>::cst(0.0); rows];
    op.eval(&xs, &mut out);
    for r in 0..rows {
        vals[r] = out[r].v;
        jac[r * n..(r + 1) * n].copy_from_slice(&out[r].d[..n]);
    }
}

// ------------------------------------------------------------------ geometry helpers

fn seg<S: Real>(x: &[S]) -> (S, S) {
    (x[2] - x[0], x[3] - x[1])
}
fn cross<S: Real>(a: (S, S), b: (S, S)) -> S {
    a.0 * b.1 - a.1 * b.0
}
fn dot<S: Real>(a: (S, S), b: (S, S)) -> S {
    a.0 * b.0 + a.1 * b.1
}
fn norm<S: Real>(a: (S, S)) -> S {
    a.0.hypot(a.1)
}
/// Signed distance from `p` to the line through `l = [a, b]` (positive on the left of a→b).
fn signed_dist<S: Real>(l: &[S], p: (S, S)) -> S {
    let d = seg(l);
    cross(d, (p.0 - l[0], p.1 - l[1])) / norm(d)
}
/// Wraps an angle to (-π, π] by a constant shift (the derivative is unchanged).
fn wrap<S: Real>(a: S) -> S {
    let k = ((a.val() + PI) / (2.0 * PI)).floor();
    let shifted = a - k * 2.0 * PI;
    if shifted.val() <= -PI {
        shifted + 2.0 * PI
    } else {
        shifted
    }
}

/// The number of (wrapped) control points in a point-on-spline layout of length `len`.
fn spline_count(len: usize, rational: bool) -> usize {
    (len - 3) / if rational { 3 } else { 2 }
}

/// A periodic parameter wrapped into `[knots[p], knots[m])` by a constant shift (the
/// derivative is unchanged).
fn wrap_parameter<S: Real>(degree: usize, knots: &[f64], m: usize, t: S) -> S {
    let (t0, t1) = (knots[degree], knots[m]);
    let period = t1 - t0;
    let k = ((t.val() - t0) / period).floor();
    if k == 0.0 || !k.is_finite() {
        t
    } else {
        t - k * period
    }
}

/// The knot span `s ∈ [p, m - 1]` (`knots[s] ≤ t < knots[s + 1]`, the last span closed) for
/// the primal parameter, clamped to the domain (wrapped first when periodic). Binary search.
pub(crate) fn spline_span(degree: usize, knots: &[f64], m: usize, t: f64, periodic: bool) -> usize {
    let t = if periodic {
        wrap_parameter(degree, knots, m, t)
    } else {
        t
    };
    let tv = t.clamp(knots[degree], knots[m]);
    degree + knots[degree + 1..m].partition_point(|&k| k <= tv)
}

/// A point on a B-spline by de Boor's algorithm over the active span (homogeneous when
/// rational). `cps` holds `m` points as `[x0, y0, x1, y1, …]` (the wrapped sequence for a
/// periodic spline) and `weights` one weight per point. Outside a clamped domain the end
/// polynomial extends; a periodic parameter wraps.
pub(crate) fn spline_eval<S: Real>(
    degree: usize,
    knots: &[f64],
    cps: &[S],
    weights: Option<&[S]>,
    t: S,
    periodic: bool,
) -> (S, S) {
    let m = cps.len() / 2;
    let t = if periodic {
        wrap_parameter(degree, knots, m, t)
    } else {
        t
    };
    let s = spline_span(degree, knots, m, t.val(), false);
    let first = s - degree;
    let w = |j: usize| weights.map_or(S::cst(1.0), |w| w[j]);
    // De Boor works on degree + 1 homogeneous points: on the stack up to degree 15.
    const STACK: usize = 16;
    let mut sx = [S::cst(0.0); STACK];
    let mut sy = [S::cst(0.0); STACK];
    let mut sw = [S::cst(0.0); STACK];
    let (mut hx, mut hy, mut hw);
    let (dx, dy, dw): (&mut [S], &mut [S], &mut [S]) = if degree < STACK {
        (&mut sx[..=degree], &mut sy[..=degree], &mut sw[..=degree])
    } else {
        hx = vec![S::cst(0.0); degree + 1];
        hy = vec![S::cst(0.0); degree + 1];
        hw = vec![S::cst(0.0); degree + 1];
        (&mut hx[..], &mut hy[..], &mut hw[..])
    };
    for j in 0..=degree {
        let wj = w(first + j);
        dx[j] = cps[2 * (first + j)] * wj;
        dy[j] = cps[2 * (first + j) + 1] * wj;
        dw[j] = wj;
    }
    for r in 1..=degree {
        for j in (r..=degree).rev() {
            let i = j + first;
            let den = knots[i + degree + 1 - r] - knots[i];
            let alpha = if den > 0.0 {
                (t - knots[i]) / den
            } else {
                S::cst(0.0)
            };
            let one = S::cst(1.0) - alpha;
            dx[j] = dx[j - 1] * one + dx[j] * alpha;
            dy[j] = dy[j - 1] * one + dy[j] * alpha;
            if weights.is_some() {
                dw[j] = dw[j - 1] * one + dw[j] * alpha;
            }
        }
    }
    if weights.is_some() {
        (dx[degree] / dw[degree], dy[degree] / dw[degree])
    } else {
        (dx[degree], dy[degree])
    }
}

/// A spline's evaluation data from current values: degree, knots, wrapped control-point
/// coordinates, wrapped weights (when rational), periodic flag.
pub(crate) type SplineData = (usize, Vec<f64>, Vec<f64>, Option<Vec<f64>>, bool);

pub(crate) fn spline_data(sys: &System, curve: EntityId, values: &[f64]) -> Option<SplineData> {
    let EntityKind::BSpline {
        degree,
        control_points,
        knots,
        weights,
        periodic,
    } = &sys.entity(curve).kind
    else {
        return None;
    };
    let (pts, ws) = wrapped_spline(*degree, control_points, weights, *periodic);
    let point = |p: EntityId| match sys.entity(p).kind {
        EntityKind::Point2 { x, y } => [values[x.index()], values[y.index()]],
        _ => unreachable!("validated point"),
    };
    let cps: Vec<f64> = pts.iter().flat_map(|c| point(*c)).collect();
    let w = (!ws.is_empty()).then(|| ws.iter().map(|v| values[v.index()]).collect());
    Some((*degree, knots.clone(), cps, w, *periodic))
}

/// The evaluation sequence of a spline's control points and weights (the first `degree`
/// repeated at the end when periodic).
pub(crate) fn wrapped_spline(
    degree: usize,
    control_points: &[EntityId],
    weights: &[VarId],
    periodic: bool,
) -> (Vec<EntityId>, Vec<VarId>) {
    let mut pts = control_points.to_vec();
    let mut ws = weights.to_vec();
    if periodic {
        pts.extend_from_slice(&control_points[..degree]);
        if !weights.is_empty() {
            ws.extend_from_slice(&weights[..degree]);
        }
    }
    (pts, ws)
}

/// The parameter of the spline point closest to `p` (deterministic, used to seed the curve
/// parameter of a point-on-spline constraint).
///
/// Each knot span lies in the convex hull of its `degree + 1` control points (weights are
/// positive), so the distance from `p` to the span's control bounding box bounds the span from
/// below. Spans are visited nearest bound first and sampled (at least 8 samples each, 256 over
/// the whole curve) until the bound exceeds the best distance found; a golden-section search
/// then refines around the best sample. The cost grows with the spans near `p`, not with the
/// control-point count.
pub(crate) fn closest_spline_parameter(sys: &System, curve: EntityId, p: [f64; 2]) -> f64 {
    let values: Vec<f64> = sys.vars.iter().map(|v| v.value).collect();
    let Some((degree, knots, cps, w, periodic)) = spline_data(sys, curve, &values) else {
        return 0.0;
    };
    let m = cps.len() / 2;
    let (t0, t1) = (knots[degree], knots[m]);
    let d2 = |t: f64| {
        let (x, y) = spline_eval(degree, &knots, &cps, w.as_deref(), t, periodic);
        (x - p[0]).powi(2) + (y - p[1]).powi(2)
    };
    let spans: Vec<usize> = (degree..m).filter(|&s| knots[s + 1] > knots[s]).collect();
    let per_span = 8.max(256 / spans.len().max(1));
    let mut order: Vec<(f64, usize)> = spans
        .iter()
        .map(|&s| {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for j in s - degree..=s {
                for k in 0..2 {
                    lo[k] = lo[k].min(cps[2 * j + k]);
                    hi[k] = hi[k].max(cps[2 * j + k]);
                }
            }
            let gap = |k: usize| (lo[k] - p[k]).max(p[k] - hi[k]).max(0.0);
            (gap(0).powi(2) + gap(1).powi(2), s)
        })
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut best = (f64::INFINITY, t0, (t1 - t0) / 256.0);
    for (bound, s) in order {
        if bound > best.0 {
            break;
        }
        let (a, b) = (knots[s], knots[s + 1]);
        let h = (b - a) / per_span as f64;
        for i in 0..=per_span {
            let t = a + h * i as f64;
            let d = d2(t);
            if d < best.0 || (d == best.0 && t < best.1) {
                best = (d, t, h);
            }
        }
    }
    let h = best.2;
    let (mut a, mut b) = (best.1 - h, best.1 + h);
    if !periodic {
        a = a.max(t0);
        b = b.min(t1);
    }
    let g = 0.5 * (5f64.sqrt() - 1.0);
    for _ in 0..60 {
        let c = b - g * (b - a);
        let d = a + g * (b - a);
        if d2(c) < d2(d) {
            b = d;
        } else {
            a = c;
        }
    }
    let t = 0.5 * (a + b);
    if d2(t) < best.0 {
        t
    } else {
        best.1
    }
}

// ------------------------------------------------------------------ ellipse helpers

/// Major and minor axis directions of `[cx, cy, a, b, rot]`.
fn ell_axes<S: Real>(e: &[S]) -> ((S, S), (S, S)) {
    let (s, c) = (e[4].sin(), e[4].cos());
    ((c, s), (-s, c))
}

/// The ellipse point at eccentric angle `t`.
fn ell_point<S: Real>(e: &[S], t: S) -> (S, S) {
    let (u, v) = ell_axes(e);
    let (ca, sb) = (e[2] * t.cos(), e[3] * t.sin());
    (e[0] + u.0 * ca + v.0 * sb, e[1] + u.1 * ca + v.1 * sb)
}

/// `dE/dt` at eccentric angle `t` (counter-clockwise).
fn ell_tangent<S: Real>(e: &[S], t: S) -> (S, S) {
    let (u, v) = ell_axes(e);
    let (sa, cb) = (-(e[2] * t.sin()), e[3] * t.cos());
    (u.0 * sa + v.0 * cb, u.1 * sa + v.1 * cb)
}

/// The counter-clockwise tangent direction at (or nearest) `p`: perpendicular to the
/// gradient of the implicit equation, scaled by `ab` to stay well sized.
fn ell_tangent_at<S: Real>(e: &[S], p: (S, S)) -> (S, S) {
    let (u, v) = ell_axes(e);
    let d = (p.0 - e[0], p.1 - e[1]);
    let (lu, lv) = (dot(d, u), dot(d, v));
    let (a, b) = (e[2], e[3]);
    // Local CCW tangent (-v/b², u/a²) times a·b.
    let (tu, tv) = (-(lv * a / b), lu * b / a);
    (u.0 * tu + v.0 * tv, u.1 * tu + v.1 * tv)
}

/// The scaled implicit equation: zero on the ellipse, about the distance to it nearby.
fn ell_implicit<S: Real>(e: &[S], p: (S, S)) -> S {
    let (u, v) = ell_axes(e);
    let d = (p.0 - e[0], p.1 - e[1]);
    let (a, b) = (e[2], e[3]);
    ((dot(d, u) / a).sq() + (dot(d, v) / b).sq() - 1.0) * ((a + b) * 0.25)
}

fn ellipse_values(sys: &System, e: EntityId) -> Option<[f64; 5]> {
    match &sys.entity(e).kind {
        EntityKind::Ellipse {
            center,
            semi_major,
            semi_minor,
            rotation,
        }
        | EntityKind::EllipticalArc {
            center,
            semi_major,
            semi_minor,
            rotation,
            ..
        } => {
            let c = sys.point(*center);
            Some([
                c[0],
                c[1],
                sys.value(*semi_major),
                sys.value(*semi_minor),
                sys.value(*rotation),
            ])
        }
        _ => None,
    }
}

fn round_values_sys(sys: &System, e: EntityId) -> Option<([f64; 2], f64)> {
    match &sys.entity(e).kind {
        EntityKind::Circle { center, radius } => Some((sys.point(*center), sys.value(*radius))),
        EntityKind::Arc { center, start, .. } => {
            let (c, s) = (sys.point(*center), sys.point(*start));
            Some((c, (s[0] - c[0]).hypot(s[1] - c[1])))
        }
        _ => None,
    }
}

/// The angle in `[0, 2π)` minimising `objective` among the zeros of `slope` (its derivative
/// up to a positive factor), found by sign changes over 720 samples and bisection. Falls back
/// to the best sample when `slope` never changes sign.
fn best_stationary_angle(objective: impl Fn(f64) -> f64, slope: impl Fn(f64) -> f64) -> f64 {
    let n = 720;
    let h = 2.0 * PI / n as f64;
    let mut best = (f64::INFINITY, 0.0);
    let mut fallback = (f64::INFINITY, 0.0);
    let mut prev = slope(0.0);
    for i in 0..n {
        let (a, b) = (h * i as f64, h * (i + 1) as f64);
        let next = slope(b);
        let fa = objective(a);
        if fa < fallback.0 {
            fallback = (fa, a);
        }
        if (prev <= 0.0) != (next <= 0.0) {
            let (mut lo, mut hi, mut slo) = (a, b, prev);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let sm = slope(mid);
                if (sm <= 0.0) == (slo <= 0.0) {
                    lo = mid;
                    slo = sm;
                } else {
                    hi = mid;
                }
            }
            let t = 0.5 * (lo + hi);
            let f = objective(t);
            if f < best.0 {
                best = (f, t);
            }
        }
        prev = next;
    }
    if best.0.is_finite() {
        best.1
    } else {
        fallback.1
    }
}

/// Starting contact angles for a tangency that carries them: none for line or round pairs,
/// one (on the ellipse) for round–ellipse, two for ellipse–ellipse (in argument order).
///
/// The contact point of a round–ellipse tangency is a point where the distance from the
/// circle's centre to the ellipse is stationary; the one whose distance is closest to the
/// radius is taken. For two ellipses, the contact on the first is a stationary point of the
/// second's implicit equation along it (closest to zero), and the contact on the second is
/// the closest point to it.
pub(crate) fn seed_tangent_angles(sys: &System, a: EntityId, b: EntityId) -> Vec<f64> {
    let ea = ellipse_values(sys, a);
    let eb = ellipse_values(sys, b);
    let closest_on = |e: [f64; 5], p: [f64; 2]| {
        best_stationary_angle(
            |t| {
                let q = ell_point(&e, t);
                (q.0 - p[0]).hypot(q.1 - p[1])
            },
            |t| {
                let q = ell_point(&e, t);
                dot((q.0 - p[0], q.1 - p[1]), ell_tangent(&e, t))
            },
        )
    };
    match (ea, eb) {
        (Some(ea), Some(eb)) => {
            let h = 1e-6;
            let f = |t: f64| ell_implicit(&eb, ell_point(&ea, t));
            let ta = best_stationary_angle(|t| f(t).abs(), |t| f(t + h) - f(t - h));
            let pa = ell_point(&ea, ta);
            vec![ta, closest_on(eb, [pa.0, pa.1])]
        }
        (Some(e), None) | (None, Some(e)) => {
            let other = if ea.is_some() { b } else { a };
            match round_values_sys(sys, other) {
                Some((c, r)) => {
                    let t = best_stationary_angle(
                        |t| {
                            let q = ell_point(&e, t);
                            ((q.0 - c[0]).hypot(q.1 - c[1]) - r).abs()
                        },
                        |t| {
                            let q = ell_point(&e, t);
                            dot((q.0 - c[0], q.1 - c[1]), ell_tangent(&e, t))
                        },
                    );
                    vec![t]
                }
                None => Vec::new(),
            }
        }
        (None, None) => Vec::new(),
    }
}

// ------------------------------------------------------------------ 3D helpers

fn unit_quat<S: Real>(q: &[S]) -> [S; 4] {
    let n = (q[0].sq() + q[1].sq() + q[2].sq() + q[3].sq()).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn qmul<S: Real>(a: &[S; 4], b: &[S; 4]) -> [S; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// The vector part of `conj(a) ⊗ b`: zero exactly when the rotations agree.
fn rel_vec<S: Real>(a: &[S; 4], b: &[S; 4]) -> [S; 3] {
    let ca = [a[0], -a[1], -a[2], -a[3]];
    let r = qmul(&ca, b);
    [r[1], r[2], r[3]]
}

fn rotate<S: Real>(q: &[S; 4], v: [S; 3]) -> [S; 3] {
    // v' = v + 2w (u × v) + 2 u × (u × v)
    let u = [q[1], q[2], q[3]];
    let t = cross3(u, v);
    let t = [t[0] * 2.0, t[1] * 2.0, t[2] * 2.0];
    let ut = cross3(u, t);
    [
        v[0] + q[0] * t[0] + ut[0],
        v[1] + q[0] * t[1] + ut[1],
        v[2] + q[0] * t[2] + ut[2],
    ]
}

fn cross3<S: Real>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot3<S: Real>(a: [S; 3], b: [S; 3]) -> S {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub3<S: Real>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// A connector's world origin, quaternion and axes `[X, Y, Z]` from its body's 7 variables.
fn connector_world<S: Real>(x: &[S], local: &Frame) -> ([S; 3], [S; 4], [[S; 3]; 3]) {
    let qb = unit_quat(&x[3..7]);
    let lo = local.origin.map(S::cst);
    let off = rotate(&qb, lo);
    let origin = [x[0] + off[0], x[1] + off[1], x[2] + off[2]];
    let ql = local.rotation.map(S::cst);
    let q = qmul(&qb, &ql);
    let (w, a, b, c) = (q[0], q[1], q[2], q[3]);
    let one = S::cst(1.0);
    let xa = [
        one - (b.sq() + c.sq()) * 2.0,
        (a * b + w * c) * 2.0,
        (a * c - w * b) * 2.0,
    ];
    let ya = [
        (a * b - w * c) * 2.0,
        one - (a.sq() + c.sq()) * 2.0,
        (b * c + w * a) * 2.0,
    ];
    let za = [
        (a * c + w * b) * 2.0,
        (b * c - w * a) * 2.0,
        one - (a.sq() + b.sq()) * 2.0,
    ];
    (origin, q, [xa, ya, za])
}

fn eval_mate<S: Real>(kind: &MateOp, la: &Frame, lb: &Frame, x: &[S], out: &mut [S]) {
    let (oa, qa, [xa, ya, za]) = connector_world(&x[0..7], la);
    let (ob, qb, [xb, yb, zb]) = connector_world(&x[7..14], lb);
    let d = sub3(ob, oa);
    match *kind {
        MateOp::Fasten => {
            out[..3].copy_from_slice(&d);
            let v = rel_vec(&qa, &qb);
            out[3..6].copy_from_slice(&v);
        }
        MateOp::Revolute => {
            out[..3].copy_from_slice(&d);
            out[3] = dot3(xa, zb);
            out[4] = dot3(ya, zb);
        }
        MateOp::Slider => {
            out[0] = dot3(xa, zb);
            out[1] = dot3(ya, zb);
            out[2] = dot3(d, xa);
            out[3] = dot3(d, ya);
            out[4] = dot3(xa, yb);
        }
        MateOp::Cylindrical => {
            out[0] = dot3(xa, zb);
            out[1] = dot3(ya, zb);
            out[2] = dot3(d, xa);
            out[3] = dot3(d, ya);
        }
        MateOp::Planar => {
            out[0] = dot3(xa, zb);
            out[1] = dot3(ya, zb);
            out[2] = dot3(d, za);
        }
        MateOp::Ball => out[..3].copy_from_slice(&d),
        MateOp::Parallel => {
            out[0] = dot3(xa, zb);
            out[1] = dot3(ya, zb);
        }
        MateOp::TangentSphere { r } => out[0] = dot3(d, za) - r,
        MateOp::TangentCylinder { r } => {
            out[0] = dot3(za, zb);
            out[1] = dot3(d, za) - r;
        }
        MateOp::DistancePoint { d: v } => out[0] = dot3(d, d).sqrt() - v,
        MateOp::DistancePlane { d: v } => out[0] = dot3(d, za) - v,
        MateOp::AngleAboutZ { theta } => {
            let s = dot3(cross3(xa, xb), za);
            let c = dot3(xa, xb);
            out[0] = wrap(s.atan2(c) - theta);
        }
        MateOp::AngleBetweenZ { theta } => {
            let cr = cross3(za, zb);
            out[0] = dot3(cr, cr).sqrt().atan2(dot3(za, zb)) - theta;
        }
        MateOp::PinSlot => {
            out[0] = dot3(d, yb);
            out[1] = dot3(d, zb);
            out[2] = dot3(za, xb);
            out[3] = dot3(za, yb);
        }
        MateOp::Screw { lead } => {
            out[0] = dot3(xa, zb);
            out[1] = dot3(ya, zb);
            out[2] = dot3(d, xa);
            out[3] = dot3(d, ya);
            let theta = dot3(cross3(xa, xb), za).atan2(dot3(xa, xb));
            out[4] = dot3(d, za) - theta * (lead / std::f64::consts::TAU);
        }
        MateOp::Gear {
            ratio,
            xa0,
            za0,
            xb0,
            zb0,
        } => {
            let ta = turned(xa0, za0, xa);
            let tb = turned(xb0, zb0, xb);
            out[0] = ta + tb * ratio;
        }
        MateOp::RackPinion {
            travel,
            xa0,
            za0,
            ob0,
            xb0,
        } => {
            let ta = turned(xa0, za0, xa);
            let moved = dot3(sub3(ob, ob0.map(S::cst)), xb0.map(S::cst));
            out[0] = moved - ta * (travel / std::f64::consts::TAU);
        }
        MateOp::Cam { ref radii } => {
            // The follower in the cam's plane: its distance from the cam's axis meets the pitch
            // curve's radius at its angle.
            let (px, py) = (dot3(d, xa), dot3(d, ya));
            out[0] = px.hypot(py) - cam_radius(radii, py.atan2(px));
        }
        MateOp::LinearCoupler {
            ratio,
            oa0,
            za0,
            ob0,
            zb0,
        } => {
            let ma = dot3(sub3(oa, oa0.map(S::cst)), za0.map(S::cst));
            let mb = dot3(sub3(ob, ob0.map(S::cst)), zb0.map(S::cst));
            out[0] = mb - ma * ratio;
        }
        MateOp::UniversalJoint { joint_point } => {
            if joint_point {
                out[..3].copy_from_slice(&d);
                out[3] = dot3(xa, xb);
            } else {
                out[0] = dot3(xa, xb);
            }
        }
        MateOp::Symmetric { ref lc } => {
            let (op, _, [_, _, zp]) = connector_world(&x[14..21], lc);
            let h = dot3(sub3(oa, op), zp) * 2.0;
            for i in 0..3 {
                out[i] = oa[i] - zp[i] * h - ob[i];
            }
        }
        MateOp::Path { ref points } => {
            let q = [dot3(d, xa), dot3(d, ya), dot3(d, za)];
            let (p0, n1, n2) = path_segment(points, q.map(|c| c.val()));
            let r = sub3(q, p0.map(S::cst));
            out[0] = dot3(r, n1.map(S::cst));
            out[1] = dot3(r, n2.map(S::cst));
        }
    }
}

/// The path segment nearest `q`: its start and two unit normals square to it. The choice is
/// held fixed for the derivative, which is exact inside a segment (the offset along it doesn't
/// enter the rows).
fn path_segment(points: &[[f64; 3]], q: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let unit = |a: [f64; 3]| {
        let n = dot(a, a).sqrt();
        a.map(|c| c / n)
    };
    let mut best = (f64::INFINITY, 0);
    for (i, w) in points.windows(2).enumerate() {
        let e = sub(w[1], w[0]);
        let len2 = dot(e, e);
        if len2 == 0.0 {
            continue;
        }
        let t = (dot(sub(q, w[0]), e) / len2).clamp(0.0, 1.0);
        let r = sub(
            q,
            [w[0][0] + e[0] * t, w[0][1] + e[1] * t, w[0][2] + e[2] * t],
        );
        let d2 = dot(r, r);
        if d2 < best.0 {
            best = (d2, i);
        }
    }
    let (p0, p1) = (points[best.1], points[best.1 + 1]);
    let t = unit(sub(p1, p0));
    // The axis least along the segment gives the first normal.
    let k = (0..3)
        .min_by(|&i, &j| t[i].abs().total_cmp(&t[j].abs()))
        .unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    let n1 = unit(cross(t, e));
    (p0, n1, cross(t, n1))
}

/// The cam's pitch radius at angle `phi` from its X: a periodic cubic (Catmull-Rom) through
/// the station radii, evenly spaced from 0, continuous in slope all the way round.
fn cam_radius<S: Real>(radii: &[f64], phi: S) -> S {
    let n = radii.len();
    let step = std::f64::consts::TAU / n as f64;
    let u = phi.val().rem_euclid(std::f64::consts::TAU) / step;
    let k = (u.floor() as usize).min(n - 1);
    // `t` runs 0..1 between stations k and k + 1, carrying phi's derivative.
    let t = (phi - (phi.val() - (u - k as f64) * step)) / step;
    let at = |i: usize| radii[(k + n + i - 1) % n];
    let (p0, p1, p2, p3) = (at(0), at(1), at(2), at(3));
    let (m1, m2) = ((p2 - p0) / 2.0, (p3 - p1) / 2.0);
    let (t2, t3) = (t * t, t * t * t);
    (t3 * 2.0 - t2 * 3.0 + 1.0) * p1
        + (t3 - t2 * 2.0 + t) * m1
        + (t2 * 3.0 - t3 * 2.0) * p2
        + (t3 - t2) * m2
}

/// How far `x` has turned about the fixed axis `z0` from the fixed direction `x0` (radians).
fn turned<S: Real>(x0: [f64; 3], z0: [f64; 3], x: [S; 3]) -> S {
    let (x0, z0) = (x0.map(S::cst), z0.map(S::cst));
    dot3(cross3(x0, x), z0).atan2(dot3(x0, x))
}

// ------------------------------------------------------------------ lowering

/// A constraint lowered for one solve.
#[derive(Clone, Debug)]
pub(crate) struct Compiled {
    pub id: ConstraintId,
    /// The variables the op reads, in layout order (may repeat).
    pub vars: Vec<VarId>,
    pub op: Op,
    /// `Some((min, max))` for a range target: the row is active only when a bound is hit.
    pub range: Option<(f64, f64)>,
    pub reference: bool,
    pub internal: bool,
    /// The unit of each residual row (for unit-aware scaling).
    pub units: Vec<Unit>,
}

struct Lower<'a> {
    sys: &'a System,
    values: &'a [f64],
    vars: Vec<VarId>,
}

impl<'a> Lower<'a> {
    fn v(&self, var: VarId) -> f64 {
        self.values[var.index()]
    }
    fn kind(&self, e: EntityId) -> &EntityKind {
        &self.sys.entity(e).kind
    }
    fn point(&mut self, p: EntityId) {
        match self.kind(p) {
            EntityKind::Point2 { x, y } => {
                let (x, y) = (*x, *y);
                self.vars.extend([x, y]);
            }
            _ => unreachable!("validated point"),
        }
    }
    fn pv(&self, p: EntityId) -> [f64; 2] {
        match self.kind(p) {
            EntityKind::Point2 { x, y } => [self.v(*x), self.v(*y)],
            _ => unreachable!("validated point"),
        }
    }
    fn line(&mut self, l: EntityId) -> [EntityId; 2] {
        match self.kind(l) {
            EntityKind::Line { start, end } => {
                let (a, b) = (*start, *end);
                self.point(a);
                self.point(b);
                [a, b]
            }
            _ => unreachable!("validated line"),
        }
    }
    fn round(&mut self, e: EntityId) -> Round {
        match self.kind(e).clone() {
            EntityKind::Circle { center, radius } => {
                self.point(center);
                self.vars.push(radius);
                Round::Circle
            }
            EntityKind::Arc { center, start, end } => {
                self.point(center);
                self.point(start);
                self.point(end);
                Round::Arc
            }
            _ => unreachable!("validated round"),
        }
    }
    /// Centre and radius values of a round entity.
    fn round_values(&self, e: EntityId) -> ([f64; 2], f64) {
        match self.kind(e) {
            EntityKind::Circle { center, radius } => (self.pv(*center), self.v(*radius)),
            EntityKind::Arc { center, start, .. } => {
                let (c, s) = (self.pv(*center), self.pv(*start));
                (c, (s[0] - c[0]).hypot(s[1] - c[1]))
            }
            _ => unreachable!(),
        }
    }
    fn body(&mut self, b: EntityId) {
        match self.kind(b) {
            EntityKind::Body {
                position,
                orientation,
            } => {
                let (p, q) = (*position, *orientation);
                self.vars.extend(p);
                self.vars.extend(q);
            }
            _ => unreachable!("validated body"),
        }
    }
    fn connector(&mut self, c: EntityId) -> Frame {
        match self.kind(c).clone() {
            EntityKind::Connector { body, local } => {
                self.body(body);
                local
            }
            _ => unreachable!("validated connector"),
        }
    }
    /// Pushes `[cx, cy, a, b, rot]` of an ellipse or elliptical arc.
    fn ellipse(&mut self, e: EntityId) {
        match self.kind(e).clone() {
            EntityKind::Ellipse {
                center,
                semi_major,
                semi_minor,
                rotation,
            }
            | EntityKind::EllipticalArc {
                center,
                semi_major,
                semi_minor,
                rotation,
                ..
            } => {
                self.point(center);
                self.vars.extend([semi_major, semi_minor, rotation]);
            }
            _ => unreachable!("validated ellipse"),
        }
    }
    fn ell_values(&self, e: EntityId) -> [f64; 5] {
        match self.kind(e) {
            EntityKind::Ellipse {
                center,
                semi_major,
                semi_minor,
                rotation,
            }
            | EntityKind::EllipticalArc {
                center,
                semi_major,
                semi_minor,
                rotation,
                ..
            } => {
                let c = self.pv(*center);
                [
                    c[0],
                    c[1],
                    self.v(*semi_major),
                    self.v(*semi_minor),
                    self.v(*rotation),
                ]
            }
            _ => unreachable!("validated ellipse"),
        }
    }
    /// Pushes a spline's wrapped control points then wrapped weights.
    fn spline(&mut self, e: EntityId) -> (usize, Vec<f64>, bool, bool) {
        match self.kind(e).clone() {
            EntityKind::BSpline {
                degree,
                control_points,
                knots,
                weights,
                periodic,
            } => {
                let (pts, ws) = wrapped_spline(degree, &control_points, &weights, periodic);
                for p in pts {
                    self.point(p);
                }
                self.vars.extend(ws);
                (degree, knots, !weights.is_empty(), periodic)
            }
            _ => unreachable!("validated spline"),
        }
    }
    /// Pushes the end layout of an open curve and returns how to read it.
    fn end(&mut self, e: EntityId, end: CurveEnd) -> EndDir {
        match self.kind(e).clone() {
            EntityKind::EllipticalArc {
                start, end: stop, ..
            } => {
                self.ellipse(e);
                self.point(if end == CurveEnd::Start { start } else { stop });
                EndDir::Ellipse
            }
            EntityKind::Line { start, end: stop } => {
                self.point(start);
                self.point(stop);
                EndDir::Segment
            }
            EntityKind::Arc {
                center,
                start,
                end: stop,
            } => {
                self.point(center);
                self.point(if end == CurveEnd::Start { start } else { stop });
                EndDir::ArcRadius
            }
            EntityKind::BSpline { control_points, .. } => {
                let n = control_points.len();
                let (a, b) = match end {
                    CurveEnd::Start => (control_points[0], control_points[1]),
                    CurveEnd::End => (control_points[n - 2], control_points[n - 1]),
                };
                self.point(a);
                self.point(b);
                EndDir::Segment
            }
            _ => unreachable!("validated open curve"),
        }
    }
}

fn sign(v: f64) -> f64 {
    if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn signed_dist_f(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    (d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0])) / d[0].hypot(d[1])
}

/// Lowers every constraint of `sys` using `values` as the starting geometry for branch hints.
fn var_unit(sys: &System, v: VarId) -> Unit {
    use crate::system::VarRole as R;
    match sys.var(v).role {
        R::X | R::Y | R::Z | R::Radius | R::SemiMajor | R::SemiMinor => Unit::Length,
        _ => Unit::Ratio,
    }
}

/// Row units of a lowered op (`vars` is its layout, for ops whose rows are raw variables).
fn row_units(op: &Op, vars: &[VarId], sys: &System) -> Vec<Unit> {
    use Unit::{Length as L, Ratio as A};
    match op {
        Op::Parallel | Op::Perpendicular | Op::TangentEnds { .. } | Op::Angle { .. } => vec![A],
        Op::CurvatureG2 { .. } => vec![A, A],
        Op::QuatNorm => vec![A],
        Op::TangentRoundEllipse { .. } => vec![L, A],
        Op::TangentEllipses => vec![L, L, A],
        Op::FixValues { .. } => vars.iter().map(|&v| var_unit(sys, v)).collect(),
        Op::VarValue { .. } => vec![var_unit(sys, vars[0])],
        Op::FixBody { .. } => vec![L, L, L, A, A, A],
        Op::FixEllipticalArc { .. } => vec![L, L, L, L, A, L, L],
        Op::Mate { kind, .. } => match kind {
            MateOp::Fasten => vec![L, L, L, A, A, A],
            MateOp::Revolute => vec![L, L, L, A, A],
            MateOp::Slider => vec![A, A, L, L, A],
            MateOp::Cylindrical => vec![A, A, L, L],
            MateOp::Planar => vec![A, A, L],
            MateOp::Ball => vec![L, L, L],
            MateOp::Parallel => vec![A, A],
            MateOp::TangentSphere { .. }
            | MateOp::DistancePoint { .. }
            | MateOp::DistancePlane { .. } => vec![L],
            MateOp::TangentCylinder { .. } => vec![A, L],
            MateOp::AngleAboutZ { .. } | MateOp::AngleBetweenZ { .. } => vec![A],
            MateOp::PinSlot => vec![L, L, A, A],
            MateOp::Screw { .. } => vec![A, A, L, L, L],
            MateOp::Gear { .. } => vec![A],
            MateOp::RackPinion { .. } | MateOp::Cam { .. } | MateOp::LinearCoupler { .. } => {
                vec![L]
            }
            MateOp::UniversalJoint { joint_point: true } => vec![L, L, L, A],
            MateOp::UniversalJoint { joint_point: false } => vec![A],
            MateOp::Symmetric { .. } => vec![L, L, L],
            MateOp::Path { .. } => vec![L, L],
        },
        other => vec![L; other.rows()],
    }
}

pub(crate) fn compile(sys: &System, values: &[f64]) -> Vec<Compiled> {
    let mut out = Vec::with_capacity(sys.constraints.len());
    for (i, rec) in sys.constraints.iter().enumerate() {
        let id = ConstraintId(i as u32);
        let mut l = Lower {
            sys,
            values,
            vars: Vec::new(),
        };
        let mut range = None;
        let (op, internal) = match &rec.constraint {
            ConstraintKindRecord::Internal { kind, entity } => {
                let op = match kind {
                    InternalKind::ArcRadius => {
                        l.round(*entity);
                        Op::ArcRadius
                    }
                    InternalKind::QuaternionNorm => {
                        let EntityKind::Body { orientation, .. } = l.kind(*entity) else {
                            unreachable!()
                        };
                        l.vars.extend(*orientation);
                        Op::QuatNorm
                    }
                    InternalKind::EllipticalArcEnds => {
                        let EntityKind::EllipticalArc { start, end, .. } = l.kind(*entity).clone()
                        else {
                            unreachable!()
                        };
                        l.ellipse(*entity);
                        l.point(start);
                        l.point(end);
                        Op::EllipticalArcEnds
                    }
                };
                (op, true)
            }
            ConstraintKindRecord::User(c) => (lower_user(&mut l, c, rec, &mut range), false),
        };
        let units = row_units(&op, &l.vars, sys);
        debug_assert_eq!(units.len(), op.rows());
        out.push(Compiled {
            id,
            vars: l.vars,
            op,
            range,
            reference: rec.reference,
            internal,
            units,
        });
    }
    out
}

fn lower_user(
    l: &mut Lower<'_>,
    c: &Constraint,
    rec: &crate::system::ConstraintRecord,
    range: &mut Option<(f64, f64)>,
) -> Op {
    use Constraint as C;
    match c {
        C::Coincident(a, b) => {
            l.point(*a);
            l.point(*b);
            Op::Coincident
        }
        C::HorizontalPoints(a, b) => {
            l.point(*a);
            l.point(*b);
            Op::SameY
        }
        C::VerticalPoints(a, b) => {
            l.point(*a);
            l.point(*b);
            Op::SameX
        }
        C::Horizontal(line) => {
            l.line(*line);
            Op::SameY
        }
        C::Vertical(line) => {
            l.line(*line);
            Op::SameX
        }
        C::Parallel(a, b) => {
            l.line(*a);
            l.line(*b);
            Op::Parallel
        }
        C::Perpendicular(a, b) => {
            l.line(*a);
            l.line(*b);
            Op::Perpendicular
        }
        C::Tangent(a, b) => {
            let a_line = matches!(l.kind(*a), EntityKind::Line { .. });
            let b_line = matches!(l.kind(*b), EntityKind::Line { .. });
            let elliptic = |l: &Lower<'_>, e: EntityId| {
                matches!(
                    l.kind(e),
                    EntityKind::Ellipse { .. } | EntityKind::EllipticalArc { .. }
                )
            };
            let (a_ell, b_ell) = (elliptic(l, *a), elliptic(l, *b));
            if (a_line && b_ell) || (b_line && a_ell) {
                let (line, ell) = if a_line { (*a, *b) } else { (*b, *a) };
                let [pa, pb] = l.line(line);
                let e = l.ell_values(ell);
                let side = sign(signed_dist_f(l.pv(pa), l.pv(pb), [e[0], e[1]]));
                l.ellipse(ell);
                Op::TangentLineEllipse { side }
            } else if a_ell && b_ell {
                l.ellipse(*a);
                l.ellipse(*b);
                l.vars.extend([rec.aux[0], rec.aux[1]]);
                Op::TangentEllipses
            } else if a_ell || b_ell {
                let (round_e, ell) = if a_ell { (*b, *a) } else { (*a, *b) };
                let round = l.round(round_e);
                l.ellipse(ell);
                l.vars.push(rec.aux[0]);
                Op::TangentRoundEllipse { round }
            } else if a_line || b_line {
                let (line, round_e) = if a_line { (*a, *b) } else { (*b, *a) };
                let [pa, pb] = l.line(line);
                let (c0, _) = l.round_values(round_e);
                let side = sign(signed_dist_f(l.pv(pa), l.pv(pb), c0));
                let round = l.round(round_e);
                Op::TangentLineRound { round, side }
            } else {
                let (ca, ra) = l.round_values(*a);
                let (cb, rb) = l.round_values(*b);
                let d = (ca[0] - cb[0]).hypot(ca[1] - cb[1]);
                let internal = (d - (ra - rb).abs()).abs() < (d - (ra + rb)).abs();
                let ra_kind = l.round(*a);
                let rb_kind = l.round(*b);
                Op::TangentRounds {
                    a: ra_kind,
                    b: rb_kind,
                    internal,
                    sigma: sign(ra - rb),
                }
            }
        }
        C::TangentAtEnd { a, a_end, b, b_end } => {
            let da = l.end(*a, *a_end);
            let db = l.end(*b, *b_end);
            Op::TangentEnds { a: da, b: db }
        }
        C::CurvatureContinuity { spline, end, arc } => {
            let EntityKind::BSpline {
                degree,
                control_points,
                knots,
                weights,
                ..
            } = l.kind(*spline).clone()
            else {
                unreachable!()
            };
            let n = control_points.len() - 1;
            let p = degree;
            let (q, e1, e2, f) = match end {
                CurveEnd::Start => (
                    [control_points[0], control_points[1], control_points[2]],
                    knots[p + 1] - knots[1],
                    knots[p + 2] - knots[2],
                    knots[p + 1] - knots[2],
                ),
                CurveEnd::End => (
                    [
                        control_points[n],
                        control_points[n - 1],
                        control_points[n - 2],
                    ],
                    knots[n + p] - knots[n],
                    knots[n + p - 1] - knots[n - 1],
                    knots[n + p - 1] - knots[n],
                ),
            };
            let (q0, q1) = (l.pv(q[0]), l.pv(q[1]));
            let (center, _) = l.round_values(*arc);
            let t = [q1[0] - q0[0], q1[1] - q0[1]];
            let sigma = sign(t[0] * (center[1] - q0[1]) - t[1] * (center[0] - q0[0]));
            for p in q {
                l.point(p);
            }
            l.round(*arc);
            let rational = !weights.is_empty();
            if rational {
                let w = match end {
                    CurveEnd::Start => [weights[0], weights[1], weights[2]],
                    CurveEnd::End => [weights[n], weights[n - 1], weights[n - 2]],
                };
                l.vars.extend(w);
            }
            Op::CurvatureG2 {
                e1,
                e2,
                f,
                degree: p as f64,
                sigma,
                rational,
            }
        }
        C::EqualLength(a, b) => {
            l.line(*a);
            l.line(*b);
            Op::EqualLength
        }
        C::EqualRadius(a, b)
            if matches!(
                l.kind(*a),
                EntityKind::Ellipse { .. } | EntityKind::EllipticalArc { .. }
            ) =>
        {
            for e in [*a, *b] {
                match l.kind(e) {
                    EntityKind::Ellipse {
                        semi_major,
                        semi_minor,
                        ..
                    }
                    | EntityKind::EllipticalArc {
                        semi_major,
                        semi_minor,
                        ..
                    } => {
                        let (sa, sb) = (*semi_major, *semi_minor);
                        l.vars.extend([sa, sb]);
                    }
                    _ => unreachable!(),
                }
            }
            Op::EqualEllipse
        }
        C::EqualRadius(a, b) => {
            let ra = l.round(*a);
            let rb = l.round(*b);
            Op::EqualRadius { a: ra, b: rb }
        }
        C::Concentric(a, b) => {
            let center = |l: &Lower<'_>, e: EntityId| match l.kind(e) {
                EntityKind::Circle { center, .. } | EntityKind::Arc { center, .. } => *center,
                _ => unreachable!(),
            };
            let (ca, cb) = (center(l, *a), center(l, *b));
            l.point(ca);
            l.point(cb);
            Op::Coincident
        }
        C::Coradial(a, b) => {
            let ra = l.round(*a);
            let rb = l.round(*b);
            Op::Coradial { a: ra, b: rb }
        }
        C::Midpoint { point, line } => {
            l.point(*point);
            l.line(*line);
            Op::Midpoint
        }
        C::Symmetric { a, b, axis } => {
            l.point(*a);
            l.point(*b);
            l.line(*axis);
            Op::Symmetric
        }
        C::Collinear(a, b) => {
            l.line(*a);
            l.line(*b);
            Op::Collinear
        }
        C::Fix(e) => match l.kind(*e).clone() {
            EntityKind::Arc { center, start, end } => {
                l.point(center);
                l.point(start);
                l.point(end);
                let t = &rec.fix_targets;
                let r = [t[4] - t[0], t[5] - t[1]];
                let n = r[0].hypot(r[1]).max(1e-300);
                Op::FixArc {
                    targets: [t[0], t[1], t[2], t[3], t[4], t[5]],
                    tangent: [-r[1] / n, r[0] / n],
                }
            }
            EntityKind::EllipticalArc { start, end, .. } => {
                l.ellipse(*e);
                l.point(start);
                l.point(end);
                let t: [f64; 9] = rec.fix_targets[..9].try_into().expect("9 targets");
                let unit = |p: [f64; 2]| {
                    let d = ell_tangent_at(&t[0..5], (p[0], p[1]));
                    let n = d.0.hypot(d.1).max(1e-300);
                    [d.0 / n, d.1 / n]
                };
                let (ts, te) = (unit([t[5], t[6]]), unit([t[7], t[8]]));
                Op::FixEllipticalArc {
                    targets: t,
                    tangents: [ts[0], ts[1], te[0], te[1]],
                }
            }
            EntityKind::Body { .. } => {
                l.body(*e);
                let t = &rec.fix_targets;
                let q = crate::system::normalize4([t[3], t[4], t[5], t[6]]);
                Op::FixBody {
                    targets: [t[0], t[1], t[2], q[0], q[1], q[2], q[3]],
                }
            }
            _ => {
                // Spline weights are held only while they are free: a fixed weight is already
                // constant and a row on it would read as redundant.
                let all = l.sys.entity_vars(*e);
                let mut targets = Vec::with_capacity(all.len());
                for (v, t) in all.into_iter().zip(&rec.fix_targets) {
                    let var = l.sys.var(v);
                    if var.role == crate::system::VarRole::Weight
                        && var.flag != crate::system::VarFlag::Free
                    {
                        continue;
                    }
                    l.vars.push(v);
                    targets.push(*t);
                }
                Op::FixValues { targets }
            }
        },
        C::PointOnCurve { point, curve } => {
            l.point(*point);
            match l.kind(*curve).clone() {
                EntityKind::Line { .. } => {
                    l.line(*curve);
                    Op::PointOnLine
                }
                EntityKind::Circle { .. } | EntityKind::Arc { .. } => Op::PointOnRound {
                    round: l.round(*curve),
                },
                EntityKind::Ellipse { .. } | EntityKind::EllipticalArc { .. } => {
                    l.ellipse(*curve);
                    Op::PointOnEllipse
                }
                EntityKind::BSpline { .. } => {
                    l.vars.push(rec.aux[0]);
                    let (degree, knots, rational, periodic) = l.spline(*curve);
                    Op::PointOnSpline {
                        degree,
                        knots,
                        rational,
                        periodic,
                    }
                }
                _ => unreachable!(),
            }
        }
        C::Distance { a, b, value } => {
            let a_point = matches!(l.kind(*a), EntityKind::Point2 { .. });
            let b_point = matches!(l.kind(*b), EntityKind::Point2 { .. });
            if a_point && b_point {
                l.point(*a);
                l.point(*b);
                Op::DistancePP { d: *value }
            } else {
                let (p, line) = if a_point { (*a, *b) } else { (*b, *a) };
                let pv = l.pv(p);
                l.point(p);
                let [la, lb] = l.line(line);
                let side = sign(signed_dist_f(l.pv(la), l.pv(lb), pv));
                Op::DistancePL { d: *value, side }
            }
        }
        C::Angle { a, b, value } => {
            l.line(*a);
            l.line(*b);
            Op::Angle { theta: *value }
        }
        C::Radius { curve, value } => Op::RadiusValue {
            round: l.round(*curve),
            scale: 1.0,
            v: *value,
        },
        C::Diameter { curve, value } => Op::RadiusValue {
            round: l.round(*curve),
            scale: 2.0,
            v: *value,
        },
        C::Length { line, value } => {
            l.line(*line);
            Op::Length { v: *value }
        }
        C::VarValue { var, value } => {
            l.vars.push(*var);
            Op::VarValue { v: *value }
        }
        C::EllipseDimension {
            curve,
            measure,
            value,
        } => {
            match l.kind(*curve) {
                EntityKind::Ellipse {
                    semi_major,
                    semi_minor,
                    ..
                }
                | EntityKind::EllipticalArc {
                    semi_major,
                    semi_minor,
                    ..
                } => {
                    let (a, b) = (*semi_major, *semi_minor);
                    l.vars.extend([a, b]);
                }
                _ => unreachable!(),
            }
            Op::EllipseValue {
                measure: *measure,
                v: *value,
            }
        }
        C::Limit {
            dimension,
            min,
            max,
        } => {
            let op = lower_user(l, dimension, rec, range);
            *range = Some((*min, *max));
            op
        }
        C::Mate(m) => {
            let (a, b) = crate::system::mate_connectors(m);
            let la = l.connector(a);
            let lb = l.connector(b);
            let mut target = |t: &Target| match *t {
                Target::Exact(v) => v,
                Target::Range { min, max } => {
                    *range = Some((min, max));
                    0.0
                }
            };
            let kind = match m {
                Mate::Fasten { .. } => MateOp::Fasten,
                Mate::Revolute { .. } => MateOp::Revolute,
                Mate::Slider { .. } => MateOp::Slider,
                Mate::Cylindrical { .. } => MateOp::Cylindrical,
                Mate::Planar { .. } => MateOp::Planar,
                Mate::Ball { .. } => MateOp::Ball,
                Mate::Parallel { .. } => MateOp::Parallel,
                Mate::Tangent {
                    kind: TangentKind::PlaneSphere { radius },
                    ..
                } => MateOp::TangentSphere { r: *radius },
                Mate::Tangent {
                    kind: TangentKind::PlaneCylinder { radius },
                    ..
                } => MateOp::TangentCylinder { r: *radius },
                Mate::Distance {
                    mode: DistanceMode::PointPoint,
                    target: t,
                    ..
                } => MateOp::DistancePoint { d: target(t) },
                Mate::Distance {
                    mode: DistanceMode::PlanePoint,
                    target: t,
                    ..
                } => MateOp::DistancePlane { d: target(t) },
                Mate::Angle {
                    mode: AngleMode::AboutZ,
                    target: t,
                    ..
                } => MateOp::AngleAboutZ { theta: target(t) },
                Mate::Angle {
                    mode: AngleMode::BetweenZ,
                    target: t,
                    ..
                } => MateOp::AngleBetweenZ { theta: target(t) },
                Mate::PinSlot { .. } => MateOp::PinSlot,
                Mate::Screw { lead, .. } => MateOp::Screw { lead: *lead },
                Mate::Cam { radii, .. } => MateOp::Cam {
                    radii: radii.clone(),
                },
                Mate::UniversalJoint { joint_point, .. } => MateOp::UniversalJoint {
                    joint_point: *joint_point,
                },
                Mate::Symmetric { plane, .. } => MateOp::Symmetric {
                    lc: l.connector(*plane),
                },
                Mate::Path { points, .. } => MateOp::Path {
                    points: points.clone(),
                },
                Mate::LinearCoupler { ratio, .. } => {
                    let x: Vec<f64> = l.vars.iter().map(|v| l.v(*v)).collect();
                    let (oa0, _, [_, _, za0]) = connector_world(&x[0..7], &la);
                    let (ob0, _, [_, _, zb0]) = connector_world(&x[7..14], &lb);
                    MateOp::LinearCoupler {
                        ratio: *ratio,
                        oa0,
                        za0,
                        ob0,
                        zb0,
                    }
                }
                Mate::Gear { ratio, .. } | Mate::RackPinion { travel: ratio, .. } => {
                    // The couplings measure turns from where this solve starts.
                    let x: Vec<f64> = l.vars.iter().map(|v| l.v(*v)).collect();
                    let (_, _, [xa0, _, za0]) = connector_world(&x[0..7], &la);
                    let (ob0, _, [xb0, _, zb0]) = connector_world(&x[7..14], &lb);
                    match m {
                        Mate::Gear { .. } => MateOp::Gear {
                            ratio: *ratio,
                            xa0,
                            za0,
                            xb0,
                            zb0,
                        },
                        _ => MateOp::RackPinion {
                            travel: *ratio,
                            xa0,
                            za0,
                            ob0,
                            xb0,
                        },
                    }
                }
            };
            Op::Mate { kind, la, lb }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks an op's AD Jacobian against central differences at `x`.
    pub(crate) fn check(op: &Op, x: &[f64]) {
        let rows = op.rows();
        let n = x.len();
        let mut vals = vec![0.0; rows];
        let mut jac = vec![0.0; rows * n];
        op.jacobian(x, &mut vals, &mut jac);
        let mut plain = vec![0.0; rows];
        op.values(x, &mut plain);
        for r in 0..rows {
            assert!((plain[r] - vals[r]).abs() < 1e-12, "{op:?} value row {r}");
        }
        let h = 1e-6;
        for j in 0..n {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += h;
            xm[j] -= h;
            let mut fp = vec![0.0; rows];
            let mut fm = vec![0.0; rows];
            op.values(&xp, &mut fp);
            op.values(&xm, &mut fm);
            for r in 0..rows {
                let fd = (fp[r] - fm[r]) / (2.0 * h);
                let ad = jac[r * n + j];
                assert!(
                    (fd - ad).abs() < 1e-6 * (1.0 + fd.abs()),
                    "{op:?}: row {r} col {j}: ad {ad} fd {fd}"
                );
            }
        }
    }

    fn pseudo(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
            })
            .collect()
    }

    #[test]
    fn every_op_jacobian_matches_finite_differences() {
        let la = Frame::axis_angle([0.3, -0.2, 0.5], [0.2, 1.0, 0.4], 0.7);
        let lb = Frame::axis_angle([-0.1, 0.4, 0.2], [1.0, -0.3, 0.2], -0.4);
        let mates = [
            MateOp::Fasten,
            MateOp::Revolute,
            MateOp::Slider,
            MateOp::Cylindrical,
            MateOp::Planar,
            MateOp::Ball,
            MateOp::Parallel,
            MateOp::TangentSphere { r: 0.5 },
            MateOp::TangentCylinder { r: 0.5 },
            MateOp::DistancePoint { d: 1.0 },
            MateOp::DistancePlane { d: 1.0 },
            MateOp::AngleAboutZ { theta: 0.3 },
            MateOp::AngleBetweenZ { theta: 0.3 },
            MateOp::PinSlot,
            MateOp::Screw { lead: 0.25 },
            MateOp::Gear {
                ratio: 2.0,
                xa0: [1.0, 0.0, 0.0],
                za0: [0.0, 0.0, 1.0],
                xb0: [0.0, 1.0, 0.0],
                zb0: [0.0, 0.0, 1.0],
            },
            MateOp::RackPinion {
                travel: 0.3,
                xa0: [1.0, 0.0, 0.0],
                za0: [0.0, 0.0, 1.0],
                ob0: [0.1, 0.2, 0.3],
                xb0: [0.0, 1.0, 0.0],
            },
            MateOp::Cam {
                radii: std::sync::Arc::from([0.6, 0.7, 1.1, 1.4, 1.2, 0.8, 0.6, 0.6]),
            },
            MateOp::LinearCoupler {
                ratio: -1.5,
                oa0: [0.1, 0.0, 0.2],
                za0: [0.0, 0.6, 0.8],
                ob0: [0.3, -0.2, 0.0],
                zb0: [1.0, 0.0, 0.0],
            },
            MateOp::UniversalJoint { joint_point: false },
            MateOp::UniversalJoint { joint_point: true },
            // Far from the path's corners, so finite differences stay in one segment.
            MateOp::Path {
                points: std::sync::Arc::from([
                    [-50.0, -40.0, 3.0],
                    [60.0, 10.0, -2.0],
                    [70.0, 90.0, 5.0],
                ]),
            },
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let mut cases: Vec<(Op, usize)> = vec![
            (Op::Coincident, 4),
            (Op::SameX, 4),
            (Op::SameY, 4),
            (Op::Parallel, 8),
            (Op::Perpendicular, 8),
            (
                Op::TangentLineRound {
                    round: Round::Circle,
                    side: 1.0,
                },
                7,
            ),
            (
                Op::TangentLineRound {
                    round: Round::Arc,
                    side: -1.0,
                },
                10,
            ),
            (
                Op::TangentRounds {
                    a: Round::Circle,
                    b: Round::Arc,
                    internal: false,
                    sigma: 1.0,
                },
                9,
            ),
            (
                Op::TangentRounds {
                    a: Round::Arc,
                    b: Round::Circle,
                    internal: true,
                    sigma: -1.0,
                },
                9,
            ),
            (
                Op::TangentEnds {
                    a: EndDir::Segment,
                    b: EndDir::ArcRadius,
                },
                8,
            ),
            (
                Op::CurvatureG2 {
                    e1: 0.5,
                    e2: 0.5,
                    f: 0.5,
                    degree: 3.0,
                    sigma: 1.0,
                    rational: false,
                },
                12,
            ),
            (
                Op::CurvatureG2 {
                    e1: 0.5,
                    e2: 0.5,
                    f: 0.5,
                    degree: 3.0,
                    sigma: 1.0,
                    rational: true,
                },
                15,
            ),
            (Op::TangentLineEllipse { side: -1.0 }, 9),
            (
                Op::TangentRoundEllipse {
                    round: Round::Circle,
                },
                9,
            ),
            (Op::TangentRoundEllipse { round: Round::Arc }, 12),
            (Op::TangentEllipses, 12),
            (
                Op::TangentEnds {
                    a: EndDir::Ellipse,
                    b: EndDir::Segment,
                },
                11,
            ),
            (Op::EqualEllipse, 4),
            (
                Op::EllipseValue {
                    measure: crate::system::EllipseMeasure::FocalDistance,
                    v: 1.0,
                },
                2,
            ),
            (
                Op::FixEllipticalArc {
                    targets: [0.0, 0.0, 2.0, 1.0, 0.3, 1.0, 0.5, -1.0, 0.2],
                    tangents: [0.6, 0.8, -0.8, 0.6],
                },
                9,
            ),
            (Op::EllipticalArcEnds, 9),
            (Op::EqualLength, 8),
            (
                Op::EqualRadius {
                    a: Round::Circle,
                    b: Round::Arc,
                },
                9,
            ),
            (
                Op::Coradial {
                    a: Round::Arc,
                    b: Round::Circle,
                },
                9,
            ),
            (Op::Midpoint, 6),
            (Op::Symmetric, 8),
            (Op::Collinear, 8),
            (
                Op::FixValues {
                    targets: vec![0.1, 0.2, 0.3],
                },
                3,
            ),
            (
                Op::FixArc {
                    targets: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    tangent: [-1.0, 0.0],
                },
                6,
            ),
            (
                Op::FixBody {
                    targets: [0.0, 0.0, 0.0, 0.9, 0.1, 0.3, 0.2],
                },
                7,
            ),
            (Op::PointOnLine, 6),
            (
                Op::PointOnRound {
                    round: Round::Circle,
                },
                5,
            ),
            (Op::PointOnRound { round: Round::Arc }, 8),
            (Op::PointOnEllipse, 7),
            (
                Op::PointOnSpline {
                    degree: 3,
                    knots: knots.clone(),
                    rational: false,
                    periodic: false,
                },
                3 + 10,
            ),
            (
                Op::PointOnSpline {
                    degree: 3,
                    knots,
                    rational: true,
                    periodic: false,
                },
                3 + 15,
            ),
            (
                // 4 periodic control points, degree 2: 6 wrapped points, 9 knots.
                Op::PointOnSpline {
                    degree: 2,
                    knots: (0..9).map(|i| (i as f64 - 2.0) / 4.0).collect(),
                    rational: true,
                    periodic: true,
                },
                3 + 18,
            ),
            (Op::DistancePP { d: 1.0 }, 4),
            (Op::DistancePL { d: 1.0, side: -1.0 }, 6),
            (Op::Angle { theta: 0.4 }, 8),
            (
                Op::RadiusValue {
                    round: Round::Arc,
                    scale: 2.0,
                    v: 1.0,
                },
                6,
            ),
            (Op::Length { v: 1.0 }, 4),
            (Op::VarValue { v: 1.0 }, 1),
            (Op::ArcRadius, 6),
            (Op::QuatNorm, 4),
        ];
        for m in mates {
            cases.push((Op::Mate { kind: m, la, lb }, 14));
        }
        let lc = Frame::axis_angle([0.2, 0.1, -0.3], [0.5, 0.2, 1.0], 0.9);
        cases.push((
            Op::Mate {
                kind: MateOp::Symmetric { lc },
                la,
                lb,
            },
            21,
        ));
        for (k, (op, n)) in cases.iter().enumerate() {
            for seed in 0..4u64 {
                let mut x = pseudo(*n, 17 + seed * 31 + k as u64 * 7);
                // Positive semi-axes where the layout has them.
                let semi: &[usize] = match op {
                    Op::PointOnEllipse => &[4, 5],
                    Op::TangentLineEllipse { .. } => &[6, 7],
                    Op::TangentRoundEllipse {
                        round: Round::Circle,
                    } => &[5, 6],
                    Op::TangentRoundEllipse { round: Round::Arc } => &[8, 9],
                    Op::TangentEllipses => &[2, 3, 7, 8],
                    Op::TangentEnds {
                        a: EndDir::Ellipse, ..
                    }
                    | Op::EllipticalArcEnds
                    | Op::FixEllipticalArc { .. } => &[2, 3],
                    Op::EllipseValue { .. } | Op::EqualEllipse => &[0, 1],
                    _ => &[],
                };
                for &i in semi {
                    x[i] = x[i].abs() + 0.5;
                }
                if let Op::PointOnSpline { rational, .. } = op {
                    x[2] = 0.13 + 0.2 * seed as f64; // off the knots (FD needs smoothness)
                    if *rational {
                        let m = spline_count(*n, true);
                        for w in &mut x[3 + 2 * m..] {
                            *w = w.abs() + 0.3;
                        }
                    }
                }
                if let Op::CurvatureG2 { rational: true, .. } = op {
                    for w in &mut x[12..15] {
                        *w = w.abs() + 0.3;
                    }
                }
                if let Op::EllipseValue { .. } = op {
                    x[0] = x[1] + 0.7; // a > b away from the circle case
                }
                check(op, &x);
            }
        }
    }
}
