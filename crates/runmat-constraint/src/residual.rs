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

/// How the tangent direction at an open-curve end is read from a 4-variable layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EndDir {
    /// `[p, q]`: direction `q - p` (a line, or a spline's end control polygon leg).
    Segment,
    /// `[c, p]`: perpendicular to the radius `p - c` (an arc end).
    ArcRadius,
}

impl EndDir {
    fn dir<S: Real>(self, x: &[S]) -> (S, S) {
        let (dx, dy) = (x[2] - x[0], x[3] - x[1]);
        match self {
            EndDir::Segment => (dx, dy),
            EndDir::ArcRadius => (-dy, dx),
        }
    }
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
    /// `[endA(4), endB(4)]` → `sin` of the angle between the end tangents.
    TangentEnds { a: EndDir, b: EndDir },
    /// `[Q0, Q1, Q2, arc(6)]` → G1 (`cos` between tangent and radius) and `κ r - σ`.
    CurvatureG2 {
        e1: f64,
        e2: f64,
        f: f64,
        degree: f64,
        sigma: f64,
    },
    /// `[l1, l2]` → `|l1| - |l2|`.
    EqualLength,
    /// `[roundA, roundB]` → `ra - rb`.
    EqualRadius { a: Round, b: Round },
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
    /// `[p, line]` → signed distance from the line.
    PointOnLine,
    /// `[p, round]` → `|p - c| - r`.
    PointOnRound { round: Round },
    /// `[p, ellipse(5)]` → scaled implicit ellipse equation.
    PointOnEllipse,
    /// `[p, t, control points…]` → `p - C(t)` (2 rows).
    PointOnSpline { degree: usize, knots: Vec<f64> },
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
    /// `[bodyA(7), bodyB(7)]` → mate rows.
    Mate { kind: MateOp, la: Frame, lb: Frame },
}

/// Mate residual kinds (connector frames are applied in [`Op::Mate`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MateOp {
    Fasten,
    Revolute,
    Slider,
    Cylindrical,
    Planar,
    Ball,
    Parallel,
    TangentSphere { r: f64 },
    TangentCylinder { r: f64 },
    DistancePoint { d: f64 },
    DistancePlane { d: f64 },
    AngleAboutZ { theta: f64 },
    AngleBetweenZ { theta: f64 },
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
            Op::CurvatureG2 { .. } => 2,
            Op::Coradial { .. } => 3,
            Op::FixValues { targets } => targets.len(),
            Op::FixArc { .. } => 5,
            Op::FixBody { .. } => 6,
            Op::Mate { kind, .. } => match kind {
                MateOp::Fasten => 6,
                MateOp::Revolute | MateOp::Slider => 5,
                MateOp::Cylindrical => 4,
                MateOp::Planar | MateOp::Ball => 3,
                MateOp::Parallel | MateOp::TangentCylinder { .. } => 2,
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
            Op::RadiusValue { v, .. } | Op::Length { v } | Op::VarValue { v } => *v = t,
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
            Op::TangentEnds { a, b } => {
                let (ta, tb) = (a.dir(&x[0..4]), b.dir(&x[4..8]));
                out[0] = cross(ta, tb) / (norm(ta) * norm(tb));
            }
            Op::CurvatureG2 {
                e1,
                e2,
                f,
                degree,
                sigma,
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
                let kappa = cross(c1, c2) / (n1 * n1 * n1);
                out[1] = kappa * Round::Arc.radius(arc) - *sigma;
            }
            Op::EqualLength => out[0] = norm(seg(&x[0..4])) - norm(seg(&x[4..8])),
            Op::EqualRadius { a, b } => out[0] = a.radius(&x[..a.len()]) - b.radius(&x[a.len()..]),
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
            Op::PointOnLine => out[0] = signed_dist(&x[2..6], (x[0], x[1])),
            Op::PointOnRound { round } => {
                let c = &x[2..];
                out[0] = (x[0] - c[0]).hypot(x[1] - c[1]) - round.radius(c);
            }
            Op::PointOnEllipse => {
                let e = &x[2..7];
                let (dx, dy) = (x[0] - e[0], x[1] - e[1]);
                let (s, c) = (e[4].sin(), e[4].cos());
                let u = c * dx + s * dy;
                let v = c * dy - s * dx;
                let (a, b) = (e[2], e[3]);
                out[0] = ((u / a).sq() + (v / b).sq() - 1.0) * ((a + b) * 0.25);
            }
            Op::PointOnSpline { degree, knots } => {
                let (cx, cy) = spline_point(*degree, knots, &x[3..], x[2]);
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
            Op::Mate { kind, la, lb } => eval_mate(*kind, la, lb, x, out),
        }
    }

    /// Residual values at `x`.
    pub(crate) fn values(&self, x: &[f64], out: &mut [f64]) {
        self.eval(x, out);
    }

    /// Residual values and the dense `rows × x.len()` Jacobian block (row-major).
    pub(crate) fn jacobian(&self, x: &[f64], vals: &mut [f64], jac: &mut [f64]) {
        let n = x.len();
        match n {
            0..=4 => jac_n::<4>(self, x, vals, jac),
            5..=8 => jac_n::<8>(self, x, vals, jac),
            9..=16 => jac_n::<16>(self, x, vals, jac),
            17..=32 => jac_n::<32>(self, x, vals, jac),
            _ => jac_n::<64>(self, x, vals, jac),
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

/// A point on a clamped B-spline by de Boor's algorithm. The span comes from the primal
/// parameter, clamped to the knot domain; outside the domain the end polynomial extends.
pub(crate) fn spline_point<S: Real>(degree: usize, knots: &[f64], cps: &[S], t: S) -> (S, S) {
    let count = cps.len() / 2;
    let last = count - 1;
    let tv = t.val().clamp(knots[degree], knots[last + 1]);
    let mut s = degree;
    while s < last && tv >= knots[s + 1] {
        s += 1;
    }
    let mut dx: Vec<S> = (0..=degree).map(|j| cps[2 * (j + s - degree)]).collect();
    let mut dy: Vec<S> = (0..=degree)
        .map(|j| cps[2 * (j + s - degree) + 1])
        .collect();
    for r in 1..=degree {
        for j in (r..=degree).rev() {
            let i = j + s - degree;
            let den = knots[i + degree + 1 - r] - knots[i];
            let alpha = if den > 0.0 {
                (t - knots[i]) / den
            } else {
                S::cst(0.0)
            };
            let one = S::cst(1.0) - alpha;
            dx[j] = dx[j - 1] * one + dx[j] * alpha;
            dy[j] = dy[j - 1] * one + dy[j] * alpha;
        }
    }
    (dx[degree], dy[degree])
}

/// The parameter of the sampled spline point closest to `p` (deterministic, used to seed the
/// curve parameter of a point-on-spline constraint).
pub(crate) fn closest_spline_parameter(sys: &System, curve: EntityId, p: [f64; 2]) -> f64 {
    let EntityKind::BSpline {
        degree,
        control_points,
        knots,
    } = &sys.entity(curve).kind
    else {
        return 0.0;
    };
    let cps: Vec<f64> = control_points.iter().flat_map(|c| sys.point(*c)).collect();
    let (t0, t1) = (knots[*degree], knots[control_points.len()]);
    let samples = 256;
    let mut best = (f64::INFINITY, t0);
    for i in 0..=samples {
        let t = t0 + (t1 - t0) * i as f64 / samples as f64;
        let (x, y) = spline_point(*degree, knots, &cps, t);
        let d = (x - p[0]).hypot(y - p[1]);
        if d < best.0 {
            best = (d, t);
        }
    }
    best.1
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

fn eval_mate<S: Real>(kind: MateOp, la: &Frame, lb: &Frame, x: &[S], out: &mut [S]) {
    let (oa, qa, [xa, ya, za]) = connector_world(&x[0..7], la);
    let (ob, qb, [xb, yb, zb]) = connector_world(&x[7..14], lb);
    let d = sub3(ob, oa);
    match kind {
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
    }
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
    /// Pushes the 4-variable end layout of an open curve and returns how to read it.
    fn end(&mut self, e: EntityId, end: CurveEnd) -> EndDir {
        match self.kind(e).clone() {
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
                };
                (op, true)
            }
            ConstraintKindRecord::User(c) => (lower_user(&mut l, c, rec, &mut range), false),
        };
        out.push(Compiled {
            id,
            vars: l.vars,
            op,
            range,
            reference: rec.reference,
            internal,
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
            if a_line || b_line {
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
            Op::CurvatureG2 {
                e1,
                e2,
                f,
                degree: p as f64,
                sigma,
            }
        }
        C::EqualLength(a, b) => {
            l.line(*a);
            l.line(*b);
            Op::EqualLength
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
            EntityKind::Body { .. } => {
                l.body(*e);
                let t = &rec.fix_targets;
                let q = crate::system::normalize4([t[3], t[4], t[5], t[6]]);
                Op::FixBody {
                    targets: [t[0], t[1], t[2], q[0], q[1], q[2], q[3]],
                }
            }
            _ => {
                l.vars = l.sys.entity_vars(*e);
                Op::FixValues {
                    targets: rec.fix_targets.clone(),
                }
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
                EntityKind::Ellipse {
                    center,
                    semi_major,
                    semi_minor,
                    rotation,
                } => {
                    l.point(center);
                    l.vars.extend([semi_major, semi_minor, rotation]);
                    Op::PointOnEllipse
                }
                EntityKind::BSpline {
                    degree,
                    control_points,
                    knots,
                } => {
                    l.vars.push(rec.aux[0]);
                    for p in &control_points {
                        l.point(*p);
                    }
                    Op::PointOnSpline { degree, knots }
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
                },
                12,
            ),
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
            (Op::PointOnSpline { degree: 3, knots }, 3 + 10),
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
        for (k, (op, n)) in cases.iter().enumerate() {
            for seed in 0..4u64 {
                let mut x = pseudo(*n, 17 + seed * 31 + k as u64 * 7);
                if matches!(op, Op::PointOnEllipse) {
                    x[4] = x[4].abs() + 0.5;
                    x[5] = x[5].abs() + 0.3;
                }
                if matches!(op, Op::PointOnSpline { .. }) {
                    x[2] = 0.1 + 0.2 * seed as f64; // stay inside one knot span per case
                }
                check(op, &x);
            }
        }
    }
}
