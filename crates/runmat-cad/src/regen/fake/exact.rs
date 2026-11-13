//! Exact surfaces behind the grid's faceted faces, for display.
//!
//! A faceted face stores the occupancy staircase the grid decided; the tool that made it knows
//! the true surface: a profile curve extruded along a direction, a profile curve revolved about
//! an axis, or a plane off the grid's axes. [`Exact`] keeps that surface with a chart
//! `(u, v) → point`, its inverse (closest point), normals and curvature, so the display mesher
//! ([`super::mesh`]) can draw the face on its surface instead of on the staircase.
//!
//! Charts are arc-length scaled: `u` and `v` are lengths along the surface (a revolve's `u` is
//! the angle times a reference radius), so distances in the chart track distances on the
//! surface and a planar triangulation of the chart is well shaped on the surface.

use super::sampled::Curve2;
use crate::regen::kernel::Transform;
use std::f64::consts::{PI, TAU};

pub(super) type P3 = [f64; 3];

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: P3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: P3) -> P3 {
    let n = norm(a);
    if n < 1e-300 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / n)
    }
}

fn d2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Arc-length parametrisation of a profile curve.
impl Curve2 {
    pub(super) fn length(&self) -> f64 {
        match self {
            Curve2::Line { a, b } => d2(*a, *b),
            Curve2::Arc { r, sweep, .. } => r * sweep.abs(),
            Curve2::Circle { r, .. } => TAU * r,
            Curve2::Poly { pts } => pts.windows(2).map(|w| d2(w[0], w[1])).sum(),
        }
    }

    /// The length of one period, for a closed curve.
    pub(super) fn period(&self) -> Option<f64> {
        match self {
            Curve2::Circle { r, .. } => Some(TAU * r),
            Curve2::Poly { pts } if pts.len() > 3 && d2(pts[0], pts[pts.len() - 1]) < 1e-12 => {
                Some(self.length())
            }
            _ => None,
        }
    }

    /// The point at arc length `u` from the start (extended past the ends along the curve's
    /// own geometry: a line or polyline end straight on, an arc around its circle).
    pub(super) fn at(&self, u: f64) -> [f64; 2] {
        match self {
            Curve2::Line { a, b } => {
                let l = d2(*a, *b).max(1e-300);
                let t = u / l;
                [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            }
            Curve2::Arc { c, r, a0, sweep } => {
                let ang = a0 + sweep.signum() * u / r;
                [c[0] + r * ang.cos(), c[1] + r * ang.sin()]
            }
            Curve2::Circle { c, r } => {
                let ang = u / r;
                [c[0] + r * ang.cos(), c[1] + r * ang.sin()]
            }
            Curve2::Poly { pts } => {
                let n = pts.len();
                // A closed polyline wraps; an open one extends straight past its ends.
                let u = match self.period() {
                    Some(p) if p > 0.0 => u.rem_euclid(p),
                    _ => u,
                };
                let mut acc = 0.0;
                for k in 0..n - 1 {
                    let l = d2(pts[k], pts[k + 1]);
                    if u <= acc + l || k == n - 2 {
                        let t = if l > 1e-300 { (u - acc) / l } else { 0.0 };
                        let (a, b) = (pts[k], pts[k + 1]);
                        return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                    }
                    acc += l;
                }
                pts[n - 1]
            }
        }
    }

    /// Arc length of the closest point to `q` (unclamped on lines and circles).
    pub(super) fn closest(&self, q: [f64; 2]) -> f64 {
        match self {
            Curve2::Line { a, b } => {
                let d = [b[0] - a[0], b[1] - a[1]];
                let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
                ((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / l
            }
            Curve2::Circle { c, r } => {
                let mut ang = (q[1] - c[1]).atan2(q[0] - c[0]);
                if ang < 0.0 {
                    ang += TAU;
                }
                ang * r
            }
            Curve2::Arc { c, r, a0, sweep } => {
                let ang = (q[1] - c[1]).atan2(q[0] - c[0]);
                // Relative angle along the sweep, chosen nearest the arc's middle.
                let mut rel = (ang - a0) * sweep.signum();
                let mid = sweep.abs() / 2.0;
                while rel - mid > PI {
                    rel -= TAU;
                }
                while rel - mid < -PI {
                    rel += TAU;
                }
                rel * r
            }
            Curve2::Poly { pts } => {
                let mut best = (f64::INFINITY, 0.0);
                let mut acc = 0.0;
                let n = pts.len();
                let closed = self.period().is_some();
                for k in 0..n - 1 {
                    let (a, b) = (pts[k], pts[k + 1]);
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let l2 = d[0] * d[0] + d[1] * d[1];
                    let l = l2.sqrt();
                    let mut t = if l2 > 1e-300 {
                        ((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / l2
                    } else {
                        0.0
                    };
                    let lo = if k == 0 && !closed {
                        f64::NEG_INFINITY
                    } else {
                        0.0
                    };
                    let hi = if k == n - 2 && !closed {
                        f64::INFINITY
                    } else {
                        1.0
                    };
                    t = t.clamp(lo, hi);
                    let p = [a[0] + t * d[0], a[1] + t * d[1]];
                    let dist = d2(p, q);
                    if dist < best.0 {
                        best = (dist, acc + t * l);
                    }
                    acc += l;
                }
                best.1
            }
        }
    }

    /// The smallest radius of curvature (infinite for a line).
    pub(super) fn min_radius(&self) -> f64 {
        match self {
            Curve2::Line { .. } => f64::INFINITY,
            Curve2::Arc { r, .. } | Curve2::Circle { r, .. } => *r,
            Curve2::Poly { pts } => {
                let mut best = f64::INFINITY;
                for w in pts.windows(3) {
                    let (a, b) = (
                        [w[1][0] - w[0][0], w[1][1] - w[0][1]],
                        [w[2][0] - w[1][0], w[2][1] - w[1][1]],
                    );
                    let (la, lb) = (d2(w[0], w[1]), d2(w[1], w[2]));
                    let turn = (a[0] * b[1] - a[1] * b[0]).atan2(a[0] * b[0] + a[1] * b[1]);
                    if turn.abs() > 1e-9 {
                        best = best.min(0.5 * (la + lb) / turn.abs());
                    }
                }
                best
            }
        }
    }

    /// The largest distance of the curve from a line through `ap` along unit `ad`.
    fn max_offset(&self, ap: [f64; 2], ad: [f64; 2]) -> f64 {
        let n = (self.length() / 1e-3).clamp(8.0, 256.0) as usize;
        (0..=n)
            .map(|i| {
                let p = self.at(self.length() * i as f64 / n as f64);
                ((p[0] - ap[0]) * ad[1] - (p[1] - ap[1]) * ad[0]).abs()
            })
            .fold(0.0, f64::max)
    }
}

/// A face's true surface with an arc-length chart.
#[derive(Clone, Debug)]
pub(super) enum Exact {
    /// `local(u, v) = o + (curve(u), 0) + v·d` (unit `d`, out of the profile plane), placed
    /// in the world by `world`.
    Extruded {
        world: Transform,
        curve: Curve2,
        o: P3,
        d: P3,
    },
    /// `curve` (in the local xy plane) revolved about the local line `ap + t·ad`; `u` is the
    /// angle from the profile half-plane (towards `sense·(ad × e1)`) times `rref`.
    Revolved {
        world: Transform,
        curve: Curve2,
        ap: [f64; 2],
        ad: [f64; 2],
        e1: [f64; 2],
        sense: f64,
        rref: f64,
    },
    /// A plane through `o` spanned by unit `e1`, `e2`.
    Plane { o: P3, e1: P3, e2: P3 },
}

impl Exact {
    pub(super) fn extruded(world: Transform, curve: Curve2, o: P3, d: P3) -> Option<Exact> {
        let d = unit(d);
        if d[2].abs() < 1e-9 || curve.length() < 1e-300 {
            return None;
        }
        Some(Exact::Extruded { world, curve, o, d })
    }

    pub(super) fn revolved(
        world: Transform,
        curve: Curve2,
        ap: [f64; 2],
        ad: [f64; 2],
        e1: [f64; 2],
        sense: f64,
    ) -> Option<Exact> {
        let rref = curve.max_offset(ap, ad);
        if rref < 1e-300 {
            return None;
        }
        Some(Exact::Revolved {
            world,
            curve,
            ap,
            ad,
            e1,
            sense: if sense < 0.0 { -1.0 } else { 1.0 },
            rref,
        })
    }

    pub(super) fn plane(o: P3, e1: P3, e2: P3) -> Exact {
        let e1 = unit(e1);
        let e2 = unit(sub(e2, scale(e1, dot(e2, e1))));
        Exact::Plane { o, e1, e2 }
    }

    /// The same surface moved by `t` (applied after its own placement).
    pub(super) fn transformed(&self, t: &Transform) -> Exact {
        match self {
            Exact::Extruded { world, curve, o, d } => Exact::Extruded {
                world: t.compose(world),
                curve: curve.clone(),
                o: *o,
                d: *d,
            },
            Exact::Revolved {
                world,
                curve,
                ap,
                ad,
                e1,
                sense,
                rref,
            } => Exact::Revolved {
                world: t.compose(world),
                curve: curve.clone(),
                ap: *ap,
                ad: *ad,
                e1: *e1,
                sense: *sense,
                rref: *rref,
            },
            Exact::Plane { o, e1, e2 } => Exact::Plane {
                o: t.apply(*o),
                e1: t.apply_vector(*e1),
                e2: t.apply_vector(*e2),
            },
        }
    }

    fn revolve_frame(ap: [f64; 2], ad: [f64; 2], e1: [f64; 2], sense: f64) -> (P3, P3, P3, P3) {
        let ap3 = [ap[0], ap[1], 0.0];
        let ad3 = [ad[0], ad[1], 0.0];
        let e13 = [e1[0], e1[1], 0.0];
        let e23 = scale(cross(ad3, e13), sense);
        (ap3, ad3, e13, e23)
    }

    pub(super) fn eval(&self, u: f64, v: f64) -> P3 {
        match self {
            Exact::Extruded { world, curve, o, d } => {
                let q = curve.at(u);
                world.apply([
                    q[0] + o[0] + v * d[0],
                    q[1] + o[1] + v * d[1],
                    o[2] + v * d[2],
                ])
            }
            Exact::Revolved {
                world,
                curve,
                ap,
                ad,
                e1,
                sense,
                rref,
            } => {
                let (ap3, ad3, e13, e23) = Self::revolve_frame(*ap, *ad, *e1, *sense);
                let q = curve.at(v);
                let rel = sub([q[0], q[1], 0.0], ap3);
                let h = dot(rel, ad3);
                let r = dot(rel, e13);
                let th = u / rref;
                let p = add(
                    add(ap3, scale(ad3, h)),
                    add(scale(e13, r * th.cos()), scale(e23, r * th.sin())),
                );
                world.apply(p)
            }
            Exact::Plane { o, e1, e2 } => add(*o, add(scale(*e1, u), scale(*e2, v))),
        }
    }

    /// Chart coordinates of the closest point to `p` (`u` in `[0, period)` when periodic).
    pub(super) fn invert(&self, p: P3) -> [f64; 2] {
        match self {
            Exact::Extruded { world, curve, o, d } => {
                let l = world.inverse().apply(p);
                let v = (l[2] - o[2]) / d[2];
                let q = [l[0] - o[0] - v * d[0], l[1] - o[1] - v * d[1]];
                let mut u = curve.closest(q);
                if let Some(per) = curve.period() {
                    u = u.rem_euclid(per);
                }
                [u, v]
            }
            Exact::Revolved {
                world,
                curve,
                ap,
                ad,
                e1,
                sense,
                rref,
            } => {
                let (ap3, ad3, e13, e23) = Self::revolve_frame(*ap, *ad, *e1, *sense);
                let l = world.inverse().apply(p);
                let rel = sub(l, ap3);
                let h = dot(rel, ad3);
                let rv = sub(rel, scale(ad3, h));
                let r = norm(rv);
                let th = dot(rv, e23).atan2(dot(rv, e13)).rem_euclid(TAU);
                let q = [ap[0] + h * ad[0] + r * e1[0], ap[1] + h * ad[1] + r * e1[1]];
                [th * rref, curve.closest(q)]
            }
            Exact::Plane { o, e1, e2 } => {
                let r = sub(p, *o);
                [dot(r, *e1), dot(r, *e2)]
            }
        }
    }

    /// The period of `u`, when the chart wraps around.
    pub(super) fn period_u(&self) -> Option<f64> {
        match self {
            Exact::Extruded { curve, .. } => curve.period(),
            Exact::Revolved { rref, .. } => Some(TAU * rref),
            Exact::Plane { .. } => None,
        }
    }

    /// The unit normal `∂u × ∂v` at `(u, v)` (the chart's own orientation).
    pub(super) fn normal(&self, u: f64, v: f64) -> P3 {
        match self {
            Exact::Plane { e1, e2, .. } => unit(cross(*e1, *e2)),
            _ => {
                let h = 1e-6 * self.scale_hint();
                let du = sub(self.eval(u + h, v), self.eval(u - h, v));
                let dv = sub(self.eval(u, v + h), self.eval(u, v - h));
                unit(cross(du, dv))
            }
        }
    }

    fn scale_hint(&self) -> f64 {
        match self {
            Exact::Extruded { curve, .. } => curve.length().max(1e-9),
            Exact::Revolved { rref, .. } => rref.max(1e-9),
            Exact::Plane { .. } => 1.0,
        }
    }

    /// Whether the surface bends at all.
    pub(super) fn curved(&self) -> bool {
        match self {
            Exact::Plane { .. } => false,
            Exact::Extruded { curve, .. } => !matches!(curve, Curve2::Line { .. }),
            Exact::Revolved { .. } => true,
        }
    }

    /// The smallest radius of curvature near `p` along `u` and along `v`.
    pub(super) fn radii(&self, p: P3) -> [f64; 2] {
        match self {
            Exact::Plane { .. } => [f64::INFINITY; 2],
            Exact::Extruded { curve, .. } => [curve.min_radius(), f64::INFINITY],
            Exact::Revolved {
                world,
                curve,
                ap,
                ad,
                ..
            } => {
                let l = world.inverse().apply(p);
                let ap3 = [ap[0], ap[1], 0.0];
                let ad3 = [ad[0], ad[1], 0.0];
                let rel = sub(l, ap3);
                let rv = sub(rel, scale(ad3, dot(rel, ad3)));
                [norm(rv).max(1e-12), curve.min_radius()]
            }
        }
    }

    /// The chart's metric along `u` at `p` (surface length per unit of `u`).
    pub(super) fn u_metric(&self, p: P3) -> f64 {
        match self {
            Exact::Revolved { rref, .. } => self.radii(p)[0] / rref,
            _ => 1.0,
        }
    }
}
