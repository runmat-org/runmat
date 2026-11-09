//! Occupancy and crossings: whether a point is inside a sampled tool, and which tool surface
//! a facet's boundary crosses.

use super::*;

impl Sampled {
    pub(in crate::regen::fake) fn inside(&self, w: [f64; 3]) -> bool {
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
    pub(super) fn on_axis(&self, c: &Curve2) -> bool {
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

    pub(super) fn revolve_in(&self, q: [f64; 2]) -> bool {
        match &self.shape {
            Shape::Revolve { polys, .. } => inside_polys(polys, q),
            _ => false,
        }
    }

    /// Profile-plane point, angle from the profile half-plane (0..2π), radius, sweep angle.
    pub(super) fn revolve_coords(&self, p: [f64; 3]) -> ([f64; 2], f64, f64, f64) {
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
    pub(super) fn surface_distances(&self, p: [f64; 3]) -> Vec<f64> {
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
    pub(in crate::regen::fake) fn crossing(&self, a: [f64; 3], b: [f64; 3]) -> Option<usize> {
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
}
