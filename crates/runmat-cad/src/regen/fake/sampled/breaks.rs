//! Grid breaks a sampled tool asks for: its own extents, thin walls and drafted walls.

use super::*;

impl Sampled {
    pub(in crate::regen::fake) fn breaks(&self, out: &mut [Vec<f64>; 3]) {
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
        self.thin_wall_breaks(out);
    }

    /// Breaks along the curved walls of a profile thinner than the uniform cells (a skin, a
    /// tube): grid lines through points along each curve, about one wall thickness apart, so
    /// cell centres fall inside the wall all the way along it. A revolve adds them at even
    /// angles about its axis too.
    pub(super) fn thin_wall_breaks(&self, out: &mut [Vec<f64>; 3]) {
        let revolve = matches!(self.shape, Shape::Revolve { .. });
        let curves: Vec<&Curve2> = match &self.shape {
            Shape::Prism { loops, .. } | Shape::Revolve { loops, .. } => loops
                .iter()
                .flatten()
                .map(|(c, _)| c)
                .filter(|c| !self.on_axis(c))
                .collect(),
            Shape::Extrusion(x) if x.draft.is_none() && x.thin.is_none() => {
                x.loops.iter().flatten().map(|(c, _, _)| c).collect()
            }
            _ => return,
        };
        let cell = (0..3)
            .map(|a| (self.hi[a] - self.lo[a]) / RES as f64)
            .filter(|w| *w > 1e-12)
            .fold(f64::INFINITY, f64::min);
        // The wall: the least distance from a wall curve's middle to another curve (a revolve's
        // straight walls turn into cylinders and cones, so they count too).
        let mut wall = f64::INFINITY;
        for (i, c) in curves.iter().enumerate() {
            if c.line().is_some() && !revolve {
                continue;
            }
            let mid = c.at(c.length() / 2.0);
            for (j, d) in curves.iter().enumerate() {
                if i != j {
                    // A wall is long beside its thickness; a short end cap is not one.
                    let dist = d.distance(mid);
                    if dist > 1e-9 && c.length() > 8.0 * dist {
                        wall = wall.min(dist);
                    }
                }
            }
        }
        if !(wall.is_finite() && wall < 0.5 * cell) {
            return;
        }
        let world = self.to_local.inverse();
        let mut push = |p: [f64; 3]| {
            let w = world.apply(p);
            for a in 0..3 {
                out[a].push(w[a]);
            }
        };
        let step = wall.max(cell / 64.0);
        let sampled = |c: &Curve2| -> Vec<[f64; 2]> {
            let n = if c.line().is_some() {
                1
            } else {
                ((c.length() / step).ceil() as usize).clamp(8, 512)
            };
            c.polyline(n)
        };
        if let Shape::Revolve {
            ap,
            ad,
            e1,
            sense,
            angle,
            ..
        } = &self.shape
        {
            // Every radius the profile reaches, at even angles about the axis.
            let mut radii: Vec<f64> = Vec::new();
            let mut hs: Vec<f64> = Vec::new();
            for c in &curves {
                for q in sampled(c) {
                    let rel = [q[0] - ap[0], q[1] - ap[1]];
                    hs.push(rel[0] * ad[0] + rel[1] * ad[1]);
                    radii.push(rel[0] * e1[0] + rel[1] * e1[1]);
                }
            }
            radii.sort_by(f64::total_cmp);
            radii.dedup_by(|a, b| (*a - *b).abs() < wall / 4.0);
            radii.truncate(128);
            let h = hs.iter().sum::<f64>() / hs.len().max(1) as f64;
            let ez = sense * (ad[0] * e1[1] - ad[1] * e1[0]);
            // Steps about one wall thickness long at the outer radius.
            let rmax = radii.iter().fold(0.0f64, |a, r| a.max(r.abs()));
            let m = ((angle * rmax / wall).ceil() as usize).clamp(4, 512);
            for r in radii {
                for k in 0..=m {
                    let th = angle * k as f64 / m as f64;
                    push([
                        ap[0] + h * ad[0] + r * th.cos() * e1[0],
                        ap[1] + h * ad[1] + r * th.cos() * e1[1],
                        r * th.sin() * ez,
                    ]);
                }
            }
            return;
        }
        for c in &curves {
            if c.line().is_some() {
                continue;
            }
            for q in sampled(c) {
                push([q[0], q[1], 0.0]);
            }
        }
    }

    /// Breaks along a drafted extrusion's tapered sides. Each of `RES` slices of the
    /// extrusion's length gets breaks where the straight sides sit at the slice's middle, so
    /// cell centres sample the taper by the midpoint rule however small the taper is beside
    /// the uniform breaks.
    pub(super) fn draft_breaks(&self, x: &Extrusion, out: &mut [Vec<f64>; 3]) {
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
}
