//! Shared curves on the display mesh: each face's charts, the angular and chordal step, runs
//! sampled along their curves, closed rims at even angles, and each face's boundary loops.

use super::*;

impl Ctx<'_> {
    /// The surfaces of `faces`, when every one draws on a chart.
    pub(super) fn surfaces(&self, faces: &BTreeSet<usize>) -> Option<Vec<Arc<Exact>>> {
        let mut out = Vec::new();
        for f in faces {
            match self.draw.get(*f)? {
                Draw::Chart(e) => out.push(e.clone()),
                _ => return None,
            }
        }
        Some(out)
    }

    pub(super) fn chart(&self, f: usize) -> Option<Arc<Exact>> {
        match &self.draw[f] {
            Draw::Chart(e) => Some(e.clone()),
            _ => None,
        }
    }

    /// The largest step along a curve of radius `r` for the tolerances.
    pub(super) fn step_for(&self, r: f64, tol: f64) -> f64 {
        if !r.is_finite() {
            return f64::INFINITY;
        }
        let chord = if tol < r {
            2.0 * (1.0 - tol / r).acos()
        } else {
            FRAC_PI_2
        };
        r * self.dtheta.min(chord).max(1e-3)
    }

    pub(super) fn sample_run(&mut self, r: usize, tol: f64) {
        let (fa, fb) = self.runs[r].faces;
        let grid = self.runs[r].grid.clone();
        let closed = grid.first() == grid.last() && grid.len() > 2;
        let (sa, sb) = (self.chart(fa), self.chart(fb));
        let (Some(sa), Some(sb)) = (sa, sb) else {
            self.runs[r].pts = grid;
            return;
        };
        let curved = sa.curved() || sb.curved();
        if !curved {
            // Two planes meet on a line: the ends are enough when the run is straight.
            let (p0, p1) = (self.pos[grid[0]], self.pos[grid[grid.len() - 1]]);
            let d = sub(p1, p0);
            let l = len(d);
            let straight = !closed
                && l > 0.0
                && grid.iter().all(|g| {
                    let q = sub(self.raw[*g], self.raw[grid[0]]);
                    let dr = sub(self.raw[grid[grid.len() - 1]], self.raw[grid[0]]);
                    len(cross(q, dr)) <= 1e-9 * len(dr).max(1e-300) * self.diag
                });
            self.runs[r].pts = if straight {
                vec![grid[0], grid[grid.len() - 1]]
            } else {
                grid
            };
            return;
        }
        let surfs = vec![sa.clone(), sb.clone()];
        // Interior points onto the curve; the ends are corners, already placed.
        let mut pts: Vec<P3> = Vec::with_capacity(grid.len());
        for (i, g) in grid.iter().enumerate() {
            let end = !closed && (i == 0 || i == grid.len() - 1);
            pts.push(if end {
                self.pos[*g]
            } else {
                snap_or(self.raw[*g], &surfs, self.diag)
            });
        }
        if closed {
            let q = snap_or(self.raw[grid[0]], &surfs, self.diag);
            pts[0] = q;
            let n = pts.len();
            pts[n - 1] = q;
            self.pos[grid[0]] = q;
        }
        let clean = clean_polyline(&pts);
        let total: f64 = clean.windows(2).map(|w| len(sub(w[1], w[0]))).sum();
        if total <= 1e-12 * self.diag {
            self.runs[r].pts = grid;
            return;
        }
        // Straight curves (a generator line, a tangent line) need only their ends.
        if !closed {
            let (p0, p1) = (clean[0], clean[clean.len() - 1]);
            let chord = sub(p1, p0);
            let cl = len(chord);
            if cl > 1e-12 {
                let dev = clean
                    .iter()
                    .map(|q| len(cross(sub(*q, p0), chord)) / cl)
                    .fold(0.0, f64::max);
                if dev <= tol {
                    self.runs[r].pts = vec![grid[0], grid[grid.len() - 1]];
                    return;
                }
            }
        }
        let radius = clean
            .iter()
            .flat_map(|q| {
                let mut v = Vec::new();
                for s in [&sa, &sb] {
                    if s.curved() {
                        let rr = s.radii(*q);
                        v.push(rr[0].min(rr[1]));
                    }
                }
                v
            })
            .fold(f64::INFINITY, f64::min);
        let h = self.step_for(radius, tol).max(total / 4000.0);
        let mut n = (total / h).ceil() as usize;
        n = n.max(if closed { 12 } else { 1 });
        // A closed curve around a periodic surface (a hole's rim) is sampled at even steps
        // of the surface's angle from its seam, so rims of one surface line up.
        if closed {
            // A multiple of four, so the quadrant points (the silhouettes of the standard
            // views) are samples.
            let n4 = n.div_ceil(4) * 4;
            if let Some(out) = self.sample_closed_by_angle(r, &clean, n4, &surfs, &sa, &sb) {
                self.runs[r].pts = out;
                return;
            }
        }
        let mut cum = vec![0.0];
        for w in clean.windows(2) {
            let l = cum[cum.len() - 1] + len(sub(w[1], w[0]));
            cum.push(l);
        }
        let mut out = vec![grid[0]];
        for i in 1..n {
            let t = total * i as f64 / n as f64;
            let k = cum.partition_point(|c| *c < t).clamp(1, clean.len() - 1);
            let seg = cum[k] - cum[k - 1];
            let f = if seg > 0.0 {
                (t - cum[k - 1]) / seg
            } else {
                0.0
            };
            let guess = lerp(clean[k - 1], clean[k], f);
            let q = snap_or(guess, &surfs, self.diag);
            self.pos.push(q);
            self.raw.push(q);
            out.push(self.pos.len() - 1);
        }
        out.push(grid[grid.len() - 1]);
        self.runs[r].pts = out;
    }

    pub(super) fn sample_closed_by_angle(
        &mut self,
        r: usize,
        clean: &[P3],
        n: usize,
        surfs: &[Arc<Exact>],
        sa: &Arc<Exact>,
        sb: &Arc<Exact>,
    ) -> Option<Vec<usize>> {
        let s = [sa, sb].into_iter().find(|s| s.period_u().is_some())?;
        let per = s.period_u()?;
        let mut uv: Vec<[f64; 2]> = clean.iter().map(|q| s.invert(*q)).collect();
        for k in 1..uv.len() {
            while uv[k][0] - uv[k - 1][0] > per / 2.0 {
                uv[k][0] -= per;
            }
            while uv[k][0] - uv[k - 1][0] < -per / 2.0 {
                uv[k][0] += per;
            }
        }
        let span = uv[uv.len() - 1][0] - uv[0][0];
        if (span.abs() - per).abs() > 0.05 * per {
            return None;
        }
        let dir = span.signum();
        // Even angles from the seam (u = 0), in the curve's direction; v interpolated.
        let at = |u: f64| -> f64 {
            // u within the unwrapped range, by linear search.
            for w in uv.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (lo, hi) = (a[0].min(b[0]), a[0].max(b[0]));
                if u >= lo - 1e-15 && u <= hi + 1e-15 {
                    let t = if (b[0] - a[0]).abs() > 1e-300 {
                        (u - a[0]) / (b[0] - a[0])
                    } else {
                        0.0
                    };
                    return a[1] + t * (b[1] - a[1]);
                }
            }
            uv[0][1]
        };
        let u0 = uv[0][0];
        // The first sample at or after the start in the curve's direction, on the even grid.
        let step = per / n as f64;
        let k0 = if dir > 0.0 {
            (u0 / step).ceil()
        } else {
            (u0 / step).floor()
        };
        let grid0 = self.runs[r].grid[0];
        let mut out = Vec::with_capacity(n + 1);
        for i in 0..n {
            let u = (k0 + dir * i as f64) * step;
            // Keep u inside the unwrapped range.
            let mut uu = u;
            let (lo, hi) = (u0.min(u0 + span), u0.max(u0 + span));
            while uu < lo {
                uu += per;
            }
            while uu > hi {
                uu -= per;
            }
            let guess = s.eval(uu, at(uu));
            let q = snap_or(guess, surfs, self.diag);
            if i == 0 {
                self.pos[grid0] = q;
                out.push(grid0);
            } else {
                self.pos.push(q);
                self.raw.push(q);
                out.push(self.pos.len() - 1);
            }
        }
        out.push(grid0);
        Some(out)
    }

    /// A face's loops as point indices (each loop once around, no repeated end).
    pub(super) fn loop_points(&self, face: usize) -> Vec<Vec<usize>> {
        self.loops[face]
            .iter()
            .map(|lp| {
                let mut out: Vec<usize> = Vec::new();
                for (r, rev) in lp {
                    let pts = &self.runs[*r].pts;
                    let it: Box<dyn Iterator<Item = &usize>> = if *rev {
                        Box::new(pts.iter().rev())
                    } else {
                        Box::new(pts.iter())
                    };
                    for p in it {
                        if out.last() != Some(p) {
                            out.push(*p);
                        }
                    }
                }
                if out.len() > 1 && out.first() == out.last() {
                    out.pop();
                }
                out
            })
            .collect()
    }
}
