//! Section lining by material: the pattern a cut region draws (ASME Y14.2 section-lining
//! symbols, and the architectural symbols for building materials). Each pattern is drawn in
//! sheet millimetres, so it reads the same at every view scale, and is clipped to the cut
//! regions even-odd, so a hole in a cut stays clear.
//!
//! - general: 45° lines (ISO 128-50), the default for a material without its own symbol;
//! - steel: 45° lines in pairs;
//! - concrete: stippling with small triangles of aggregate;
//! - wood across the grain: diagonals corner to corner of each piece;
//! - insulation (batt, loose fill): a row of loops across the cavity;
//! - rigid insulation: 45° lines both ways;
//! - gypsum and plaster: fine stippling;
//! - earth and fill: short strokes in groups of three, alternating direction.

use super::marks::{line, prim};
use super::{Prim, Shape, Std, Xy};
use runmat_cad::model::materials::SectionSymbol;

/// The lining of `symbol` over the regions `rings` (flat x, y, sheet mm), `spacing` apart.
pub fn lining(
    std: &Std,
    id: &str,
    rings: &[Vec<f64>],
    symbol: SectionSymbol,
    spacing: f64,
) -> Vec<Prim> {
    let region = Region::new(rings);
    if region.edges.is_empty() {
        return Vec::new();
    }
    let mark = |shape: Shape| prim(id, "view", "Views", "hatch", shape);
    let strokes = |runs: Vec<Vec<Xy>>| -> Vec<Prim> {
        runs.into_iter()
            .map(|run| mark(line(run.into_iter().flatten().collect())))
            .collect()
    };
    match symbol {
        SectionSymbol::General => std.hatch(id, rings, spacing),
        SectionSymbol::Steel => {
            let mut out = strokes(region.parallel(45.0, 2.0 * spacing, 0.0));
            out.extend(strokes(region.parallel(45.0, 2.0 * spacing, 0.6)));
            out
        }
        SectionSymbol::RigidInsulation => {
            let mut out = strokes(region.parallel(45.0, spacing, 0.5));
            out.extend(strokes(region.parallel(-45.0, spacing, 0.5)));
            out
        }
        SectionSymbol::Wood => strokes(region.pieces().flat_map(|(lo, hi)| {
            let mut runs = region.clip(&[lo, hi]);
            runs.extend(region.clip(&[[lo[0], hi[1]], [hi[0], lo[1]]]));
            runs
        })
        .collect()),
        SectionSymbol::Insulation => strokes(
            region
                .pieces()
                .flat_map(|(lo, hi)| region.clip(&loops(lo, hi)))
                .collect(),
        ),
        SectionSymbol::Gypsum => region
            .scatter(0.8 * spacing, 1)
            .into_iter()
            .map(|(p, _)| mark(dot(p)))
            .collect(),
        SectionSymbol::Concrete => {
            let mut out = Vec::new();
            for (p, k) in region.scatter(1.2 * spacing, 2) {
                if k % 5 == 0 {
                    // A triangle of aggregate, turned by its cell, drawn only when whole.
                    let size = 1.4 * spacing;
                    let turn = (k as f64 * 0.7).rem_euclid(std::f64::consts::TAU);
                    let tri: Vec<Xy> = (0..4)
                        .map(|i| {
                            let a = turn + i as f64 * std::f64::consts::TAU / 3.0;
                            [p[0] + size / 2.0 * a.cos(), p[1] + size / 2.0 * a.sin()]
                        })
                        .collect();
                    if tri.iter().all(|q| region.inside(*q)) {
                        out.push(mark(line(tri.into_iter().flatten().collect())));
                    }
                } else {
                    out.push(mark(dot(p)));
                }
            }
            out
        }
        SectionSymbol::Earth => {
            let cell = 3.0 * spacing;
            let mut runs = Vec::new();
            for (i, j) in region.cells(cell) {
                {
                    let c = [(i as f64 + 0.5) * cell, (j as f64 + 0.5) * cell];
                    // Up to the right in one cell, down to the right in the next.
                    let (d, n) = if (i + j) % 2 == 0 {
                        ([1.0, 1.0], [-1.0, 1.0])
                    } else {
                        ([1.0, -1.0], [1.0, 1.0])
                    };
                    let half = 0.3 * cell / std::f64::consts::SQRT_2;
                    let gap = 0.18 * cell / std::f64::consts::SQRT_2;
                    for k in [-1.0, 0.0, 1.0] {
                        let m = [c[0] + n[0] * gap * k, c[1] + n[1] * gap * k];
                        runs.extend(region.clip(&[
                            [m[0] - d[0] * half, m[1] - d[1] * half],
                            [m[0] + d[0] * half, m[1] + d[1] * half],
                        ]));
                    }
                }
            }
            strokes(runs)
        }
    }
}

/// A stipple dot.
fn dot(p: Xy) -> Shape {
    Shape::Circle {
        x: p[0],
        y: p[1],
        r: 0.1,
    }
}

/// The batt symbol across the piece `lo`–`hi`: loops along its long side, as tall as the piece
/// is thick, one loop for each half thickness along it.
fn loops(lo: Xy, hi: Xy) -> Vec<Xy> {
    let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
    let (long, thick) = (w.max(h), w.min(h));
    if thick <= 0.0 {
        return Vec::new();
    }
    // A prolate cycloid: x = a t - b sin t, y = c cos t, looping where b > a.
    let pitch = 0.5 * thick;
    let a = pitch / std::f64::consts::TAU;
    let b = 0.22 * thick;
    let c = 0.45 * thick;
    let turns = (long / pitch).ceil() as usize;
    let steps = turns * 24;
    (0..=steps)
        .map(|k| {
            let t = k as f64 * std::f64::consts::TAU / 24.0;
            let along = a * t - b * t.sin();
            let across = thick / 2.0 + c * t.cos();
            if w >= h {
                [lo[0] + along, lo[1] + across]
            } else {
                [lo[0] + across, lo[1] + along]
            }
        })
        .collect()
}

/// Closed rings as one region, even-odd.
struct Region {
    rings: Vec<Vec<Xy>>,
    edges: Vec<(Xy, Xy)>,
}

impl Region {
    fn new(rings: &[Vec<f64>]) -> Self {
        let rings: Vec<Vec<Xy>> = rings
            .iter()
            .map(|r| r.chunks_exact(2).map(|p| [p[0], p[1]]).collect::<Vec<Xy>>())
            .filter(|r| r.len() >= 3)
            .collect();
        let edges = rings
            .iter()
            .flat_map(|r| {
                (0..r.len()).filter_map(move |i| {
                    let (a, b) = (r[i], r[(i + 1) % r.len()]);
                    (a != b).then_some((a, b))
                })
            })
            .collect();
        Region { rings, edges }
    }

    /// The grid cells, `cell` wide, that the pieces' boxes touch, each once.
    fn cells(&self, cell: f64) -> std::collections::BTreeSet<(i64, i64)> {
        let mut out = std::collections::BTreeSet::new();
        for (lo, hi) in self.pieces() {
            let (i0, i1) = ((lo[0] / cell).floor() as i64, (hi[0] / cell).ceil() as i64);
            let (j0, j1) = ((lo[1] / cell).floor() as i64, (hi[1] / cell).ceil() as i64);
            for i in i0..i1 {
                for j in j0..j1 {
                    out.insert((i, j));
                }
            }
        }
        out
    }

    /// Each ring's box: the pieces a per-piece symbol (wood, batt) draws across.
    fn pieces(&self) -> impl Iterator<Item = (Xy, Xy)> + '_ {
        self.rings.iter().map(|r| bounds(r.iter()))
    }

    /// Even-odd: inside when a ray to the right crosses the edges an odd number of times.
    fn inside(&self, p: Xy) -> bool {
        self.edges
            .iter()
            .filter(|(a, b)| {
                (a[1] > p[1]) != (b[1] > p[1])
                    && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            })
            .count()
            % 2
            == 1
    }

    /// The runs of the polyline `path` inside the region.
    fn clip(&self, path: &[Xy]) -> Vec<Vec<Xy>> {
        let mut runs: Vec<Vec<Xy>> = Vec::new();
        let mut open = false;
        for w in path.windows(2) {
            let (p, q) = (w[0], w[1]);
            let d = [q[0] - p[0], q[1] - p[1]];
            let mut cuts = vec![0.0, 1.0];
            for (a, b) in &self.edges {
                let e = [b[0] - a[0], b[1] - a[1]];
                let den = d[0] * e[1] - d[1] * e[0];
                if den.abs() < 1e-12 {
                    continue;
                }
                let f = [a[0] - p[0], a[1] - p[1]];
                let t = (f[0] * e[1] - f[1] * e[0]) / den;
                let u = (f[0] * d[1] - f[1] * d[0]) / den;
                if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                    cuts.push(t);
                }
            }
            cuts.sort_by(f64::total_cmp);
            let at = |t: f64| [p[0] + d[0] * t, p[1] + d[1] * t];
            for c in cuts.windows(2) {
                if c[1] - c[0] < 1e-9 {
                    continue;
                }
                if self.inside(at((c[0] + c[1]) / 2.0)) {
                    if !open {
                        runs.push(vec![at(c[0])]);
                        open = true;
                    }
                    if let Some(run) = runs.last_mut() {
                        run.push(at(c[1]));
                    }
                } else {
                    open = false;
                }
            }
        }
        runs
    }

    /// Lines at `degrees` from the x axis, `spacing` apart, shifted by `phase` of a spacing.
    fn parallel(&self, degrees: f64, spacing: f64, phase: f64) -> Vec<Vec<Xy>> {
        let (s, c) = degrees.to_radians().sin_cos();
        let (dir, normal) = ([c, s], [-s, c]);
        let along = |p: &Xy| p[0] * dir[0] + p[1] * dir[1];
        let across = |p: &Xy| p[0] * normal[0] + p[1] * normal[1];
        let all: Vec<&Xy> = self.rings.iter().flatten().collect();
        let fold = |f: &dyn Fn(&Xy) -> f64| {
            all.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(f(p)), hi.max(f(p)))
            })
        };
        let ((a0, a1), (n0, n1)) = (fold(&along), fold(&across));
        let mut out = Vec::new();
        let mut n = ((n0 / spacing).floor() + phase) * spacing;
        while n <= n1 {
            if n >= n0 {
                let at = |a: f64| [dir[0] * a + normal[0] * n, dir[1] * a + normal[1] * n];
                out.extend(self.clip(&[at(a0 - 1.0), at(a1 + 1.0)]));
            }
            n += spacing;
        }
        out
    }

    /// Points jittered about a grid `cell` apart, inside the region, each with its cell's
    /// number (the same region scatters the same way). `per_cell` points per cell.
    fn scatter(&self, cell: f64, per_cell: usize) -> Vec<(Xy, u64)> {
        let mut out = Vec::new();
        for (i, j) in self.cells(cell) {
            {
                for k in 0..per_cell {
                    let h = mix((i as u64) << 32 ^ (j as u64 & 0xffff_ffff) ^ (k as u64) << 60);
                    let fx = (h & 0xffff) as f64 / 65535.0;
                    let fy = ((h >> 16) & 0xffff) as f64 / 65535.0;
                    let p = [(i as f64 + fx) * cell, (j as f64 + fy) * cell];
                    if self.inside(p) {
                        out.push((p, h >> 32));
                    }
                }
            }
        }
        out
    }
}

/// A 64-bit mix (SplitMix64's finalizer): a fixed jitter per grid cell.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn bounds<'a>(points: impl Iterator<Item = &'a Xy>) -> (Xy, Xy) {
    points.fold(
        ([f64::MAX, f64::MAX], [f64::MIN, f64::MIN]),
        |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1])],
                [hi[0].max(p[0]), hi[1].max(p[1])],
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn std() -> Std {
        Std::of(&serde_json::Value::Null)
    }

    /// A 40 x 10 mm piece with a 10 x 4 mm hole in its middle.
    fn holed() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 40.0, 0.0, 40.0, 10.0, 0.0, 10.0, 0.0, 0.0],
            vec![15.0, 3.0, 25.0, 3.0, 25.0, 7.0, 15.0, 7.0, 15.0, 3.0],
        ]
    }

    fn points(prims: &[Prim]) -> Vec<Xy> {
        prims
            .iter()
            .flat_map(|p| match &p.shape {
                Shape::Polyline { points, .. } => {
                    points.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
                }
                Shape::Circle { x, y, .. } => vec![[*x, *y]],
                _ => Vec::new(),
            })
            .collect()
    }

    #[test]
    fn every_symbol_draws_inside_its_cut_and_clear_of_its_hole() {
        use SectionSymbol::*;
        let rings = holed();
        for symbol in [
            General,
            Steel,
            Concrete,
            Wood,
            Insulation,
            RigidInsulation,
            Gypsum,
            Earth,
        ] {
            let prims = lining(&std(), "view:a", &rings, symbol, 1.0);
            assert!(!prims.is_empty(), "{symbol:?} draws nothing");
            assert!(prims.iter().all(|p| p.style == "hatch"), "{symbol:?}");
            for p in points(&prims) {
                let tol = 1e-6;
                let outside = p[0] < -tol || p[0] > 40.0 + tol || p[1] < -tol || p[1] > 10.0 + tol;
                let in_hole = p[0] > 15.0 + tol
                    && p[0] < 25.0 - tol
                    && p[1] > 3.0 + tol
                    && p[1] < 7.0 - tol;
                assert!(!outside && !in_hole, "{symbol:?} draws at {p:?}");
            }
        }
    }

    #[test]
    fn symbols_read_apart() {
        use SectionSymbol::*;
        let rings = vec![vec![0.0, 0.0, 40.0, 0.0, 40.0, 10.0, 0.0, 10.0, 0.0, 0.0]];
        let draw = |s| lining(&std(), "view:a", &rings, s, 1.0);
        // Stippling is dots; batt is a few long looped lines; wood is the two diagonals.
        assert!(draw(Gypsum)
            .iter()
            .all(|p| matches!(p.shape, Shape::Circle { .. })));
        assert_eq!(draw(Wood).len(), 2);
        let batt = draw(Insulation);
        assert!(!batt.is_empty() && batt.len() < 4, "{} runs", batt.len());
        // Steel pairs its lines: twice the general hatch's lines at the same spacing, about.
        assert!(draw(Steel).len() >= draw(General).len() / 2);
        assert!(draw(Concrete)
            .iter()
            .any(|p| matches!(p.shape, Shape::Polyline { .. })));
    }
}
