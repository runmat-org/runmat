//! Joining a view's line runs: hidden runs that a visible line already draws are dropped.

use super::ViewLines;
use std::collections::HashMap;

pub(super) fn push(out: &mut ViewLines, hidden: bool, pts: Vec<f64>) {
    if hidden {
        out.hidden.push(pts)
    } else {
        out.visible.push(pts)
    }
}

/// How near (view mm) a point lies to a visible segment to count as on it.
const ON: f64 = 1e-3;

/// The visible lines' segments, bucketed on a square grid in view space, so a point tests only
/// the segments that pass near it. An assembly view holds hundreds of thousands of runs; testing
/// every hidden run against every visible one does not finish.
struct Segments {
    segs: Vec<[f64; 4]>,
    cell: f64,
    cells: HashMap<(i64, i64), Vec<u32>>,
}

impl Segments {
    fn new(visible: &[Vec<f64>]) -> Segments {
        let mut segs = Vec::new();
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for l in visible {
            for w in l.chunks_exact(2).collect::<Vec<_>>().windows(2) {
                let s = [w[0][0], w[0][1], w[1][0], w[1][1]];
                if (s[2] - s[0]).powi(2) + (s[3] - s[1]).powi(2) < 1e-12 {
                    continue;
                }
                for k in 0..2 {
                    lo[k] = lo[k].min(s[k]).min(s[k + 2]);
                    hi[k] = hi[k].max(s[k]).max(s[k + 2]);
                }
                segs.push(s);
            }
        }
        // About as many cells as segments over the lines' box, so a cell holds a few.
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-6);
        let n = (segs.len() as f64).sqrt().clamp(1.0, 4096.0);
        let cell = (span / n).max(ON * 4.0);
        let mut cells: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
        for (i, s) in segs.iter().enumerate() {
            let at = |v: f64| (v / cell).floor() as i64;
            let (y0, y1) = (at(s[1].min(s[3]) - ON), at(s[1].max(s[3]) + ON));
            // Row by row, the cells the segment (widened by `ON`) crosses in that row, so a
            // long diagonal line enters only the cells along it.
            for y in y0..=y1 {
                let (b0, b1) = (y as f64 * cell - ON, (y + 1) as f64 * cell + ON);
                let x_at = |v: f64| {
                    if (s[3] - s[1]).abs() < 1e-12 {
                        None
                    } else {
                        Some(s[0] + (s[2] - s[0]) * ((v - s[1]) / (s[3] - s[1])).clamp(0.0, 1.0))
                    }
                };
                let (xa, xb) = match (x_at(b0), x_at(b1)) {
                    (Some(a), Some(b)) => (a.min(b), a.max(b)),
                    _ => (s[0].min(s[2]), s[0].max(s[2])),
                };
                for x in at(xa - ON)..=at(xb + ON) {
                    cells.entry((x, y)).or_default().push(i as u32);
                }
            }
        }
        Segments { segs, cell, cells }
    }

    /// Whether `(x, y)` lies on a visible segment.
    fn on(&self, x: f64, y: f64) -> bool {
        let key = (
            (x / self.cell).floor() as i64,
            (y / self.cell).floor() as i64,
        );
        let Some(list) = self.cells.get(&key) else {
            return false;
        };
        list.iter().any(|&i| {
            let [ax, ay, bx, by] = self.segs[i as usize];
            let (dx, dy) = (bx - ax, by - ay);
            let len2 = dx * dx + dy * dy;
            let t = (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0);
            let (px, py) = (ax + t * dx, ay + t * dy);
            (px - x).powi(2) + (py - y).powi(2) < ON * ON
        })
    }
}

/// Hidden runs that lie on a visible line are drawn by the visible line.
pub(super) fn merge_overlaps(out: &mut ViewLines) {
    if out.hidden.is_empty() {
        return;
    }
    let visible = Segments::new(&out.visible);
    out.hidden.retain(|l| {
        let n = l.len() / 2;
        let mid = |i: usize| (l[2 * i], l[2 * i + 1]);
        let (a, b) = (mid(0), mid(n - 1));
        let c = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        !(visible.on(a.0, a.1) && visible.on(b.0, b.1) && visible.on(c.0, c.1))
    });
}
