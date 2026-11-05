//! Joining a view's line runs: hidden runs that a visible line already draws are dropped.

use super::ViewLines;

pub(super) fn push(out: &mut ViewLines, hidden: bool, pts: Vec<f64>) {
    if hidden {
        out.hidden.push(pts)
    } else {
        out.visible.push(pts)
    }
}

/// Hidden runs that lie on a visible line are drawn by the visible line.
pub(super) fn merge_overlaps(out: &mut ViewLines) {
    let on_visible = |x: f64, y: f64, visible: &[Vec<f64>]| {
        visible.iter().any(|l| {
            l.chunks_exact(2).collect::<Vec<_>>().windows(2).any(|w| {
                let (a, b) = ((w[0][0], w[0][1]), (w[1][0], w[1][1]));
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let len2 = dx * dx + dy * dy;
                if len2 < 1e-12 {
                    return false;
                }
                let t = (((x - a.0) * dx + (y - a.1) * dy) / len2).clamp(0.0, 1.0);
                let (px, py) = (a.0 + t * dx, a.1 + t * dy);
                (px - x).powi(2) + (py - y).powi(2) < 1e-6
            })
        })
    };
    let visible = out.visible.clone();
    out.hidden.retain(|l| {
        let n = l.len() / 2;
        let mid = |i: usize| (l[2 * i], l[2 * i + 1]);
        let (a, b) = (mid(0), mid(n - 1));
        let c = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        !(on_visible(a.0, a.1, &visible)
            && on_visible(b.0, b.1, &visible)
            && on_visible(c.0, c.1, &visible))
    });
}
