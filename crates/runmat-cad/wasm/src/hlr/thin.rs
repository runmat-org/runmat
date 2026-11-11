//! A view's lines as few points as the sheet needs.
//!
//! The hidden-line pass classifies each model edge in pieces of at most 0.5 mm and emits each
//! edge segment as its own run, so an assembly view holds tens of thousands of short polylines.
//! Before the view's lines go into the answer, [`thin`] joins runs that meet end to end into
//! one polyline, drops the points a straight or gently curved run doesn't need (a chord
//! tolerance of [`SHEET_TOLERANCE`] on the sheet, so in model millimetres it scales with the
//! view), and rounds coordinates to a grid finer than that tolerance. At sheet resolution the
//! lines draw the same.
//!
//! Picks (what a click on a line means) join only runs with the same reference. A view that
//! draws visible lines only keeps no hidden lines.

use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// The largest distance (sheet mm) a thinned line moves from the line the pass computed: 10 µm,
/// a quarter of a 600 dpi printer dot.
pub const SHEET_TOLERANCE: f64 = 0.01;

/// Thins one view's `hlr` entry drawn at `scale` (sheet mm per model mm). `hidden`: whether
/// the view draws hidden lines.
pub fn thin(entry: &mut Value, scale: f64, hidden: bool) {
    let Value::Object(o) = entry else { return };
    let tol = SHEET_TOLERANCE / scale.max(1e-9);
    // A grid at most a quarter of the tolerance: 1 µm in model mm at 1:2 to 2:1.
    let grid = 10f64.powf((tol / 4.0).log10().floor());
    if let Some(Value::Array(lines)) = o.get_mut("visible") {
        *lines = runs_out(thin_runs(runs_in(lines), tol, grid));
    }
    if let Some(Value::Array(lines)) = o.get_mut("hidden") {
        *lines = if hidden {
            runs_out(thin_runs(runs_in(lines), tol, grid))
        } else {
            Vec::new()
        };
    }
    if let Some(Value::Array(picks)) = o.get_mut("picks") {
        *picks = thin_picks(std::mem::take(picks), tol, grid);
    }
}

fn runs_in(lines: &mut Vec<Value>) -> Vec<Vec<f64>> {
    std::mem::take(lines)
        .into_iter()
        .map(|l| {
            l.as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default()
        })
        .collect()
}

fn runs_out(runs: Vec<Vec<f64>>) -> Vec<Value> {
    runs.into_iter()
        .map(|r| Value::Array(r.into_iter().map(Value::from).collect()))
        .collect()
}

/// Picks with the same reference joined and thinned, in the order each reference first appears.
fn thin_picks(picks: Vec<Value>, tol: f64, grid: f64) -> Vec<Value> {
    let mut order: Vec<String> = Vec::new();
    type PickRuns = (Map<String, Value>, Vec<Vec<f64>>);
    let mut by: BTreeMap<String, PickRuns> = BTreeMap::new();
    let mut other = Vec::new();
    for p in picks {
        let Value::Object(mut o) = p else {
            other.push(p);
            continue;
        };
        let Some(reference) = o
            .get("reference")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            other.push(Value::Object(o));
            continue;
        };
        let points: Vec<f64> = o
            .remove("points")
            .and_then(|v| {
                v.as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).collect())
            })
            .unwrap_or_default();
        let e = by.entry(reference.clone()).or_insert_with(|| {
            order.push(reference);
            (o, Vec::new())
        });
        e.1.push(points);
    }
    let mut out = Vec::new();
    for reference in order {
        let Some((fields, runs)) = by.remove(&reference) else {
            continue;
        };
        for run in thin_runs(runs, tol, grid) {
            let mut o = fields.clone();
            o.insert(
                "points".into(),
                Value::Array(run.into_iter().map(Value::from).collect()),
            );
            out.push(Value::Object(o));
        }
    }
    out.extend(other);
    out
}

/// Runs joined end to end, simplified to `tol` and rounded to `grid`.
pub fn thin_runs(runs: Vec<Vec<f64>>, tol: f64, grid: f64) -> Vec<Vec<f64>> {
    chain(runs)
        .into_iter()
        .filter_map(|r| {
            let pts: Vec<[f64; 2]> = r.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            let kept = simplify(&pts, tol);
            let mut out: Vec<f64> = Vec::with_capacity(kept.len() * 2);
            // Divided by a whole power of ten, so the coordinate prints as its short decimal.
            let inv = (1.0 / grid).round();
            for p in kept {
                let q = [(p[0] * inv).round() / inv, (p[1] * inv).round() / inv];
                let n = out.len();
                if n >= 2 && out[n - 2] == q[0] && out[n - 1] == q[1] {
                    continue;
                }
                out.extend(q);
            }
            (out.len() >= 4).then_some(out)
        })
        .collect()
}

fn key(x: f64, y: f64) -> (i64, i64) {
    ((x * 1e6).round() as i64, (y * 1e6).round() as i64)
}

/// Runs that meet end to end (either way round) joined into one, in a deterministic order.
fn chain(runs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let runs: Vec<Vec<f64>> = runs.into_iter().filter(|r| r.len() >= 4).collect();
    let ends = |r: &Vec<f64>| (key(r[0], r[1]), key(r[r.len() - 2], r[r.len() - 1]));
    let mut at: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (i, r) in runs.iter().enumerate() {
        let (a, b) = ends(r);
        at.entry(a).or_default().push(i);
        if b != a {
            at.entry(b).or_default().push(i);
        }
    }
    let mut used = vec![false; runs.len()];
    let mut out = Vec::new();
    for i in 0..runs.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut line = runs[i].clone();
        // Forward from the end, then backward from the start.
        for forward in [true, false] {
            loop {
                let n = line.len();
                let end = if forward {
                    key(line[n - 2], line[n - 1])
                } else {
                    key(line[0], line[1])
                };
                if !forward && key(line[n - 2], line[n - 1]) == end {
                    break;
                }
                let Some(j) = at
                    .get(&end)
                    .and_then(|c| c.iter().copied().find(|&j| !used[j]))
                else {
                    break;
                };
                used[j] = true;
                let r = &runs[j];
                let (a, _) = ends(r);
                // `next` runs away from `end`.
                let next: Vec<f64> = if a == end {
                    r.clone()
                } else {
                    r.chunks_exact(2).rev().flatten().copied().collect()
                };
                if forward {
                    line.extend_from_slice(&next[2..]);
                } else {
                    let mut back: Vec<f64> =
                        next.chunks_exact(2).rev().flatten().copied().collect();
                    back.truncate(back.len() - 2);
                    back.extend_from_slice(&line);
                    line = back;
                }
                let m = line.len();
                if key(line[0], line[1]) == key(line[m - 2], line[m - 1]) {
                    break;
                }
            }
        }
        out.push(line);
    }
    out
}

/// The points of `pts` a polyline needs to stay within `tol` of it (Douglas-Peucker; the ends
/// stay).
fn simplify(pts: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (p, q) = (pts[a], pts[b]);
        let d = [q[0] - p[0], q[1] - p[1]];
        let len2 = d[0] * d[0] + d[1] * d[1];
        let (mut far, mut at) = (0.0, a);
        for (k, r) in pts.iter().enumerate().take(b).skip(a + 1) {
            // To the segment, not its line: a run that doubles back along itself keeps its turn.
            let e = [r[0] - p[0], r[1] - p[1]];
            let t = if len2 < 1e-24 {
                0.0
            } else {
                ((e[0] * d[0] + e[1] * d[1]) / len2).clamp(0.0, 1.0)
            };
            let dist = ((e[0] - t * d[0]).powi(2) + (e[1] - t * d[1]).powi(2)).sqrt();
            if dist > far {
                far = dist;
                at = k;
            }
        }
        if far > tol {
            keep[at] = true;
            stack.push((a, at));
            stack.push((at, b));
        }
    }
    pts.iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| *p)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_end_to_end_join_and_straight_points_drop() {
        // Three pieces of one straight line, the middle one reversed, and a separate run.
        let runs = vec![
            vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0],
            vec![2.0, 0.0, 1.5, 0.0, 1.0, 0.0],
            vec![2.0, 0.0, 3.0, 0.0],
            vec![0.0, 5.0, 1.0, 5.0],
        ];
        let out = thin_runs(runs, 0.01, 0.001);
        assert_eq!(
            out,
            vec![vec![0.0, 0.0, 3.0, 0.0], vec![0.0, 5.0, 1.0, 5.0]]
        );
    }

    #[test]
    fn a_run_that_doubles_back_keeps_its_far_end() {
        // Out along a line and back over it (two runs drawn over each other, end to end).
        let runs = vec![
            vec![0.0, 0.0, 5.0, 0.0, 10.0, 0.0],
            vec![10.0, 0.0, 5.0, 0.0, 0.01, 0.0],
        ];
        let out = thin_runs(runs, 0.01, 0.001);
        assert_eq!(out, vec![vec![0.0, 0.0, 10.0, 0.0, 0.01, 0.0]]);
    }

    #[test]
    fn a_curve_keeps_its_shape_within_the_tolerance() {
        // A quarter circle of radius 100 in 0.5 mm pieces, each its own run.
        let n = 314;
        let p = |k: usize| {
            let t = std::f64::consts::FRAC_PI_2 * k as f64 / n as f64;
            [100.0 * t.cos(), 100.0 * t.sin()]
        };
        let runs: Vec<Vec<f64>> = (0..n)
            .map(|k| vec![p(k)[0], p(k)[1], p(k + 1)[0], p(k + 1)[1]])
            .collect();
        let out = thin_runs(runs, 0.01, 0.001);
        assert_eq!(out.len(), 1, "one polyline");
        let pts: Vec<[f64; 2]> = out[0].chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        assert!(pts.len() < n / 3 && pts.len() > 4, "{} points", pts.len());
        // Every original point lies within the tolerance (plus the rounding) of the thinned line.
        for k in 0..=n {
            let q = p(k);
            let d = pts
                .windows(2)
                .map(|w| seg_dist(q, w[0], w[1]))
                .fold(f64::MAX, f64::min);
            assert!(d <= 0.0105, "point {k} is {d} mm off");
        }
    }

    fn seg_dist(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let d = [b[0] - a[0], b[1] - a[1]];
        let t = (((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]))
            .clamp(0.0, 1.0);
        ((q[0] - a[0] - t * d[0]).powi(2) + (q[1] - a[1] - t * d[1]).powi(2)).sqrt()
    }

    #[test]
    fn picks_join_per_reference_and_hidden_lines_go_when_not_drawn() {
        let mut entry = serde_json::json!({
            "visible": [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 2.0, 0.0]],
            "hidden": [[0.0, 1.0, 1.0, 1.0]],
            "picks": [
                { "reference": "a", "points": [0.0, 0.0, 1.0, 0.0] },
                { "reference": "b", "points": [1.0, 0.0, 2.0, 0.0] },
                { "reference": "a", "points": [1.0, 0.0, 1.0, 1.0] }
            ]
        });
        thin(&mut entry, 1.0, false);
        assert_eq!(entry["visible"], serde_json::json!([[0.0, 0.0, 2.0, 0.0]]));
        assert_eq!(entry["hidden"], serde_json::json!([]));
        assert_eq!(
            entry["picks"],
            serde_json::json!([
                { "reference": "a", "points": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0] },
                { "reference": "b", "points": [1.0, 0.0, 2.0, 0.0] }
            ])
        );
    }
}
