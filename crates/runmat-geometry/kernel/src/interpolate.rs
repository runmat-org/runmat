//! Cubic B-spline interpolation through points: the curve a sketch spline draws.
//!
//! [`interpolate`] fits a clamped cubic through every point in order (chord-length
//! parameters, knots by averaging; The NURBS Book, algorithm A9.1). The curve passes through
//! each point exactly and starts and ends at the first and last points. A closed spline repeats
//! its first point at the end, so the curve closes on itself; the seam is where it starts.

use crate::op::SegmentKind;

/// The clamped cubic through `points` (at least two, consecutive points distinct), or `None`
/// when there are fewer than two points or two consecutive points coincide. Two points give a
/// straight degree-1 spline and three a degree-2 one.
pub fn interpolate(points: &[[f64; 2]], closed: bool) -> Option<SegmentKind> {
    let mut q: Vec<[f64; 2]> = points.to_vec();
    if closed && q.len() > 2 {
        q.push(q[0]);
    }
    let n = q.len();
    if n < 2 {
        return None;
    }
    let degree = (n - 1).min(3);
    // Chord-length parameters on [0, 1].
    let mut u = vec![0.0; n];
    let mut total = 0.0;
    for i in 1..n {
        let d = (q[i][0] - q[i - 1][0]).hypot(q[i][1] - q[i - 1][1]);
        if d <= 1e-12 {
            return None;
        }
        total += d;
        u[i] = total;
    }
    for x in u.iter_mut() {
        *x /= total;
    }
    u[n - 1] = 1.0;
    // Knots: clamped ends, interior by averaging `degree` parameters.
    let mut flat = vec![0.0; degree + 1];
    for j in 1..n - degree {
        flat.push(u[j..j + degree].iter().sum::<f64>() / degree as f64);
    }
    flat.extend(std::iter::repeat_n(1.0, degree + 1));
    // The collocation system N(u_k) P = Q.
    let mut a = vec![vec![0.0; n]; n];
    for (k, row) in a.iter_mut().enumerate() {
        let span = find_span(n, degree, u[k], &flat);
        let basis = basis_functions(span, u[k], degree, &flat);
        for (j, b) in basis.iter().enumerate() {
            row[span - degree + j] = *b;
        }
    }
    let poles = solve(a, &q)?;
    let (knots, multiplicities) = distinct(&flat);
    Some(SegmentKind::Spline {
        degree: degree as u32,
        poles,
        weights: Vec::new(),
        knots,
        multiplicities,
    })
}

fn find_span(n: usize, p: usize, u: f64, knots: &[f64]) -> usize {
    if u >= knots[n] {
        return n - 1;
    }
    let mut span = p;
    while span + 1 < n && knots[span + 1] <= u {
        span += 1;
    }
    span
}

/// The `p + 1` non-zero basis functions at `u` in `span` (A2.2).
fn basis_functions(span: usize, u: f64, p: usize, knots: &[f64]) -> Vec<f64> {
    let mut n = vec![0.0; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let t = if denom.abs() > 1e-300 { n[r] / denom } else { 0.0 };
            n[r] = saved + right[r + 1] * t;
            saved = left[j - r] * t;
        }
        n[j] = saved;
    }
    n
}

/// Gaussian elimination with partial pivoting, two right-hand sides.
fn solve(mut a: Vec<Vec<f64>>, b: &[[f64; 2]]) -> Option<Vec<[f64; 2]>> {
    let n = b.len();
    let mut x: Vec<[f64; 2]> = b.to_vec();
    for c in 0..n {
        let pivot = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[pivot][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, pivot);
        x.swap(c, pivot);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            if f == 0.0 {
                continue;
            }
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            x[r][0] -= f * x[c][0];
            x[r][1] -= f * x[c][1];
        }
    }
    for c in (0..n).rev() {
        for k in c + 1..n {
            x[c][0] -= a[c][k] * x[k][0];
            x[c][1] -= a[c][k] * x[k][1];
        }
        x[c][0] /= a[c][c];
        x[c][1] /= a[c][c];
    }
    Some(x)
}

fn distinct(flat: &[f64]) -> (Vec<f64>, Vec<u32>) {
    let mut knots: Vec<f64> = Vec::new();
    let mut mults: Vec<u32> = Vec::new();
    for &k in flat {
        match knots.last() {
            Some(&last) if (k - last).abs() <= 1e-15 => *mults.last_mut().unwrap() += 1,
            _ => {
                knots.push(k);
                mults.push(1);
            }
        }
    }
    (knots, mults)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::SketchEntityId;
    use crate::op::ProfileSegment;

    fn segment(points: &[[f64; 2]], closed: bool) -> ProfileSegment {
        let kind = interpolate(points, closed).expect("fits");
        ProfileSegment {
            curve: SketchEntityId::new("s"),
            kind,
            start: points[0],
            end: if closed { points[0] } else { *points.last().unwrap() },
        }
    }

    #[test]
    fn the_curve_passes_through_every_point() {
        let pts = [[0.0, 0.0], [1.0, 0.6], [2.5, 0.8], [4.0, 0.2], [5.0, -0.3], [6.0, 0.0]];
        let s = segment(&pts, false);
        let samples = s.polyline(2000);
        for p in pts {
            let d = samples
                .iter()
                .map(|q| (q[0] - p[0]).hypot(q[1] - p[1]))
                .fold(f64::MAX, f64::min);
            assert!(d < 2e-3, "{p:?} misses by {d}");
        }
        assert_eq!(s.point_at(0.0), pts[0]);
        let end = s.point_at(1.0);
        assert!((end[0] - 6.0).abs() < 1e-12 && end[1].abs() < 1e-12);
    }

    #[test]
    fn points_on_a_circle_give_a_round_curve_and_closed_splines_close() {
        let pts: Vec<[f64; 2]> = (0..12)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 12.0;
                [a.cos(), a.sin()]
            })
            .collect();
        let s = segment(&pts, true);
        assert!(s.is_closed_curve());
        for p in s.polyline(400) {
            let r = p[0].hypot(p[1]);
            assert!((r - 1.0).abs() < 0.01, "radius {r}");
        }
    }

    #[test]
    fn short_inputs_lower_the_degree_and_bad_inputs_are_refused() {
        assert!(matches!(interpolate(&[[0.0, 0.0], [1.0, 1.0]], false), Some(SegmentKind::Spline { degree: 1, .. })));
        assert!(matches!(interpolate(&[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], false), Some(SegmentKind::Spline { degree: 2, .. })));
        assert!(interpolate(&[[0.0, 0.0]], false).is_none());
        assert!(interpolate(&[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], false).is_none());
    }
}
