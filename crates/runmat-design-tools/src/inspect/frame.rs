//! A body's own frame for the `mass` section: its principal axes (the inertia's eigenvectors)
//! and its extent, so "along the body" means along its long axis, whatever way the part is
//! drawn.

use serde_json::{json, Value};

type M3 = [[f64; 3]; 3];

/// The eigenvalues and unit eigenvectors of symmetric `m` (cyclic Jacobi), smallest value first.
pub(super) fn eigen(m: M3) -> [(f64, [f64; 3]); 3] {
    let mut a = m;
    let mut v: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-15 * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()).max(1e-300) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in &mut a {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in v.iter_mut() {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut out: [(f64, [f64; 3]); 3] = std::array::from_fn(|i| {
        let mut axis = [v[0][i], v[1][i], v[2][i]];
        // A stable sign: the largest component positive.
        let big = (0..3)
            .max_by(|&x, &y| axis[x].abs().total_cmp(&axis[y].abs()))
            .unwrap_or(0);
        if axis[big] < 0.0 {
            axis = axis.map(|c| -c);
        }
        (a[i][i], axis)
    });
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    out
}

/// The principal axes of inertia `inertia` (any unit), smallest moment first: for a long body
/// the first is its long axis.
pub(super) fn principal_axes(inertia: M3, scale: f64) -> Value {
    let round = |v: f64, places: i32| (v * 10f64.powi(places)).round() / 10f64.powi(places);
    Value::Array(
        eigen(inertia)
            .iter()
            .map(|(moment, axis)| json!({ "axis": axis.map(|c| round(c, 4)), "moment_kg_mm2": round(moment * scale, 6) }))
            .collect(),
    )
}

/// The box around `positions` (flat x, y, z in metres), in mm: `{ "min": …, "max": … }`.
pub(super) fn extent_mm<'a>(positions: impl Iterator<Item = &'a [f32]>) -> Value {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in positions {
        for chunk in p.chunks_exact(3) {
            for k in 0..3 {
                lo[k] = lo[k].min(chunk[k] as f64);
                hi[k] = hi[k].max(chunk[k] as f64);
            }
        }
    }
    if !lo[0].is_finite() {
        return Value::Null;
    }
    let mm = |v: f64| (v * 1e4).round() / 10.0;
    json!({ "min": lo.map(mm), "max": hi.map(mm) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_body_has_its_long_axis_first() {
        // A rod along a tilted axis in the x-y plane: its inertia is small along the rod.
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let r = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let d = [1.0, 10.0, 10.5];
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = (0..3).map(|k| r[i][k] * d[k] * r[j][k]).sum();
            }
        }
        let e = eigen(m);
        assert!(
            (e[0].0 - 1.0).abs() < 1e-9
                && (e[1].0 - 10.0).abs() < 1e-9
                && (e[2].0 - 10.5).abs() < 1e-9,
            "{e:?}"
        );
        assert!(
            (e[0].1[0] - c).abs() < 1e-9 && (e[0].1[1] - s).abs() < 1e-9,
            "{e:?}"
        );
    }
}
