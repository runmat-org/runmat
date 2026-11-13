//! Normalized rank-one inverse updates, validated before any matrix mutation.
use super::*;

pub(super) async fn update(
    h: &mut [f64],
    s: &[f64],
    y: &[f64],
    kind: Update,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let sy = inner(s, y, cooperation).await?;
    if !sy.is_finite() || sy <= 0. || (kind == Update::Bfgs && sy <= 1e-14) {
        return Ok(());
    }
    let mut v = mat_vec(h, y, s.len(), cooperation).await?;
    let yhy = inner(y, &v, cooperation).await?;
    if !yhy.is_finite() || yhy <= 0. {
        return Ok(());
    }
    let root_sy = sy.sqrt();
    let root_yhy = yhy.sqrt();
    let mut u = copy(s, cooperation).await?;
    for (u, v) in u.chunks_mut(1024).zip(v.chunks_mut(1024)) {
        for (u, v) in u.iter_mut().zip(&mut *v) {
            *u /= root_sy;
            *v /= root_yhy;
        }
        cooperation.completed(u.len() as u64).await?;
    }
    if u.iter().chain(&v).any(|value| !value.is_finite()) {
        return Ok(());
    }
    let w = if kind == Update::Bfgs {
        // BFGS = H + uu' + ww' - vv', where
        // u=s/sqrt(s'y), v=Hy/sqrt(y'Hy),
        // w=sqrt(y'Hy)/sqrt(s'y)*u-v. This avoids forming s_i*s_j
        // before division and avoids the overflowing rho/coeff products.
        let scale = root_yhy / root_sy;
        if !scale.is_finite() {
            return Ok(());
        }
        let mut w = copy(&u, cooperation).await?;
        for (w, v) in w.chunks_mut(1024).zip(v.chunks(1024)) {
            for (w, v) in w.iter_mut().zip(v) {
                *w = scale.mul_add(*w, -v);
            }
            cooperation.completed(w.len() as u64).await?;
        }
        if w.iter().any(|value| !value.is_finite()) {
            return Ok(());
        }
        Some(w)
    } else {
        None
    };
    let candidate = |row: usize, col: usize| {
        let correction = u[row] * u[col] - v[row] * v[col];
        let correction = correction + w.as_ref().map_or(0., |w| w[row] * w[col]);
        h[row * s.len() + col] + correction
    };
    // A nonrepresentable secant update is skipped, not a partially corrupted
    // model. Validate first, then recompute the same finite candidates in place;
    // no duplicate n-by-n matrix is required.
    for row in 0..s.len() {
        for col in 0..s.len() {
            if !candidate(row, col).is_finite() {
                return Ok(());
            }
        }
        cooperation.completed(s.len() as u64).await?;
    }
    for row in 0..s.len() {
        for col in 0..s.len() {
            let correction =
                u[row] * u[col] - v[row] * v[col] + w.as_ref().map_or(0., |w| w[row] * w[col]);
            h[row * s.len() + col] += correction;
        }
        cooperation.completed(s.len() as u64).await?;
    }
    Ok(())
}
