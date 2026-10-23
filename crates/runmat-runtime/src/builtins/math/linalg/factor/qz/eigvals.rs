//! Generalized eigenvalues in the retained Schur block order.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{self, GevdParams};
use faer::{c64, Col, Mat, Par};

fn magnitude(value: c64) -> f64 {
    value.re.hypot(value.im)
}

pub(super) fn ratio(alpha: c64, beta: c64) -> (f64, f64) {
    if beta.re == 0.0 && beta.im == 0.0 {
        return if alpha.re == 0.0 && alpha.im == 0.0 {
            (f64::NAN, f64::NAN)
        } else {
            (f64::INFINITY, 0.0)
        };
    }
    crate::builtins::math::linalg::ops::complex_division::divide_finite_components(
        alpha.re, alpha.im, beta.re, beta.im,
    )
    .unwrap_or_else(|| {
        let value = num_complex::Complex64::new(alpha.re, alpha.im)
            / num_complex::Complex64::new(beta.re, beta.im);
        (value.re, value.im)
    })
}

pub(super) fn complex_schur_values(
    aa: &Mat<c64>,
    bb: &Mat<c64>,
) -> Result<Vec<(f64, f64)>, &'static str> {
    let n = aa.nrows();
    if aa.ncols() != n || bb.nrows() != n || bb.ncols() != n {
        return Err("two-input ordeig requires same-order square Schur factors");
    }
    if (0..n).any(|j| {
        (0..n).any(|i| {
            let a = aa[(i, j)];
            let b = bb[(i, j)];
            !a.re.is_finite() || !a.im.is_finite() || !b.re.is_finite() || !b.im.is_finite()
        })
    }) {
        return Err("two-input ordeig requires finite Schur factors");
    }
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            let a = aa[(i, j)];
            let b = bb[(i, j)];
            m.max(a.re.abs())
                .max(a.im.abs())
                .max(b.re.abs())
                .max(b.im.abs())
        });
    let tolerance = 128.0 * f64::EPSILON * (n as f64).max(1.0) * scale;
    for j in 0..n {
        for i in (j + 1)..n {
            if magnitude(aa[(i, j)]) > tolerance || magnitude(bb[(i, j)]) > tolerance {
                return Err("two-input ordeig requires an upper triangular generalized Schur pair");
            }
        }
    }
    Ok((0..n).map(|i| ratio(aa[(i, i)], bb[(i, i)])).collect())
}

pub(super) fn real_schur_values(
    aa: &Mat<f64>,
    bb: &Mat<f64>,
) -> Result<Vec<(f64, f64)>, &'static str> {
    Ok(real_schur_homogeneous(aa, bb)?
        .into_iter()
        .map(|(alpha, beta)| ratio(alpha, beta))
        .collect())
}

pub(super) fn real_schur_homogeneous(
    aa: &Mat<f64>,
    bb: &Mat<f64>,
) -> Result<Vec<(c64, c64)>, &'static str> {
    let n = aa.nrows();
    if aa.ncols() != n || bb.nrows() != n || bb.ncols() != n {
        return Err("two-input ordeig requires same-order square Schur factors");
    }
    if (0..n).any(|j| (0..n).any(|i| !aa[(i, j)].is_finite() || !bb[(i, j)].is_finite())) {
        return Err("two-input ordeig requires finite Schur factors");
    }
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            m.max(aa[(i, j)].abs()).max(bb[(i, j)].abs())
        });
    let tolerance = 128.0 * f64::EPSILON * (n as f64).max(1.0) * scale;
    for j in 0..n {
        for i in 0..n {
            if (i > j + 1 && aa[(i, j)].abs() > tolerance)
                || (i > j && bb[(i, j)].abs() > tolerance)
            {
                return Err("two-input ordeig requires a real generalized Schur pair");
            }
        }
    }
    let mut values = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && aa[(i + 1, i)] != 0.0 {
            if i + 2 < n && aa[(i + 2, i + 1)] != 0.0 {
                return Err("overlapping real generalized Schur blocks");
            }
            let mut a = Mat::<f64>::from_fn(2, 2, |r, c| aa[(i + r, i + c)]);
            let mut b = Mat::<f64>::from_fn(2, 2, |r, c| bb[(i + r, i + c)]);
            let mut alpha_re = Col::<f64>::zeros(2);
            let mut alpha_im = Col::<f64>::zeros(2);
            let mut beta = Col::<f64>::zeros(2);
            let params: GevdParams = faer::auto!(f64);
            let scratch = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(2, Par::Seq, params.schur);
            let mut memory =
                MemBuffer::try_new(scratch).map_err(|_| "ordeig workspace allocation failed")?;
            gevd::qz_real::hessenberg_to_qz(
                a.as_mut(),
                b.as_mut(),
                None,
                None,
                alpha_re.as_mut(),
                alpha_im.as_mut(),
                beta.as_mut(),
                ComputeEigenvectors::No,
                Par::Seq,
                params.schur,
                MemStack::new(&mut memory),
            );
            for k in 0..2 {
                values.push((c64::new(alpha_re[k], alpha_im[k]), c64::new(beta[k], 0.0)));
            }
            i += 2;
        } else {
            values.push((c64::new(aa[(i, i)], 0.0), c64::new(bb[(i, i)], 0.0)));
            i += 1;
        }
    }
    Ok(values)
}
