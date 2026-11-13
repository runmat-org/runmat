//! Tall-skinny block kernels with fixed row blocking (deterministic under threads).

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};

use crate::par;

const ROW_BLOCK: usize = 8192;

/// `Vᵀ W` for column-major `V` (`n × k`) and `W` (`n × p`); returns `k × p` column-major.
pub(crate) fn gemm_tn(v: &[f64], k: usize, w: &[f64], p: usize, n: usize) -> Vec<f64> {
    if k == 0 || p == 0 {
        return vec![0.0; k * p];
    }
    let blocks = n.div_ceil(ROW_BLOCK).max(1);
    let partials: Vec<Vec<f64>> = par::map_range(blocks, 1, |block| {
        let start = block * ROW_BLOCK;
        let len = (n - start).min(ROW_BLOCK);
        let mut out = vec![0.0; k * p];
        let vb = MatRef::from_column_major_slice(v, n, k).subrows(start, len);
        let wb = MatRef::from_column_major_slice(w, n, p).subrows(start, len);
        matmul(
            MatMut::from_column_major_slice_mut(&mut out, k, p),
            Accum::Replace,
            vb.transpose(),
            wb,
            1.0,
            Par::Seq,
        );
        out
    });
    let mut total = vec![0.0; k * p];
    for partial in partials {
        for (t, v) in total.iter_mut().zip(partial) {
            *t += v;
        }
    }
    total
}

/// `W ← β W + α V H` for `V` (`n × k`), `H` (`k × p`), `W` (`n × p`), all column-major.
pub(crate) struct ProductScale {
    pub alpha: f64,
    pub beta: f64,
}

pub(crate) fn gemm_nn(
    w: &mut [f64],
    p: usize,
    scale: ProductScale,
    v: &[f64],
    k: usize,
    h: &[f64],
    n: usize,
) {
    let ProductScale { alpha, beta } = scale;
    if p == 0 {
        return;
    }
    if k == 0 {
        if beta != 1.0 {
            w.iter_mut().for_each(|x| *x *= beta);
        }
        return;
    }
    let blocks = n.div_ceil(ROW_BLOCK).max(1);
    // Compute row blocks of the product independently, then write back.
    let products: Vec<Vec<f64>> = par::map_range(blocks, 1, |block| {
        let start = block * ROW_BLOCK;
        let len = (n - start).min(ROW_BLOCK);
        let mut out = vec![0.0; len * p];
        let vb = MatRef::from_column_major_slice(v, n, k).subrows(start, len);
        matmul(
            MatMut::from_column_major_slice_mut(&mut out, len, p),
            Accum::Replace,
            vb,
            MatRef::from_column_major_slice(h, k, p),
            1.0,
            Par::Seq,
        );
        out
    });
    for (block, out) in products.into_iter().enumerate() {
        let start = block * ROW_BLOCK;
        let len = (n - start).min(ROW_BLOCK);
        for c in 0..p {
            for i in 0..len {
                let target = &mut w[c * n + start + i];
                *target = beta * *target + alpha * out[c * len + i];
            }
        }
    }
}

/// Applies a sparse matrix to each of `p` columns.
pub(crate) fn spmm(a: crate::csr::CsrRef<'_, f64>, x: &[f64], y: &mut [f64], p: usize) {
    let n = a.nrows();
    for c in 0..p {
        a.spmv(&x[c * n..(c + 1) * n], &mut y[c * n..(c + 1) * n]);
    }
}
