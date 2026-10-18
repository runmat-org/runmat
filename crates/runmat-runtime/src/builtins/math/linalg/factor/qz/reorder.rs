//! Adjacent two-sided exchanges for a complex triangular QZ pencil.

use faer::c64;

use super::algorithm::{ComplexQz, ComplexQzView};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ClusterOrder {
    /// Logical and region selectors encode selected as zero.
    SelectedFirst,
    /// Numeric cluster indices place the highest index first.
    NumericDescending,
}

impl ClusterOrder {
    pub(super) fn precedes(self, candidate: u64, current: u64) -> bool {
        match self {
            Self::SelectedFirst => candidate < current,
            Self::NumericDescending => candidate > current,
        }
    }

    pub(super) fn pair_label(self, first: u64, second: u64) -> u64 {
        match self {
            Self::SelectedFirst => first.min(second),
            Self::NumericDescending => first.max(second),
        }
    }
}

fn conjugate(z: c64) -> c64 {
    c64::new(z.re, -z.im)
}

fn magnitude(z: c64) -> f64 {
    z.re.hypot(z.im)
}

fn unit(z: c64, norm: f64) -> c64 {
    c64::new(z.re / norm, z.im / norm)
}

fn mix(x: c64, y: c64, first: c64, second: c64) -> c64 {
    let scale = x.re.abs().max(x.im.abs()).max(y.re.abs()).max(y.im.abs());
    if scale == 0.0 {
        return c64::new(0.0, 0.0);
    }
    let value = c64::new(x.re / scale, x.im / scale) * first
        + c64::new(y.re / scale, y.im / scale) * second;
    c64::new(value.re * scale, value.im * scale)
}

fn adjacent_exchange(form: &mut ComplexQzView<'_>, k: usize) -> Result<(), &'static str> {
    let n = form.aa.nrows();
    let a = form.aa[(k, k)];
    let c = form.aa[(k, k + 1)];
    let d = form.aa[(k + 1, k + 1)];
    let b = form.bb[(k, k)];
    let e = form.bb[(k, k + 1)];
    let f = form.bb[(k + 1, k + 1)];
    if magnitude(d) == 0.0 && magnitude(f) == 0.0 {
        return Err("indeterminate generalized eigenvalue cannot be reordered");
    }
    // The first column of the right rotation spans the trailing generalized
    // eigenvector. Scale before cross-products to avoid overflow for large
    // but finite Schur coefficients.
    let scale = [a, b, c, d, e, f]
        .into_iter()
        .fold(0.0_f64, |maximum, value| {
            maximum.max(value.re.abs()).max(value.im.abs())
        });
    if scale == 0.0 {
        return Err("zero generalized pencil has no ordered invariant subspace");
    }
    let scaled = |value: c64| c64::new(value.re / scale, value.im / scale);
    let (a, b, c, d, e, f) = (
        scaled(a),
        scaled(b),
        scaled(c),
        scaled(d),
        scaled(e),
        scaled(f),
    );
    let mut v0 = d * e - f * c;
    let mut v1 = f * a - d * b;
    let norm = magnitude(v0).hypot(magnitude(v1));
    if norm == 0.0 {
        if magnitude(c) != 0.0 || magnitude(e) != 0.0 {
            return Err("generalized Schur blocks are inseparable");
        }
        v0 = c64::new(0.0, 0.0);
        v1 = c64::new(1.0, 0.0);
    } else {
        v0 = unit(v0, norm);
        v1 = unit(v1, norm);
    }
    let w0 = -conjugate(v1);
    let w1 = conjugate(v0);
    let c0 = a * v0 + c * v1;
    let c1 = d * v1;
    let d0 = b * v0 + e * v1;
    let d1 = f * v1;
    let c_norm = magnitude(c0).hypot(magnitude(c1));
    let d_norm = magnitude(d0).hypot(magnitude(d1));
    let (u0, u1) = if c_norm >= d_norm && c_norm > 0.0 {
        (unit(c0, c_norm), unit(c1, c_norm))
    } else if d_norm > 0.0 {
        (unit(d0, d_norm), unit(d1, d_norm))
    } else {
        return Err("generalized Schur exchange has no left invariant direction");
    };
    let t0 = -conjugate(u1);
    let t1 = conjugate(u0);

    // Both unitary transformations act on the whole pair, not merely on the
    // 2x2 diagonal block. The input form is private and is discarded on error.
    for row in 0..n {
        let (x, y) = (form.aa[(row, k)], form.aa[(row, k + 1)]);
        form.aa[(row, k)] = mix(x, y, v0, v1);
        form.aa[(row, k + 1)] = mix(x, y, w0, w1);
        let (x, y) = (form.bb[(row, k)], form.bb[(row, k + 1)]);
        form.bb[(row, k)] = mix(x, y, v0, v1);
        form.bb[(row, k + 1)] = mix(x, y, w0, w1);
        let (x, y) = (form.right[(row, k)], form.right[(row, k + 1)]);
        form.right[(row, k)] = mix(x, y, v0, v1);
        form.right[(row, k + 1)] = mix(x, y, w0, w1);
        let (x, y) = (form.left[(row, k)], form.left[(row, k + 1)]);
        form.left[(row, k)] = mix(x, y, u0, u1);
        form.left[(row, k + 1)] = mix(x, y, t0, t1);
    }
    for col in 0..n {
        let (x, y) = (form.aa[(k, col)], form.aa[(k + 1, col)]);
        form.aa[(k, col)] = mix(x, y, conjugate(u0), conjugate(u1));
        form.aa[(k + 1, col)] = mix(x, y, conjugate(t0), conjugate(t1));
        let (x, y) = (form.bb[(k, col)], form.bb[(k + 1, col)]);
        form.bb[(k, col)] = mix(x, y, conjugate(u0), conjugate(u1));
        form.bb[(k + 1, col)] = mix(x, y, conjugate(t0), conjugate(t1));
    }
    if [&form.aa, &form.bb, &form.left, &form.right]
        .into_iter()
        .any(|matrix| {
            (0..n).any(|j| {
                (0..n).any(|i| {
                    let value = matrix[(i, j)];
                    !value.re.is_finite() || !value.im.is_finite()
                })
            })
        })
    {
        return Err("generalized Schur exchange produced nonfinite factors");
    }
    let residual = magnitude(form.aa[(k + 1, k)]).max(magnitude(form.bb[(k + 1, k)]));
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            let a = form.aa[(i, j)];
            let b = form.bb[(i, j)];
            m.max(a.re.abs())
                .max(a.im.abs())
                .max(b.re.abs())
                .max(b.im.abs())
        });
    if residual > 128.0 * f64::EPSILON * (n as f64).max(1.0) * scale {
        return Err("generalized Schur blocks are too ill-conditioned to exchange");
    }
    form.aa[(k + 1, k)] = c64::new(0.0, 0.0);
    form.bb[(k + 1, k)] = c64::new(0.0, 0.0);
    Ok(())
}

/// Stable order selected by the caller's typed selector. The labels, rather
/// than ratios alpha/beta, remain authoritative across successive swaps.
pub(super) fn reorder_clusters(
    form: &mut ComplexQz,
    labels: &[u64],
    order: ClusterOrder,
) -> Result<(), &'static str> {
    let mut labels = labels.to_vec();
    let mut view = ComplexQzView {
        aa: form.aa.as_mut(),
        bb: form.bb.as_mut(),
        left: form.left.as_mut(),
        right: form.right.as_mut(),
        alpha: form.alpha.as_mut(),
        beta: form.beta.as_mut(),
    };
    reorder_clusters_in_place(&mut view, &mut labels, order)
}

pub(super) fn reorder_clusters_in_place(
    form: &mut ComplexQzView<'_>,
    labels: &mut [u64],
    order: ClusterOrder,
) -> Result<(), &'static str> {
    let n = form.aa.nrows();
    if labels.len() != n {
        return Err("generalized Schur cluster length does not match pencil order");
    }
    for destination in 0..n {
        let mut best = destination;
        for candidate in (destination + 1)..n {
            if order.precedes(labels[candidate], labels[best]) {
                best = candidate;
            }
        }
        for position in (destination..best).rev() {
            adjacent_exchange(form, position)?;
            labels.swap(position, position + 1);
        }
    }
    Ok(())
}
