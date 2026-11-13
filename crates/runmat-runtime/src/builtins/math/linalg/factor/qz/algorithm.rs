//! Portable generalized Schur reduction backed by faer's low-level QZ kernels.
//!
//! The input pair is owned by this routine.  The QR reduction of B and the
//! generalized Hessenberg reduction are required preconditions of faer's QZ
//! kernel; calling that kernel on an arbitrary pair is not a QZ factorization.

use faer::col::ColMut;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{self, GevdParams};
use faer::mat::{AsMatMut, MatMut};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{c64, Col, Mat, Par, Spec};
use std::mem::MaybeUninit;

pub(super) struct ComplexQz {
    pub aa: Mat<c64>,
    pub bb: Mat<c64>,
    /// Faer's accumulated left unitary, with A = left * AA * right^H.
    pub left: Mat<c64>,
    pub right: Mat<c64>,
    pub alpha: Col<c64>,
    pub beta: Col<c64>,
}

/// The same QZ factor authority over caller-owned, exact scratch matrices.
/// Both the ordinary Mat wrapper and admitted Riccati storage use this view;
/// there is one low-level QZ algorithm, not two solver implementations.
pub(super) struct ComplexQzView<'a> {
    pub aa: MatMut<'a, c64>,
    pub bb: MatMut<'a, c64>,
    pub left: MatMut<'a, c64>,
    pub right: MatMut<'a, c64>,
    pub alpha: ColMut<'a, c64>,
    pub beta: ColMut<'a, c64>,
}

/// Same reduction with an optional accumulated left unitary. Full EVD does
/// not allocate an unused n-by-n Q when only right vectors were requested.
pub(in crate::builtins::math::linalg) struct ComplexQzOptionalView<'a> {
    pub aa: MatMut<'a, c64>,
    pub bb: MatMut<'a, c64>,
    pub left: Option<MatMut<'a, c64>>,
    pub right: MatMut<'a, c64>,
    pub alpha: ColMut<'a, c64>,
    pub beta: ColMut<'a, c64>,
}

pub(super) struct RealQz {
    pub aa: Mat<f64>,
    pub bb: Mat<f64>,
    /// Faer's accumulated left orthogonal, with A = left * AA * right^T.
    pub left: Mat<f64>,
    pub right: Mat<f64>,
    pub alpha_re: Col<f64>,
    pub alpha_im: Col<f64>,
    pub beta: Col<f64>,
}

pub(super) fn real_qz(mut aa: Mat<f64>, mut bb: Mat<f64>) -> Result<RealQz, &'static str> {
    let n = aa.nrows();
    if aa.ncols() != n || bb.nrows() != n || bb.ncols() != n {
        return Err("generalized Schur inputs must be equally sized square matrices");
    }
    if !(0..n).all(|j| (0..n).all(|i| aa[(i, j)].is_finite() && bb[(i, j)].is_finite())) {
        return Err("generalized Schur inputs must be finite");
    }
    let mut left = Mat::<f64>::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut right = left.clone();
    if n == 0 {
        return Ok(RealQz {
            aa,
            bb,
            left,
            right,
            alpha_re: Col::zeros(0),
            alpha_im: Col::zeros(0),
            beta: Col::zeros(0),
        });
    }
    let mut alphar = Col::<f64>::zeros(n);
    let mut alphai = Col::<f64>::zeros(n);
    let mut beta = Col::<f64>::zeros(n);
    let params: GevdParams = faer::auto!(f64);
    let scratch = gevd::gevd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        ComputeEigenvectors::Yes,
        Par::Seq,
        Spec::new(params),
    );
    let mut memory =
        MemBuffer::try_new(scratch).map_err(|_| "generalized Schur workspace allocation failed")?;
    let stack = MemStack::new(&mut memory);
    let block_size = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
    {
        let (mut householder, stack) =
            unsafe { faer::linalg::temp_mat_uninit::<f64, _, _>(block_size, n, stack) };
        let mut householder = householder.as_mat_mut();
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            bb.as_mut(),
            householder.rb_mut(),
            Par::Seq,
            stack,
            Default::default(),
        );
        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            bb.as_ref(), householder.rb(), faer::Conj::Yes,
            aa.as_mut(), Par::Seq, stack,
        );
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bb.as_ref(),
            householder.rb(),
            faer::Conj::No,
            left.as_mut(),
            Par::Seq,
            stack,
        );
        for j in 0..n {
            for i in (j + 1)..n {
                bb[(i, j)] = 0.0;
            }
        }
    }
    gevd::gen_hessenberg::generalized_hessenberg(
        aa.as_mut(),
        bb.as_mut(),
        Some(left.as_mut()),
        Some(right.as_mut()),
        Par::Seq,
        stack,
        params.hessenberg,
    );
    gevd::qz_real::hessenberg_to_qz(
        aa.as_mut(),
        bb.as_mut(),
        Some(left.as_mut()),
        Some(right.as_mut()),
        alphar.as_mut(),
        alphai.as_mut(),
        beta.as_mut(),
        ComputeEigenvectors::Yes,
        Par::Seq,
        params.schur,
        stack,
    );
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            m.max(aa[(i, j)].abs()).max(bb[(i, j)].abs())
        });
    let tolerance = 128.0 * f64::EPSILON * (n as f64).max(1.0) * scale;
    if (0..n).any(|i| !alphar[i].is_finite() || !alphai[i].is_finite() || !beta[i].is_finite()) {
        return Err("generalized Schur iteration produced nonfinite eigenvalue metadata");
    }
    for j in 0..n {
        for i in 0..n {
            if !aa[(i, j)].is_finite() || !bb[(i, j)].is_finite() {
                return Err("generalized Schur iteration did not produce finite factors");
            }
            if (i > j + 1 && aa[(i, j)].abs() > tolerance)
                || (i > j && bb[(i, j)].abs() > tolerance)
            {
                return Err(
                    "generalized Schur iteration did not converge to real quasi-triangular form",
                );
            }
        }
    }
    Ok(RealQz {
        aa,
        bb,
        left,
        right,
        alpha_re: alphar,
        alpha_im: alphai,
        beta,
    })
}

/// QZ of a finite, square complex pair.  The caller validates dimensions,
/// floating class, and resource authority before allocating either matrix.
pub(super) fn complex_qz(mut aa: Mat<c64>, mut bb: Mat<c64>) -> Result<ComplexQz, &'static str> {
    let n = aa.nrows();
    if aa.ncols() != n || bb.nrows() != n || bb.ncols() != n {
        return Err("generalized Schur inputs must be equally sized square matrices");
    }
    if !(0..n).all(|j| {
        (0..n).all(|i| {
            let a = aa[(i, j)];
            let b = bb[(i, j)];
            a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite()
        })
    }) {
        return Err("generalized Schur inputs must be finite");
    }

    let mut left = Mat::<c64>::from_fn(n, n, |i, j| {
        if i == j {
            c64::new(1.0, 0.0)
        } else {
            c64::new(0.0, 0.0)
        }
    });
    let mut right = left.clone();
    let mut alpha = Col::<c64>::zeros(n);
    let mut beta = Col::<c64>::zeros(n);
    if n == 0 {
        return Ok(ComplexQz {
            aa,
            bb,
            left,
            right,
            alpha,
            beta,
        });
    }

    let scratch = complex_qz_scratch(n);
    let mut memory =
        MemBuffer::try_new(scratch).map_err(|_| "generalized Schur workspace allocation failed")?;
    let mut view = ComplexQzView {
        aa: aa.as_mut(),
        bb: bb.as_mut(),
        left: left.as_mut(),
        right: right.as_mut(),
        alpha: alpha.as_mut(),
        beta: beta.as_mut(),
    };
    complex_qz_in_place(&mut view, &mut memory)?;
    Ok(ComplexQz {
        aa,
        bb,
        left,
        right,
        alpha,
        beta,
    })
}

pub(in crate::builtins::math::linalg) fn complex_qz_scratch(n: usize) -> faer::dyn_stack::StackReq {
    let params: GevdParams = faer::auto!(c64);
    gevd::gevd_scratch::<c64>(
        n,
        ComputeEigenvectors::Yes,
        ComputeEigenvectors::Yes,
        Par::Seq,
        Spec::new(params),
    )
}

pub(super) fn complex_qz_in_place(
    form: &mut ComplexQzView<'_>,
    memory: &mut [MaybeUninit<u8>],
) -> Result<(), &'static str> {
    complex_qz_optional_in_place(
        &mut ComplexQzOptionalView {
            aa: form.aa.rb_mut(),
            bb: form.bb.rb_mut(),
            left: Some(form.left.rb_mut()),
            right: form.right.rb_mut(),
            alpha: form.alpha.rb_mut(),
            beta: form.beta.rb_mut(),
        },
        memory,
    )
}

pub(in crate::builtins::math::linalg) fn complex_qz_optional_in_place(
    form: &mut ComplexQzOptionalView<'_>,
    memory: &mut [MaybeUninit<u8>],
) -> Result<(), &'static str> {
    let n = form.aa.nrows();
    if form.aa.ncols() != n
        || form.bb.nrows() != n
        || form.bb.ncols() != n
        || form
            .left
            .as_ref()
            .is_some_and(|left| left.nrows() != n || left.ncols() != n)
        || form.right.nrows() != n
        || form.right.ncols() != n
        || form.alpha.nrows() != n
        || form.beta.nrows() != n
    {
        return Err("generalized Schur factor dimensions are invalid");
    }
    let (aa, bb, left, right, alpha, beta) = (
        &mut form.aa,
        &mut form.bb,
        &mut form.left,
        &mut form.right,
        &mut form.alpha,
        &mut form.beta,
    );
    if !(0..n).all(|j| {
        (0..n).all(|i| {
            let a = aa[(i, j)];
            let b = bb[(i, j)];
            a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite()
        })
    }) {
        return Err("generalized Schur inputs must be finite");
    }
    for col in 0..n {
        for row in 0..n {
            let diagonal = if row == col {
                c64::new(1.0, 0.0)
            } else {
                c64::new(0.0, 0.0)
            };
            if let Some(left) = left.as_mut() {
                left[(row, col)] = diagonal;
            }
            right[(row, col)] = diagonal;
        }
    }
    if n == 0 {
        return Ok(());
    }
    let params: GevdParams = faer::auto!(c64);
    let stack = MemStack::new(memory);
    let block_size = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<c64>(n, n);
    {
        let (mut householder, stack) =
            unsafe { faer::linalg::temp_mat_uninit::<c64, _, _>(block_size, n, stack) };
        let mut householder = householder.as_mat_mut();
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            bb.as_mut(),
            householder.rb_mut(),
            Par::Seq,
            stack,
            Default::default(),
        );
        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            bb.as_ref(), householder.rb(), faer::Conj::Yes,
            aa.as_mut(), Par::Seq, stack,
        );
        if let Some(left) = left.as_mut() {
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            bb.as_ref(),
            householder.rb(),
            faer::Conj::No,
            left.as_mut(),
            Par::Seq,
            stack,
        );
        }
        for j in 0..n {
            for i in (j + 1)..n {
                bb[(i, j)] = c64::new(0.0, 0.0);
            }
        }
    }
    gevd::gen_hessenberg::generalized_hessenberg(
        aa.as_mut(),
        bb.as_mut(),
        left.as_mut().map(|left| left.rb_mut()),
        Some(right.as_mut()),
        Par::Seq,
        stack,
        params.hessenberg,
    );
    gevd::qz_cplx::hessenberg_to_qz(
        aa.as_mut(),
        bb.as_mut(),
        left.as_mut().map(|left| left.rb_mut()),
        Some(right.as_mut()),
        alpha.as_mut(),
        beta.as_mut(),
        ComputeEigenvectors::Yes,
        Par::Seq,
        params.schur,
        stack,
    );

    // The low-level QZ entry point has no convergence result.  A failed
    // deflation must not be published as a generalized Schur form.
    let scale = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .fold(0.0_f64, |m, (i, j)| {
            m.max(aa[(i, j)].re.abs())
                .max(aa[(i, j)].im.abs())
                .max(bb[(i, j)].re.abs())
                .max(bb[(i, j)].im.abs())
        });
    let tolerance = 128.0 * f64::EPSILON * (n as f64).max(1.0) * scale;
    if (0..n).any(|i| {
        let a = alpha[i];
        let b = beta[i];
        !a.re.is_finite() || !a.im.is_finite() || !b.re.is_finite() || !b.im.is_finite()
    }) {
        return Err("generalized Schur iteration produced nonfinite eigenvalue metadata");
    }
    for j in 0..n {
        for i in 0..n {
            let a = aa[(i, j)];
            let b = bb[(i, j)];
            if !a.re.is_finite() || !a.im.is_finite() || !b.re.is_finite() || !b.im.is_finite() {
                return Err("generalized Schur iteration did not produce finite factors");
            }
            if i > j && (a.re.hypot(a.im) > tolerance || b.re.hypot(b.im) > tolerance) {
                return Err("generalized Schur iteration did not converge to triangular form");
            }
        }
    }
    Ok(())
}
