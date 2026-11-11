//! Normwise backward errors of computed solutions, in the ∞-norm.
//!
//! A backward error is the smallest relative perturbation of the problem's data for which the
//! computed answer is exact:
//!
//! - linear system `A x = b` (Rigal and Gaches, 1967; Higham, *Accuracy and Stability of
//!   Numerical Algorithms*, 2nd ed., Thm. 7.1):
//!   `η(x) = ‖b − A x‖ / (‖A‖ ‖x‖ + ‖b‖)`;
//! - eigenpair of the pencil `A x = λ B x` (Tisseur, *Backward error and condition of polynomial
//!   eigenvalue problems*, LAA 309, 2000, Thm. 1 with `p = 1`):
//!   `η(x, λ) = ‖A x − λ B x‖ / ((‖A‖ + |λ| ‖B‖) ‖x‖)`.
//!
//! Both formulas hold for any subordinate norm. The ∞-norm is used because `‖A‖_∞`, the largest
//! absolute row sum, is computed exactly in one pass over the stored entries, so the measure
//! needs no norm estimate. The residual and `‖x‖` use the same norm.
//!
//! The measure is invariant to scaling `A`, `B`, `b` or `x`, and a backward-stable method returns
//! `η` of the order of the unit roundoff whatever the conditioning (Higham, §1.5 and Ch. 7). The
//! relative residual `‖A x − λ B x‖ / (‖A x‖ + |λ| ‖B x‖)` lacks that property: storing an
//! exact eigenvector in double precision already leaves it near `u · κ`, which grows with mesh
//! refinement.
//!
//! The evaluation itself has a floor: computing `A x` row by row commits an error of at most
//! `γ_{m}|A||x|` with `m` the largest row length (Higham, §3.5), so `η` below about `m · u`
//! cannot be distinguished from zero.

use crate::csr::CsrRef;
use crate::par;
use crate::scalar::Scalar;

/// The measure an algebraic accuracy tolerance applies to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ResidualMeasure {
    /// Normwise backward error in the ∞-norm ([`eigenpair_backward_error`],
    /// [`linear_backward_error`]).
    #[default]
    BackwardError,
    /// Euclidean residual relative to the size of the terms that form it:
    /// `‖A x − λ B x‖₂ / (‖A x‖₂ + |λ| ‖B x‖₂)` for an eigenpair.
    RelativeResidual,
}

/// `‖r‖_∞ / ((‖A‖_∞ + |λ| ‖B‖_∞) ‖x‖_∞)` from its parts, with `r = A x − λ B x`. A zero vector
/// or a zero pencil gives infinity unless the residual is also zero.
pub fn eigenpair_backward_error(
    residual_inf: f64,
    lambda: f64,
    a_norm_inf: f64,
    b_norm_inf: f64,
    x_norm_inf: f64,
) -> f64 {
    quotient(
        residual_inf,
        (a_norm_inf + lambda.abs() * b_norm_inf) * x_norm_inf,
    )
}

/// `‖b − A x‖_∞ / (‖A‖_∞ ‖x‖_∞ + ‖b‖_∞)` for a square or rectangular `A`.
pub fn linear_backward_error<T: Scalar>(a: CsrRef<'_, T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::zero(); a.nrows()];
    a.spmv(x, &mut ax);
    let mut residual = vec![T::zero(); a.nrows()];
    par::sub_into(b, &ax, &mut residual);
    linear_backward_error_from_parts(
        par::norm_inf(&residual),
        a.norm_inf(),
        par::norm_inf(x),
        par::norm_inf(b),
    )
}

/// `‖r‖_∞ / (‖A‖_∞ ‖x‖_∞ + ‖b‖_∞)` from its parts, for callers that already hold the residual.
pub fn linear_backward_error_from_parts(
    residual_inf: f64,
    a_norm_inf: f64,
    x_norm_inf: f64,
    b_norm_inf: f64,
) -> f64 {
    quotient(residual_inf, a_norm_inf * x_norm_inf + b_norm_inf)
}

fn quotient(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        0.0
    } else if denominator > 0.0 && denominator.is_finite() {
        numerator / denominator
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::{CsrMatrix, TripletBuilder};

    fn tridiagonal(n: usize, diagonal: f64, off: f64) -> CsrMatrix<f64> {
        let mut builder = TripletBuilder::new(n, n);
        for i in 0..n {
            builder.push(i, i, diagonal);
            if i + 1 < n {
                builder.push(i, i + 1, off);
                builder.push(i + 1, i, off);
            }
        }
        builder.build()
    }

    /// Thomas elimination: backward stable for this SPD tridiagonal matrix.
    fn solve_tridiagonal(n: usize, diagonal: f64, off: f64, b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        c[0] = off / diagonal;
        d[0] = b[0] / diagonal;
        for i in 1..n {
            let pivot = diagonal - off * c[i - 1];
            c[i] = off / pivot;
            d[i] = (b[i] - off * d[i - 1]) / pivot;
        }
        let mut x = d;
        for i in (0..n - 1).rev() {
            x[i] -= c[i] * x[i + 1];
        }
        x
    }

    #[test]
    fn linear_backward_error_is_scale_invariant_and_rejects_a_wrong_solution() {
        // A 1-D Laplacian with a uniform load: κ ≈ 4n²/π², and a backward-stable solve leaves a
        // relative residual ‖r‖/‖b‖ far above 1e-8 while the backward error stays near u.
        let n = 60_000;
        let a = tridiagonal(n, 2.0, -1.0);
        let b = vec![1.0; n];
        let x = solve_tridiagonal(n, 2.0, -1.0, &b);
        let eta = linear_backward_error(a.as_ref(), &x, &b);
        let mut residual = b.clone();
        a.as_ref().spmv_scaled(-1.0, &x, 1.0, &mut residual);
        let relative = par::norm2(&residual) / par::norm2(&b);
        assert!(eta < 1.0e-15, "η = {eta}");
        assert!(relative > 1.0e-8, "relative residual {relative}");
        // Scaling the data leaves η unchanged (a power of two scales exactly).
        let scale = 2f64.powi(30);
        let scaled = CsrMatrix::linear_combination(&[(scale, a.as_ref())]).unwrap();
        let b_scaled: Vec<f64> = b.iter().map(|v| v * scale).collect();
        let eta_scaled = linear_backward_error(scaled.as_ref(), &x, &b_scaled);
        assert_eq!(eta_scaled, eta);
        // A wrong solution: the right one plus an oscillating error of 1e-9 of its size, as an
        // unconverged smoother leaves. η sees it at about its own size.
        let amplitude = 1.0e-9 * par::norm_inf(&x);
        let wrong: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 2 == 0 { amplitude } else { -amplitude })
            .collect();
        let eta_wrong = linear_backward_error(a.as_ref(), &wrong, &b);
        assert!(eta_wrong > 1.0e-10, "η of the wrong solution {eta_wrong}");
    }

    #[test]
    fn backward_error_parts_handle_zero_terms() {
        assert_eq!(eigenpair_backward_error(0.0, 1.0, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(
            eigenpair_backward_error(1.0, 1.0, 0.0, 0.0, 1.0),
            f64::INFINITY
        );
        assert_eq!(eigenpair_backward_error(1.0, 2.0, 1.0, 1.0, 1.0), 1.0 / 3.0);
        assert_eq!(linear_backward_error_from_parts(1.0, 2.0, 3.0, 4.0), 0.1);
    }
}
