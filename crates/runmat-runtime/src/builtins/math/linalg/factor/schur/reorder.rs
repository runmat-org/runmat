//! Adjacent Schur-block exchange without recomputing the source matrix.
//!
//! For `[[A,C],[0,B]]`, the invariant subspace belonging to `B` is spanned by
//! `[[X],[I]]`, where `A X - X B = -C`. A local QR builds the orthogonal/unitary
//! similarity whose leading columns span this subspace. Failure to separate
//! nearby blocks is reported instead of publishing a non-Schur factorization.

use nalgebra::{ComplexField, DMatrix, DVector};
use num_complex::Complex64;

pub(crate) fn exchange_real(
    vectors: &mut DMatrix<f64>,
    triangular: &mut DMatrix<f64>,
    start: usize,
    left: usize,
    right: usize,
) -> Result<(), &'static str> {
    exchange(vectors, triangular, start, left, right)
}

pub(crate) fn exchange_complex(
    vectors: &mut DMatrix<Complex64>,
    triangular: &mut DMatrix<Complex64>,
    start: usize,
    left: usize,
    right: usize,
) -> Result<(), &'static str> {
    exchange(vectors, triangular, start, left, right)
}

fn exchange<T>(
    vectors: &mut DMatrix<T>,
    triangular: &mut DMatrix<T>,
    start: usize,
    left: usize,
    right: usize,
) -> Result<(), &'static str>
where
    T: ComplexField<RealField = f64> + Copy,
{
    let width = left + right;
    let local = triangular.view((start, start), (width, width)).into_owned();
    // A uniform scale leaves the Sylvester solution and invariant subspace
    // unchanged, but avoids overflowing finite A/B/C coefficients.
    let magnitude = local
        .iter()
        .map(|value| value.modulus())
        .fold(0.0_f64, f64::max);
    let divisor = T::from_real(if magnitude == 0.0 { 1.0 } else { magnitude });
    let scaled = local.map(|value| value / divisor);
    let a = scaled.view((0, 0), (left, left)).into_owned();
    let b = scaled.view((left, left), (right, right)).into_owned();
    let c = scaled.view((0, left), (left, right)).into_owned();
    let unknowns = left * right;
    let mut sylvester = DMatrix::<T>::zeros(unknowns, unknowns);
    let mut rhs = DVector::<T>::zeros(unknowns);
    for column in 0..right {
        for row in 0..left {
            let equation = row + column * left;
            rhs[equation] = -c[(row, column)];
            for k in 0..left {
                sylvester[(equation, k + column * left)] += a[(row, k)];
            }
            for k in 0..right {
                sylvester[(equation, row + k * left)] -= b[(k, column)];
            }
        }
    }
    let solution = sylvester
        .lu()
        .solve(&rhs)
        .or_else(|| {
            // A block-diagonal pair can be permuted even when its eigenvalues
            // coincide and the Sylvester operator is singular.
            c.iter()
                .all(|value| *value == T::zero())
                .then(|| DVector::<T>::zeros(unknowns))
        })
        .ok_or("Schur blocks cannot be separated")?;
    let x = DMatrix::from_column_slice(left, right, solution.as_slice());
    let residual = (&a * &x - &x * &b + &c).norm();
    let scale = (a.norm() + b.norm()) * x.norm() + c.norm();
    if !residual.is_finite()
        || !scale.is_finite()
        || residual > 1e-10 * scale.max(f64::MIN_POSITIVE)
    {
        return Err("Schur block exchange is ill-conditioned");
    }

    let mut basis = DMatrix::<T>::zeros(width, width);
    basis.view_mut((0, 0), (left, right)).copy_from(&x);
    for i in 0..right {
        basis[(left + i, i)] = T::from_real(1.0);
    }
    for i in 0..left {
        basis[(i, right + i)] = T::from_real(1.0);
    }
    if !basis.iter().all(|value| value.is_finite()) {
        return Err("Schur block exchange is ill-conditioned");
    }
    let transform = basis.qr().q();
    let transformed = transform.adjoint() * &scaled * &transform;
    let lower = transformed.view((right, 0), (left, right)).norm();
    let local_norm = scaled.norm();
    if !lower.is_finite()
        || !local_norm.is_finite()
        || lower > 1e-10 * local_norm.max(f64::MIN_POSITIVE)
    {
        return Err("Schur block exchange lost its invariant subspace");
    }

    let updated_vectors = vectors.columns(start, width).into_owned() * &transform;
    let updated_rows = transform.adjoint() * triangular.rows(start, width).into_owned();
    vectors
        .columns_mut(start, width)
        .copy_from(&updated_vectors);
    triangular.rows_mut(start, width).copy_from(&updated_rows);
    let updated_columns = triangular.columns(start, width).into_owned() * &transform;
    triangular
        .columns_mut(start, width)
        .copy_from(&updated_columns);
    // The exact invariant-subspace result has a zero lower-left block. Remove
    // only the certified floating residual introduced by the local QR solve.
    triangular
        .view_mut((start + right, start), (left, right))
        .fill(T::zero());
    if !vectors.iter().all(|value| value.is_finite())
        || !triangular.iter().all(|value| value.is_finite())
    {
        return Err("Schur block exchange produced nonfinite factors");
    }
    Ok(())
}
