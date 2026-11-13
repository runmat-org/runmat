//! Residual certification is separate from the ordinary SpMV hot path.
//!
//! Product roundoff is recovered with FMA and addition roundoff is accumulated
//! separately, including conversion back to the scalar's storage precision.
//! See Ogita, Rump and Oishi, Accurate Sum and Dot Product (2005),
//! https://doi.org/10.1137/030601818. This is not arbitrary-precision arithmetic
//! or a promise of correctly rounded results for every input.

use rayon::prelude::*;

use super::{CscRef, CsrRef, SPMV_ROWS_PER_TASK};
use crate::{par, scalar::Scalar};

fn add<T: Scalar>(sum: &mut f64, error: &mut f64, value: f64) {
    let next = *sum + value;
    let stored = T::from_f64(next).re();
    if !next.is_finite() || !stored.is_finite() {
        *sum = stored;
        *error = 0.0;
        return;
    }
    let virtual_value = next - *sum;
    let roundoff = (*sum - (next - virtual_value)) + (value - virtual_value);
    *error = T::from_f64(*error + (roundoff + (next - stored))).re();
    *sum = stored;
}

fn product<T: Scalar>(sum: &mut f64, error: &mut f64, a: f64, x: f64, sign: f64) {
    let high = a * x;
    add::<T>(sum, error, sign * high);
    if high.is_finite() {
        add::<T>(sum, error, sign * a.mul_add(x, -high));
    }
}

fn subtract_product<T: Scalar>(sum: T, error: T, a: T, x: T) -> (T, T) {
    let (mut real, mut real_error) = (sum.re(), error.re());
    product::<T>(&mut real, &mut real_error, a.re(), x.re(), -1.0);
    let (mut imag, mut imag_error) = (sum.im(), error.im());
    if T::IS_COMPLEX {
        product::<T>(&mut real, &mut real_error, a.im(), x.im(), 1.0);
        product::<T>(&mut imag, &mut imag_error, a.re(), x.im(), -1.0);
        product::<T>(&mut imag, &mut imag_error, a.im(), x.re(), -1.0);
    }
    (
        T::from_parts(real, imag),
        T::from_parts(real_error, imag_error),
    )
}

fn finish<T: Scalar>(sum: T, error: T) -> T {
    T::from_parts(sum.re() + error.re(), sum.im() + error.im())
}

impl<T: Scalar> CsrRef<'_, T> {
    /// Evaluate `b - A*x` with compensated products and sums. Unlike an
    /// ordinary SpMV followed by subtraction, this retains a small defect
    /// after cancellation. No additional vector allocation is performed.
    pub fn residual(&self, x: &[T], b: &[T], residual: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "residual input length");
        assert_eq!(b.len(), self.nrows, "residual RHS length");
        assert_eq!(residual.len(), self.nrows, "residual output length");
        let row_kernel = |row: usize| {
            let (mut sum, mut error) = (b[row], T::zero());
            for entry in self.row_ptr[row]..self.row_ptr[row + 1] {
                (sum, error) =
                    subtract_product(sum, error, self.values[entry], x[self.col_idx[entry]]);
            }
            finish(sum, error)
        };
        if par::threads() > 1 && self.nrows >= 4 * SPMV_ROWS_PER_TASK {
            residual
                .par_chunks_mut(SPMV_ROWS_PER_TASK)
                .enumerate()
                .for_each(|(task, chunk)| {
                    let first = task * SPMV_ROWS_PER_TASK;
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        *value = row_kernel(first + offset);
                    }
                });
        } else {
            for (row, value) in residual.iter_mut().enumerate() {
                *value = row_kernel(row);
            }
        }
    }
}

impl<T: Scalar> CscRef<'_, T> {
    /// Compensated `b - A*x` using caller-owned row-error scratch. The same
    /// column order as CSR is retained; no matrix conversion or allocation is
    /// performed. Scratch and output must each contain one scalar per row.
    pub fn residual(&self, x: &[T], b: &[T], residual: &mut [T], errors: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "residual input length");
        assert_eq!(b.len(), self.nrows, "residual RHS length");
        assert_eq!(residual.len(), self.nrows, "residual output length");
        assert_eq!(errors.len(), self.nrows, "residual scratch length");
        residual.copy_from_slice(b);
        errors.fill(T::zero());
        for (column, &operand) in x.iter().enumerate() {
            for entry in self.col_ptr[column]..self.col_ptr[column + 1] {
                let row = self.row_idx[entry];
                (residual[row], errors[row]) =
                    subtract_product(residual[row], errors[row], self.values[entry], operand);
            }
        }
        for (value, error) in residual.iter_mut().zip(errors) {
            *value = finish(*value, *error);
        }
    }
}

#[cfg(test)]
mod tests;
