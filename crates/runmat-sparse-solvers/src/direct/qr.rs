//! Sparse Householder QR (COLAMD column ordering) for full-column-rank least squares.

use faer::sparse::linalg::qr::{factorize_symbolic_qr, QrRef, SymbolicQr};
use faer::{Conj, MatMut};

use super::{memory, try_vec, FactorScalar};
use crate::control::SolveControl;
use crate::csr::{CscRef, CsrMatrix};
use crate::error::{SolverError, SolverResult};
use crate::par;

/// `min ‖A x − b‖₂` for `m ≥ n` with full column rank. A rank-deficient matrix yields non-finite or
/// unreliable output (reported as `Singular`); a rank-revealing (complete orthogonal) treatment is
/// left to the caller.
pub struct LeastSquaresQr<T: FactorScalar> {
    symbolic: SymbolicQr<usize>,
    indices: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: FactorScalar> LeastSquaresQr<T> {
    /// Factors the CSC matrix `A` (`m × n`, `m ≥ n`).
    pub fn new(matrix: CscRef<'_, T>, control: SolveControl<'_>) -> SolverResult<Self> {
        let (m, n) = (matrix.nrows(), matrix.ncols());
        if m < n {
            return Err(SolverError::Dimension(
                "QR least squares requires rows >= columns".into(),
            ));
        }
        if let Some(entry) = matrix.values().iter().position(|v| !v.is_finite()) {
            return Err(SolverError::NonFinite { entry });
        }
        control.check()?;
        let symbolic = factorize_symbolic_qr(matrix.faer_symbolic(), Default::default())?;
        let mut indices = try_vec(symbolic.len_idx(), 0usize)?;
        let mut values = try_vec(symbolic.len_val(), T::zero())?;
        let par = par::faer_par();
        let mut buffer =
            memory(symbolic.factorize_numeric_qr_scratch::<T>(par, Default::default()))?;
        control.check()?;
        symbolic.factorize_numeric_qr::<T>(
            &mut indices,
            &mut values,
            matrix.faer(),
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
            Default::default(),
        );
        Ok(Self {
            symbolic,
            indices,
            values,
            nrows: m,
            ncols: n,
        })
    }

    /// Factors a CSR matrix by first converting it to column storage.
    pub fn from_csr(matrix: &CsrMatrix<T>, control: SolveControl<'_>) -> SolverResult<Self> {
        let transposed = matrix.as_ref().transpose(false);
        let csc = CscRef::new_unchecked(
            matrix.nrows(),
            matrix.ncols(),
            transposed.row_ptr(),
            transposed.col_idx(),
            transposed.values(),
        );
        Self::new(csc, control)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Solves for `k` column-major right-hand sides `b` (`m × k`); returns `x` (`n × k`).
    pub fn solve(&self, b: &[T], k: usize) -> SolverResult<Vec<T>> {
        let (m, n) = (self.nrows, self.ncols);
        if b.len() != m * k {
            return Err(SolverError::Dimension("least squares rhs".into()));
        }
        let mut work = b.to_vec();
        let par = par::faer_par();
        // SAFETY: indices and values were produced by this symbolic factorization.
        let qr = unsafe {
            QrRef::<usize, T>::new_unchecked(&self.symbolic, &self.indices, &self.values)
        };
        let mut buffer = memory(self.symbolic.solve_in_place_scratch::<T>(k, par))?;
        qr.solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(&mut work, m, k),
            par,
            faer::dyn_stack::MemStack::new(&mut buffer),
        );
        let mut x = Vec::with_capacity(n * k);
        for column in 0..k {
            x.extend_from_slice(&work[column * m..column * m + n]);
        }
        if let Some(position) = x.iter().position(|v| !v.is_finite()) {
            return Err(SolverError::Singular {
                pivot: position % n.max(1),
            });
        }
        Ok(x)
    }
}
