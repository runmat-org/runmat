//! Sparse direct factorization of one overlapping subdomain operator.
//!
//! The factorization sits behind a narrow interface so the shared sparse solver layer can
//! replace this implementation without touching the decomposition.

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{MatMut, Side};

use crate::error::{DistributedError, DistributedResult};

/// A factorized symmetric positive definite subdomain operator.
pub trait SubdomainFactorization: Send + Sync {
    fn dimension(&self) -> usize;
    /// Overwrite `rhs` with the solution of the factorized system.
    fn solve_in_place(&self, rhs: &mut [f64]);
}

/// Builds factorizations from the lower triangle of a CSC matrix (sorted row indices).
pub trait SubdomainSolverFactory: Send + Sync {
    fn factorize(
        &self,
        dimension: usize,
        column_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> DistributedResult<Box<dyn SubdomainFactorization>>;
}

/// Supernodal sparse Cholesky with fill-reducing ordering.
#[derive(Clone, Copy, Debug, Default)]
pub struct FaerCholeskyFactory;

struct FaerCholesky {
    dimension: usize,
    factor: Llt<usize, f64>,
}

impl SubdomainFactorization for FaerCholesky {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn solve_in_place(&self, rhs: &mut [f64]) {
        let rows = rhs.len();
        self.factor
            .solve_in_place(MatMut::from_column_major_slice_mut(rhs, rows, 1));
    }
}

impl SubdomainSolverFactory for FaerCholeskyFactory {
    fn factorize(
        &self,
        dimension: usize,
        column_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> DistributedResult<Box<dyn SubdomainFactorization>> {
        if column_offsets.len() != dimension + 1 || row_indices.len() != values.len() {
            return Err(DistributedError::Factorization(
                "subdomain operator has an inconsistent sparse layout".into(),
            ));
        }
        let symbolic = SymbolicSparseColMatRef::new_checked(
            dimension,
            dimension,
            &column_offsets,
            None,
            &row_indices,
        );
        let matrix = SparseColMatRef::new(symbolic, &values);
        let analysis = SymbolicLlt::try_new(symbolic, Side::Lower)
            .map_err(|error| DistributedError::Factorization(format!("{error:?}")))?;
        let factor = Llt::try_new_with_symbolic(analysis, matrix, Side::Lower)
            .map_err(|error| DistributedError::Factorization(format!("{error:?}")))?;
        Ok(Box::new(FaerCholesky { dimension, factor }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorizes_and_solves_a_tridiagonal_spd_system() {
        let n = 50;
        let mut offsets = vec![0];
        let mut rows = Vec::new();
        let mut values = Vec::new();
        for column in 0..n {
            rows.push(column);
            values.push(2.0);
            if column + 1 < n {
                rows.push(column + 1);
                values.push(-1.0);
            }
            offsets.push(rows.len());
        }
        let factor = FaerCholeskyFactory
            .factorize(n, offsets, rows, values)
            .unwrap();
        let mut rhs = vec![1.0; n];
        factor.solve_in_place(&mut rhs);
        // Residual of the full symmetric operator.
        for i in 0..n {
            let mut product = 2.0 * rhs[i];
            if i > 0 {
                product -= rhs[i - 1];
            }
            if i + 1 < n {
                product -= rhs[i + 1];
            }
            assert!((product - 1.0).abs() < 1e-10);
        }
    }
}
