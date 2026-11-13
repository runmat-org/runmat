//! Generated model problems shared by unit tests and benchmarks.

pub mod elasticity;

use crate::csr::{CsrMatrix, TripletBuilder};
use crate::scalar::Scalar;

/// 7-point Laplacian on an `nx × ny × nz` grid with Dirichlet boundaries (SPD).
pub fn poisson3d<T: Scalar>(nx: usize, ny: usize, nz: usize) -> CsrMatrix<T> {
    let n = nx * ny * nz;
    let index = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let mut builder = TripletBuilder::with_capacity(n, n, 7 * n);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let row = index(i, j, k);
                builder.push(row, row, T::from_f64(6.0));
                if i > 0 {
                    builder.push(row, index(i - 1, j, k), T::from_f64(-1.0));
                }
                if i + 1 < nx {
                    builder.push(row, index(i + 1, j, k), T::from_f64(-1.0));
                }
                if j > 0 {
                    builder.push(row, index(i, j - 1, k), T::from_f64(-1.0));
                }
                if j + 1 < ny {
                    builder.push(row, index(i, j + 1, k), T::from_f64(-1.0));
                }
                if k > 0 {
                    builder.push(row, index(i, j, k - 1), T::from_f64(-1.0));
                }
                if k + 1 < nz {
                    builder.push(row, index(i, j, k + 1), T::from_f64(-1.0));
                }
            }
        }
    }
    builder.build()
}

/// 5-point Laplacian on an `nx × ny` grid.
pub fn poisson2d<T: Scalar>(nx: usize, ny: usize) -> CsrMatrix<T> {
    poisson3d(nx, ny, 1).with_shifted_diagonal(T::from_f64(-2.0))
}

impl<T: Scalar> CsrMatrix<T> {
    /// Adds `shift` to every stored diagonal entry.
    pub fn with_shifted_diagonal(mut self, shift: T) -> Self {
        let n = self.nrows().min(self.ncols());
        for row in 0..n {
            let (start, end) = (self.row_ptr()[row], self.row_ptr()[row + 1]);
            if let Some(offset) = self.col_idx()[start..end].iter().position(|&c| c == row) {
                self.values_mut()[start + offset] += shift;
            }
        }
        self
    }
}
