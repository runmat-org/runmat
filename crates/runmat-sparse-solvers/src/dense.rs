//! Small dense kernels (projected eigenproblems, coarse grids, local blocks) on faer with explicit
//! parallelism. Matrices are column-major.

use faer::{MatMut, MatRef};

use crate::error::{SolverError, SolverResult};
use crate::par;
use crate::scalar::Scalar;

/// Eigen-decomposition of a real symmetric `n × n` matrix (lower triangle read). Returns
/// ascending eigenvalues and column-major orthonormal eigenvectors.
pub fn symmetric_eigen(a: &[f64], n: usize) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    assert_eq!(a.len(), n * n);
    let mut values = vec![0.0; n];
    let mut vectors = vec![0.0; n * n];
    if n == 0 {
        return Ok((values, vectors));
    }
    let parallelism = if n >= 512 { par::faer_par() } else { faer::Par::Seq };
    let req = faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        parallelism,
        Default::default(),
    );
    let mut buffer = faer::dyn_stack::MemBuffer::try_new(req)
        .map_err(|_| SolverError::OutOfMemory { bytes: req.size_bytes() as u64 })?;
    faer::linalg::evd::self_adjoint_evd(
        MatRef::from_column_major_slice(a, n, n),
        faer::diag::DiagMut::from_slice_mut(&mut values),
        Some(MatMut::from_column_major_slice_mut(&mut vectors, n, n)),
        parallelism,
        faer::dyn_stack::MemStack::new(&mut buffer),
        Default::default(),
    )
    .map_err(|error| SolverError::Breakdown(format!("dense symmetric eigensolver: {error:?}")))?;
    Ok((values, vectors))
}

/// In-place dense Cholesky `A = L Lᵀ` (lower triangle of the column-major input overwritten).
pub fn cholesky_in_place(a: &mut [f64], n: usize) -> SolverResult<()> {
    assert_eq!(a.len(), n * n);
    let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        faer::Par::Seq,
        Default::default(),
    );
    let mut buffer = faer::dyn_stack::MemBuffer::try_new(req)
        .map_err(|_| SolverError::OutOfMemory { bytes: req.size_bytes() as u64 })?;
    faer::linalg::cholesky::llt::factor::cholesky_in_place(
        MatMut::from_column_major_slice_mut(a, n, n),
        Default::default(),
        faer::Par::Seq,
        faer::dyn_stack::MemStack::new(&mut buffer),
        Default::default(),
    )
    .map_err(|error| match error {
        faer::linalg::cholesky::llt::factor::LltError::NonPositivePivot { index } => {
            SolverError::NotPositiveDefinite { pivot: index }
        }
    })?;
    Ok(())
}

/// Solves `L Lᵀ X = B` with a factor from [`cholesky_in_place`] (`k` column-major right-hand sides).
pub fn cholesky_solve(l: &[f64], n: usize, rhs: &mut [f64], k: usize) {
    faer::linalg::cholesky::llt::solve::solve_in_place_with_conj(
        MatRef::from_column_major_slice(l, n, n),
        faer::Conj::No,
        MatMut::from_column_major_slice_mut(rhs, n, k),
        faer::Par::Seq,
        faer::dyn_stack::MemStack::new(&mut faer::dyn_stack::MemBuffer::new(
            faer::linalg::cholesky::llt::solve::solve_in_place_scratch::<f64>(n, k, faer::Par::Seq),
        )),
    );
}

/// Dense LU with partial pivoting for small general systems (column-major), any scalar.
#[derive(Debug, Clone)]
pub struct DenseLu<T> {
    n: usize,
    lu: Vec<T>,
    pivots: Vec<usize>,
}

impl<T: Scalar> DenseLu<T> {
    /// Factors column-major `a`; pivots of modulus `<= pivot_floor` are rejected as singular.
    pub fn with_pivot_floor(a: Vec<T>, n: usize, pivot_floor: f64) -> SolverResult<Self> {
        Self::factor(a, n, pivot_floor)
    }

    /// Factors a row-major matrix (`a[row * n + column]`).
    pub fn from_row_major(a: &[T], n: usize, pivot_floor: f64) -> SolverResult<Self> {
        let mut column_major = vec![T::zero(); n * n];
        for row in 0..n {
            for column in 0..n {
                column_major[column * n + row] = a[row * n + column];
            }
        }
        Self::factor(column_major, n, pivot_floor)
    }

    pub fn new(a: Vec<T>, n: usize) -> SolverResult<Self> {
        Self::factor(a, n, 0.0)
    }

    fn factor(mut a: Vec<T>, n: usize, pivot_floor: f64) -> SolverResult<Self> {
        assert_eq!(a.len(), n * n);
        let mut pivots = vec![0usize; n];
        for k in 0..n {
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[k * n + i].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best <= pivot_floor || best == 0.0 || !best.is_finite() {
                return Err(SolverError::Singular { pivot: k });
            }
            pivots[k] = p;
            if p != k {
                for c in 0..n {
                    a.swap(c * n + k, c * n + p);
                }
            }
            let pivot = a[k * n + k];
            for i in k + 1..n {
                a[k * n + i] = a[k * n + i] / pivot;
            }
            for c in k + 1..n {
                let factor = a[c * n + k];
                if factor != T::zero() {
                    for i in k + 1..n {
                        let l = a[k * n + i];
                        a[c * n + i] -= l * factor;
                    }
                }
            }
        }
        Ok(Self { n, lu: a, pivots })
    }

    pub fn solve_in_place(&self, b: &mut [T]) {
        let n = self.n;
        for k in 0..n {
            b.swap(k, self.pivots[k]);
        }
        for c in 0..n {
            let value = b[c];
            for i in c + 1..n {
                b[i] -= self.lu[c * n + i] * value;
            }
        }
        for c in (0..n).rev() {
            b[c] = b[c] / self.lu[c * n + c];
            let value = b[c];
            for i in 0..c {
                b[i] -= self.lu[c * n + i] * value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_kernels_match_known_values() {
        let a = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let (values, _) = symmetric_eigen(&a, 3).unwrap();
        let expected = [2.0 - 2f64.sqrt(), 2.0, 2.0 + 2f64.sqrt()];
        for (v, e) in values.iter().zip(expected) {
            assert!((v - e).abs() < 1e-12);
        }
        let mut l = a.clone();
        cholesky_in_place(&mut l, 3).unwrap();
        let mut x = vec![1.0, 0.0, 1.0];
        cholesky_solve(&l, 3, &mut x, 1);
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
        let lu = DenseLu::new(vec![0.0, 1.0, 2.0, 3.0], 2).unwrap();
        let mut y = vec![2.0, 4.0];
        lu.solve_in_place(&mut y);
        // [[0,2],[1,3]] y = [2,4] -> y = [1, 1]
        assert!((y[0] - 1.0).abs() < 1e-14 && (y[1] - 1.0).abs() < 1e-14);
    }
}
