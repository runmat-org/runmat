//! Small dense kernels (projected eigenproblems, coarse grids, local blocks) on faer with explicit
//! parallelism. Matrices are column-major.

use faer::{MatMut, MatRef};

use crate::error::{SolverError, SolverResult};
use crate::par;
use crate::scalar::Scalar;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

fn measured_dense<R>(
    category: UsageCategory,
    elements: usize,
    per_element: u64,
    work: impl FnOnce() -> R,
) -> SolverResult<R> {
    let Some(context) = UsageContext::capture() else {
        return Ok(work());
    };
    let elements = u64::try_from(elements).map_err(|_| LedgerError::CountOverflow)?;
    let count = OperationCount::from_operations(per_element)
        .checked_mul(elements)
        .map_err(LedgerError::from)?;
    let mut reservation = context.ledger().reserve(category, count)?;
    let result = work();
    reservation.complete(count)?;
    Ok(result)
}

/// Parallel above 512 rows, with at most `n` threads so the scratch stays within
/// [`symmetric_eigen_scratch_values`].
fn symmetric_eigen_parallelism(n: usize) -> faer::Par {
    let parallelism = par::faer_par();
    if n >= 512 && parallelism.degree() <= n {
        parallelism
    } else {
        faer::Par::Seq
    }
}

fn symmetric_eigen_scratch(n: usize, parallelism: faer::Par) -> faer::dyn_stack::StackReq {
    faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        parallelism,
        Default::default(),
    )
}

/// Upper bound on the scratch [`symmetric_eigen`] allocates for an `n × n` matrix, in `f64`
/// values. The parallel tridiagonalization holds one `n`-vector per thread, and the solver runs
/// at most `n` threads, so the parallel case adds at most `n²` values to the serial scratch.
/// The bound does not depend on the host's thread count.
pub fn symmetric_eigen_scratch_values(n: usize) -> usize {
    let serial = symmetric_eigen_scratch(n, faer::Par::Seq)
        .size_bytes()
        .div_ceil(std::mem::size_of::<f64>());
    let parallel = if n >= 512 { n.saturating_mul(n) } else { 0 };
    serial.saturating_add(parallel)
}

/// Eigen-decomposition of a real symmetric `n × n` matrix (lower triangle read). Returns
/// ascending eigenvalues and column-major orthonormal eigenvectors.
/// The opaque faer eigen kernel is explicitly excluded from rsua-1 in 0.7.0;
/// independently measured caller kernels retain their own charges.
pub fn symmetric_eigen(a: &[f64], n: usize) -> SolverResult<(Vec<f64>, Vec<f64>)> {
    assert_eq!(a.len(), n * n);
    let mut values = vec![0.0; n];
    let mut vectors = vec![0.0; n * n];
    if n == 0 {
        return Ok((values, vectors));
    }
    let parallelism = symmetric_eigen_parallelism(n);
    let req = symmetric_eigen_scratch(n, parallelism);
    let mut buffer =
        faer::dyn_stack::MemBuffer::try_new(req).map_err(|_| SolverError::OutOfMemory {
            bytes: req.size_bytes() as u64,
        })?;
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
    let mut buffer =
        faer::dyn_stack::MemBuffer::try_new(req).map_err(|_| SolverError::OutOfMemory {
            bytes: req.size_bytes() as u64,
        })?;
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
            let magnitude_evaluations = if T::IS_COMPLEX {
                (k..n).filter(|&i| a[k * n + i].im() != 0.0).count()
            } else {
                0
            };
            let (p, best) = measured_dense(
                UsageCategory::Factorization,
                magnitude_evaluations,
                1,
                || {
                    let mut p = k;
                    let mut best = a[k * n + k].abs();
                    for i in k + 1..n {
                        let v = a[k * n + i].abs();
                        if v > best {
                            best = v;
                            p = i;
                        }
                    }
                    (p, best)
                },
            )?;
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
            let remaining = n - k - 1;
            measured_dense(
                UsageCategory::Factorization,
                remaining,
                if T::IS_COMPLEX { 11 } else { 1 },
                || {
                    for i in k + 1..n {
                        a[k * n + i] /= pivot;
                    }
                },
            )?;
            for c in k + 1..n {
                let factor = a[c * n + k];
                if factor != T::zero() {
                    measured_dense(
                        UsageCategory::Factorization,
                        remaining,
                        if T::IS_COMPLEX { 8 } else { 2 },
                        || {
                            for i in k + 1..n {
                                let l = a[k * n + i];
                                a[c * n + i] -= l * factor;
                            }
                        },
                    )?;
                }
            }
        }
        Ok(Self { n, lu: a, pivots })
    }

    pub fn solve_in_place(&self, b: &mut [T]) -> SolverResult<()> {
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::Dimension(
                "dense LU right-hand side extent".into(),
            ));
        }
        for c in 0..n {
            measured_dense(
                UsageCategory::Solve,
                n - c - 1,
                if T::IS_COMPLEX { 8 } else { 2 },
                || {
                    if c == 0 {
                        for k in 0..n {
                            b.swap(k, self.pivots[k]);
                        }
                    }
                    let value = b[c];
                    for (i, target) in b.iter_mut().enumerate().take(n).skip(c + 1) {
                        *target -= self.lu[c * n + i] * value;
                    }
                },
            )?;
        }
        for c in (0..n).rev() {
            measured_dense(
                UsageCategory::Solve,
                1,
                if T::IS_COMPLEX { 11 } else { 1 },
                || {
                    b[c] /= self.lu[c * n + c];
                },
            )?;
            let value = b[c];
            measured_dense(
                UsageCategory::Solve,
                c,
                if T::IS_COMPLEX { 8 } else { 2 },
                || {
                    for (i, target) in b.iter_mut().enumerate().take(c) {
                        *target -= self.lu[c * n + i] * value;
                    }
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
        lu.solve_in_place(&mut y).unwrap();
        // [[0,2],[1,3]] y = [2,4] -> y = [1, 1]
        assert!((y[0] - 1.0).abs() < 1e-14 && (y[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn dense_lu_solve_refuses_before_permutation_and_conserves_completed_work() {
        let lu = DenseLu::new(vec![0.0, 1.0, 2.0, 3.0], 2).unwrap();
        let count = OperationCount::from_operations;
        let ledger =
            Arc::new(runmat_simulation_usage::UsageLedger::new(count(6), count(1)).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        let _guard = context.enter();
        let original = [2.0, 4.0];
        let mut rhs = original;
        assert!(matches!(
            lu.solve_in_place(&mut rhs),
            Err(SolverError::Usage(LedgerError::MoreGrantRequired { .. }))
        ));
        assert_eq!(rhs, original);
        assert_eq!(
            ledger.snapshot().unwrap().completed.solve,
            OperationCount::ZERO
        );
        ledger.update_grant(count(6)).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();
        assert_eq!(rhs, [1.0, 1.0]);
        assert_eq!(ledger.snapshot().unwrap().completed.solve, count(6));
    }
}
