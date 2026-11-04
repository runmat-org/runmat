//! Sparse matrix–matrix products (row-wise Gustavson), threaded over fixed row blocks with each
//! output row accumulated sequentially, so results are independent of the thread count.

use crate::csr::{CsrMatrix, CsrRef};
use crate::error::{SolverError, SolverResult};
use crate::par;
use crate::scalar::Scalar;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

const ROWS_PER_TASK: usize = 256;

type SparseRow<T> = (Vec<usize>, Vec<T>);

fn reference_block_work<T: Scalar>(
    a: CsrRef<'_, T>,
    b: CsrRef<'_, T>,
    start: usize,
    end: usize,
) -> SolverResult<OperationCount> {
    let mut marker = vec![usize::MAX; b.ncols()];
    let mut operations = 0_u64;
    for row in start..end {
        let (a_cols, _) = a.row(row);
        for &k in a_cols {
            let (b_cols, _) = b.row(k);
            for &j in b_cols {
                let update = marker[j] == row;
                marker[j] = row;
                let cost = if T::IS_COMPLEX {
                    if update {
                        8
                    } else {
                        6
                    }
                } else if update {
                    2
                } else {
                    1
                };
                operations = operations
                    .checked_add(cost)
                    .ok_or(LedgerError::CountOverflow)?;
            }
        }
    }
    Ok(OperationCount::from_operations(operations))
}

pub enum SpGemmCapacityError<E> {
    Numerical(SolverError),
    Admission(E),
}

impl<E> From<SolverError> for SpGemmCapacityError<E> {
    fn from(error: SolverError) -> Self {
        Self::Numerical(error)
    }
}

fn multiply_block<T: Scalar>(
    a: CsrRef<'_, T>,
    b: CsrRef<'_, T>,
    start: usize,
    end: usize,
) -> Vec<(Vec<usize>, Vec<T>)> {
    let ncols = b.ncols();
    let mut marker = vec![usize::MAX; ncols];
    let mut accumulator = vec![T::zero(); ncols];
    let mut rows = Vec::with_capacity(end - start);
    let mut columns: Vec<usize> = Vec::new();
    for row in start..end {
        columns.clear();
        let (a_cols, a_vals) = a.row(row);
        for (&k, &a_value) in a_cols.iter().zip(a_vals) {
            let (b_cols, b_vals) = b.row(k);
            for (&j, &b_value) in b_cols.iter().zip(b_vals) {
                if marker[j] != row {
                    marker[j] = row;
                    accumulator[j] = a_value * b_value;
                    columns.push(j);
                } else {
                    accumulator[j] += a_value * b_value;
                }
            }
        }
        columns.sort_unstable();
        let values = columns.iter().map(|&j| accumulator[j]).collect();
        rows.push((columns.clone(), values));
    }
    rows
}

/// Await the exact symbolic reference count of each immutable output row before multiplying it.
pub async fn multiply_with_capacity<T: Scalar, E>(
    a: CsrRef<'_, T>,
    b: CsrRef<'_, T>,
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<CsrMatrix<T>, SpGemmCapacityError<E>> {
    if a.ncols() != b.nrows() {
        return Err(SolverError::Dimension(format!(
            "spgemm {}x{} times {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
        ))
        .into());
    }
    let mut rows = Vec::with_capacity(a.nrows());
    for start in 0..a.nrows() {
        let end = start + 1;
        let quote = reference_block_work(a, b, start, end)?;
        owner
            .ensure_capacity(quote)
            .await
            .map_err(SpGemmCapacityError::Admission)?;
        let context = UsageContext::capture();
        let mut reservation = context
            .as_ref()
            .map(|context| {
                context
                    .ledger()
                    .reserve(UsageCategory::Factorization, quote)
            })
            .transpose()
            .map_err(SolverError::from)?;
        rows.extend(multiply_block(a, b, start, end));
        if let Some(reservation) = reservation.as_mut() {
            reservation.complete(quote).map_err(SolverError::from)?;
        }
    }
    Ok(CsrMatrix::from_rows(a.nrows(), b.ncols(), rows))
}

/// `C = A B`.
pub fn multiply<T: Scalar>(a: CsrRef<'_, T>, b: CsrRef<'_, T>) -> SolverResult<CsrMatrix<T>> {
    if a.ncols() != b.nrows() {
        return Err(SolverError::Dimension(format!(
            "spgemm {}x{} times {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }
    let ncols = b.ncols();
    let blocks = a.nrows().div_ceil(ROWS_PER_TASK);
    let usage = UsageContext::capture();
    let block_rows: Vec<SolverResult<Vec<SparseRow<T>>>> = par::map_range(blocks, 1, |block| {
        let start = block * ROWS_PER_TASK;
        let end = (start + ROWS_PER_TASK).min(a.nrows());
        let quote = if usage.is_some() {
            Some(reference_block_work(a, b, start, end)?)
        } else {
            None
        };
        let mut reservation = match (usage.as_ref(), quote) {
            (Some(context), Some(count)) => Some(
                context
                    .ledger()
                    .reserve(UsageCategory::Factorization, count)?,
            ),
            _ => None,
        };
        let rows = multiply_block(a, b, start, end);
        if let (Some(reservation), Some(count)) = (reservation.as_mut(), quote) {
            reservation.complete(count)?;
        }
        Ok(rows)
    });
    let rows = block_rows
        .into_iter()
        .collect::<SolverResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(CsrMatrix::from_rows(a.nrows(), ncols, rows))
}

/// Galerkin product `Pᴴ A P` (`Pᵀ A P` for real scalars).
pub fn galerkin<T: Scalar>(a: CsrRef<'_, T>, p: CsrRef<'_, T>) -> SolverResult<CsrMatrix<T>> {
    let ap = multiply(a, p)?;
    let pt = p.transpose(true);
    multiply(pt.as_ref(), ap.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::poisson3d;

    #[test]
    fn galerkin_matches_dense_product() {
        let a = poisson3d::<f64>(4, 3, 2);
        let n = a.nrows();
        // Piecewise-constant aggregation of pairs.
        let mut t = crate::csr::TripletBuilder::new(n, n.div_ceil(2));
        for i in 0..n {
            t.push(i, i / 2, 1.0 + (i % 3) as f64);
        }
        let p = t.build();
        let c = galerkin(a.as_ref(), p.as_ref()).unwrap();
        let (ad, pd) = (a.as_ref().to_dense(), p.as_ref().to_dense());
        let m = p.ncols();
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for r in 0..n {
                    for s in 0..n {
                        sum += pd[r * m + i] * ad[r * n + s] * pd[s * m + j];
                    }
                }
                assert!((c.as_ref().get(i, j) - sum).abs() < 1e-12);
            }
        }
    }
}
