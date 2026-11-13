use super::{FactorKind, SymbolicFactorization, SymbolicInner};
use runmat_simulation_usage::{CountOverflow, OperationCount};

pub(super) enum DirectPhase {
    Factorization,
    Solve { rhs_columns: usize },
}

/// The rsua-1 decision for one sparse direct numeric kernel. An exclusion is
/// explicit and does not imply that its enclosing solve has zero measured work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectMeasurementCoverage {
    Measured(DirectReferenceWork),
    Excluded(ExcludedDirectKernel),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExcludedDirectKernel {
    /// Pivot-dependent sparse Bunch–Kaufman LBLᴴ factorization and substitution.
    SparseBunchKaufman,
}

/// One completed reference batch. Numerical buffers stay with their existing
/// solver owner; the reservation borrows the admitted run's shared ledger.
pub(super) fn with_direct_usage<R>(
    symbolic: &SymbolicFactorization,
    phase: DirectPhase,
    work: impl FnOnce() -> crate::SolverResult<R>,
) -> crate::SolverResult<R> {
    use runmat_simulation_usage::{LedgerError, UsageCategory, UsageContext};
    let Some(context) = UsageContext::capture() else {
        return work();
    };
    let coverage = symbolic
        .measurement_coverage()
        .map_err(|error| match error {
            ReferenceWorkError::Overflow(error) => crate::SolverError::Usage(error.into()),
            error => crate::SolverError::Unsupported(error.to_string()),
        })?;
    let DirectMeasurementCoverage::Measured(reference) = coverage else {
        // Only the named pivot-dependent sparse kernels are excluded. The
        // caller still charges independently owned residual/vector batches.
        return work();
    };
    let (category, operations) = match phase {
        DirectPhase::Factorization => (
            UsageCategory::Factorization,
            reference.factorization_operations(),
        ),
        DirectPhase::Solve { rhs_columns } => (
            UsageCategory::Solve,
            reference.solve_operations(
                u64::try_from(rhs_columns).map_err(|_| crate::SolverError::IndexOverflow)?,
            ),
        ),
    };
    let operations = operations.map_err(LedgerError::from)?;
    let mut reservation = context.ledger().reserve(category, operations)?;
    let result = work()?;
    reservation.complete(operations)?;
    Ok(result)
}

/// Reference arithmetic for a direct factorization and its substitutions,
/// derived from symbolic columns rather than a floating-point planning estimate.
/// Numerical execution owners determine which work actually completed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectReferenceWork {
    pub basic_operations: OperationCount,
    pub square_roots: u64,
    /// Forward and backward substitution for one right-hand side, including
    /// diagonal scaling and every stored off-diagonal multiply/subtract.
    pub solve_operations_per_rhs: OperationCount,
}

impl DirectReferenceWork {
    /// Complete factorization under rsua-1: each real square root contributes
    /// one reference operation, independent of its backend implementation.
    pub fn factorization_operations(self) -> Result<OperationCount, CountOverflow> {
        self.basic_operations
            .checked_add(OperationCount::from_operations(self.square_roots))
    }

    /// Applying a previously computed factor charges only the substitutions.
    pub fn solve_operations(self, rhs_columns: u64) -> Result<OperationCount, CountOverflow> {
        self.solve_operations_per_rhs.checked_mul(rhs_columns)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ReferenceWorkError {
    #[error("the direct reference schedule does not apply to {0:?}")]
    FactorKind(FactorKind),
    #[error("invalid reference column: the diagonal entry is absent")]
    InvalidColumn,
    #[error(transparent)]
    Overflow(#[from] CountOverflow),
}

impl SymbolicFactorization {
    /// Classifies the exact rsua-1 arithmetic boundary for this symbolic kind.
    /// A new factor kind must be classified here; an unsupported schedule must
    /// never be interpreted as a measured zero.
    pub fn measurement_coverage(&self) -> Result<DirectMeasurementCoverage, ReferenceWorkError> {
        match self.kind {
            FactorKind::Cholesky | FactorKind::Ldlt | FactorKind::Lu => self
                .reference_work()
                .map(DirectMeasurementCoverage::Measured),
            FactorKind::BunchKaufman => Ok(DirectMeasurementCoverage::Excluded(
                ExcludedDirectKernel::SparseBunchKaufman,
            )),
        }
    }

    /// LLT/LDLT reference arithmetic uses the stored symbolic columns. LU uses
    /// elimination fill of the symmetrized pattern recorded at analysis. This
    /// count is independent of the floating-point memory/performance estimate.
    /// Square roots are retained separately for diagnostics and included by
    /// `DirectReferenceWork::factorization_operations` in the measured total.
    pub fn reference_work(&self) -> Result<DirectReferenceWork, ReferenceWorkError> {
        match (&self.inner, self.kind) {
            (SymbolicInner::Lu { reference, .. }, FactorKind::Lu) => *reference,
            (SymbolicInner::Cholesky(symbolic), FactorKind::Cholesky | FactorKind::Ldlt) => {
                hermitian_columns(cholesky_column_heights(symbolic), self.complex, self.kind)
            }
            _ => Err(ReferenceWorkError::FactorKind(self.kind)),
        }
    }
}

/// Column heights (diagonal plus stored off-diagonal rows) of a symbolic
/// Cholesky factor, in elimination order.
pub(super) fn cholesky_column_heights(
    symbolic: &faer::sparse::linalg::cholesky::SymbolicCholesky<usize>,
) -> Box<dyn Iterator<Item = usize> + '_> {
    match symbolic.raw() {
        faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Simplicial(factor) => {
            Box::new(factor.col_ptr().windows(2).map(|pair| pair[1] - pair[0]))
        }
        faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Supernodal(factor) => {
            let begin = factor.supernode_begin();
            let end = factor.supernode_end();
            Box::new((0..factor.n_supernodes()).flat_map(move |node| {
                let width = end[node] - begin[node];
                let trailing = factor.supernode(node).pattern().len();
                (0..width).map(move |column| width - column + trailing)
            }))
        }
    }
}

/// LU reference schedule over the symmetrized pattern's elimination fill. A
/// column whose L part holds p off-diagonal entries has a U row of p entries.
/// Factorization forms one pivot reciprocal (real 1, complex 6: |d|², two
/// divisions and a negation), scales p entries of L (real 1, complex 6 each)
/// and applies p² trailing multiply/subtract updates (real 2, complex 8 each).
/// One right-hand side costs p multiply/subtract updates in each triangle plus
/// one pivot scaling. The charge is exact for structurally symmetric matrices
/// pivoting on the diagonal; numerical pivoting does not change it, so the
/// same pattern always meters the same work.
pub(super) fn general_columns(
    columns: impl Iterator<Item = usize>,
    complex: bool,
) -> Result<DirectReferenceWork, ReferenceWorkError> {
    let (pivot, scale, update) = if complex { (6, 6, 8) } else { (1, 1, 2) };
    let mut work = DirectReferenceWork::default();
    for height in columns {
        let p = height
            .checked_sub(1)
            .ok_or(ReferenceWorkError::InvalidColumn)?;
        let p = u128::try_from(p).map_err(|_| CountOverflow)?;
        let factor = p
            .checked_mul(p)
            .and_then(|square| square.checked_mul(update))
            .and_then(|value| value.checked_add(p.checked_mul(scale)?))
            .and_then(|value| value.checked_add(pivot))
            .ok_or(CountOverflow)?;
        work.basic_operations =
            work.basic_operations
                .checked_add(OperationCount::from_operations(
                    u64::try_from(factor).map_err(|_| CountOverflow)?,
                ))?;
        let solve = p
            .checked_mul(2 * update)
            .and_then(|value| value.checked_add(scale))
            .ok_or(CountOverflow)?;
        work.solve_operations_per_rhs =
            work.solve_operations_per_rhs
                .checked_add(OperationCount::from_operations(
                    u64::try_from(solve).map_err(|_| CountOverflow)?,
                ))?;
    }
    Ok(work)
}

/// One column with p off-diagonal entries performs p divisions and p(p+1)/2
/// trailing lower-triangle multiply/subtract updates: p²+2p basic real ops.
/// For Hermitian complex values, real-diagonal division costs 2p; diagonal
/// updates cost 4p and strictly lower updates cost 8*p(p−1)/2, totaling 4p²+2p.
/// LLT columns also have one real square root, represented separately. LDLT
/// retains each pre-division column for the same reference trailing updates.
pub(super) fn hermitian_columns(
    columns: impl Iterator<Item = usize>,
    complex: bool,
    kind: FactorKind,
) -> Result<DirectReferenceWork, ReferenceWorkError> {
    let diagonal_scale = match kind {
        FactorKind::Cholesky => {
            if complex {
                4
            } else {
                2
            }
        }
        FactorKind::Ldlt => {
            if complex {
                2
            } else {
                1
            }
        }
        kind => return Err(ReferenceWorkError::FactorKind(kind)),
    };
    let mut work = DirectReferenceWork::default();
    for height in columns {
        let p = height
            .checked_sub(1)
            .ok_or(ReferenceWorkError::InvalidColumn)?;
        let p = u128::try_from(p).map_err(|_| CountOverflow)?;
        let square = p.checked_mul(p).ok_or(CountOverflow)?;
        let operations = square
            .checked_mul(if complex { 4 } else { 1 })
            .and_then(|value| {
                p.checked_mul(2)
                    .and_then(|linear| value.checked_add(linear))
            })
            .ok_or(CountOverflow)?;
        let operations = u64::try_from(operations).map_err(|_| CountOverflow)?;
        work.basic_operations = work
            .basic_operations
            .checked_add(OperationCount::from_operations(operations))?;
        if kind == FactorKind::Cholesky {
            work.square_roots = work.square_roots.checked_add(1).ok_or(CountOverflow)?;
        }
        // Two triangular substitutions. LLT divides by two real diagonals;
        // LDLT has unit triangular diagonals and one real D division.
        let solve_operations = p
            .checked_mul(if complex { 16 } else { 4 })
            .and_then(|value| value.checked_add(diagonal_scale))
            .ok_or(CountOverflow)?;
        work.solve_operations_per_rhs =
            work.solve_operations_per_rhs
                .checked_add(OperationCount::from_operations(
                    u64::try_from(solve_operations).map_err(|_| CountOverflow)?,
                ))?;
    }
    Ok(work)
}

#[cfg(test)]
#[path = "reference_work/tests.rs"]
mod tests;
