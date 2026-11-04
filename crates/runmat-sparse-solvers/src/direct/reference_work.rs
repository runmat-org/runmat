use super::{FactorKind, SymbolicFactorization, SymbolicInner};
use runmat_simulation_usage::{CountOverflow, OperationCount};

pub(super) enum DirectPhase {
    Factorization,
    Solve { rhs_columns: usize },
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
    let reference = symbolic.reference_work().map_err(|error| match error {
        ReferenceWorkError::Overflow(error) => crate::SolverError::Usage(error.into()),
        error => crate::SolverError::Unsupported(error.to_string()),
    })?;
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
    /// LLT/LDLT reference arithmetic over the stored symbolic fill. This count is
    /// independent of the floating-point memory/performance planning estimate.
    /// Square roots are retained separately for diagnostics and included by
    /// `DirectReferenceWork::factorization_operations` in the measured total.
    pub fn reference_work(&self) -> Result<DirectReferenceWork, ReferenceWorkError> {
        if !matches!(self.kind, FactorKind::Cholesky | FactorKind::Ldlt) {
            return Err(ReferenceWorkError::FactorKind(self.kind));
        }
        let SymbolicInner::Cholesky(symbolic) = &self.inner else {
            return Err(ReferenceWorkError::FactorKind(self.kind));
        };
        match symbolic.raw() {
            faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Simplicial(factor) => {
                hermitian_columns(
                    factor.col_ptr().windows(2).map(|pair| pair[1] - pair[0]),
                    self.complex,
                    self.kind,
                )
            }
            faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Supernodal(factor) => {
                let begin = factor.supernode_begin();
                let end = factor.supernode_end();
                hermitian_columns(
                    (0..factor.n_supernodes()).flat_map(|node| {
                        let width = end[node] - begin[node];
                        let trailing = factor.supernode(node).pattern().len();
                        (0..width).map(move |column| width - column + trailing)
                    }),
                    self.complex,
                    self.kind,
                )
            }
        }
    }
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
