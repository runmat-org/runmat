//! Reference batches owned by preconditioner construction and application.

use crate::error::SolverResult;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

pub(super) fn measured<R>(
    category: UsageCategory,
    count: OperationCount,
    work: impl FnOnce() -> SolverResult<R>,
) -> SolverResult<R> {
    let Some(context) = UsageContext::capture() else {
        return work();
    };
    let mut reservation = context.ledger().reserve(category, count)?;
    let result = work()?;
    reservation.complete(count)?;
    Ok(result)
}

/// Admit one elimination batch, then commit only its successfully completed
/// prefix. The counter is stack-local; no ledger lock is taken per coefficient.
pub(super) fn measured_prefix<R>(
    category: UsageCategory,
    capacity: OperationCount,
    work: impl FnOnce(&mut u64) -> SolverResult<R>,
) -> SolverResult<R> {
    let Some(context) = UsageContext::capture() else {
        return work(&mut 0);
    };
    let mut reservation = context.ledger().reserve(category, capacity)?;
    let mut completed = 0_u64;
    let result = work(&mut completed);
    reservation.complete(OperationCount::from_operations(completed))?;
    result
}

pub(super) fn point_jacobi_setup(elements: usize, complex: bool) -> SolverResult<OperationCount> {
    count(elements, if complex { 5 } else { 1 })
}

pub(super) fn point_jacobi_apply(elements: usize, complex: bool) -> SolverResult<OperationCount> {
    count(elements, if complex { 6 } else { 1 })
}

pub(super) fn block_jacobi_apply(
    elements: usize,
    block: usize,
    complex: bool,
) -> SolverResult<OperationCount> {
    if block == 0 || !elements.is_multiple_of(block) {
        return Err(crate::SolverError::Dimension(
            "block Jacobi application extent".into(),
        ));
    }
    let rows = u64::try_from(elements).map_err(|_| LedgerError::CountOverflow)?;
    let width = u64::try_from(block).map_err(|_| LedgerError::CountOverflow)?;
    let per_row = if complex {
        width.checked_mul(8).and_then(|value| value.checked_sub(2))
    } else {
        width.checked_mul(2).and_then(|value| value.checked_sub(1))
    }
    .ok_or(LedgerError::CountOverflow)?;
    OperationCount::from_operations(per_row)
        .checked_mul(rows)
        .map_err(|error| LedgerError::from(error).into())
}

pub(super) fn triangular_apply(
    rows: usize,
    off_diagonal: usize,
    complex: bool,
) -> SolverResult<OperationCount> {
    // L and U each use one multiply/subtract for every stored off-diagonal;
    // the U diagonal contributes one general division per row.
    count(off_diagonal, if complex { 8 } else { 2 })?
        .checked_add(count(rows, if complex { 11 } else { 1 })?)
        .map_err(|error| LedgerError::from(error).into())
}

pub(super) fn elimination_capacity(
    updates: usize,
    complex: bool,
    checked_pivot: bool,
) -> SolverResult<OperationCount> {
    let base = (if complex { 11 } else { 1 })
        + if checked_pivot {
            if complex {
                5
            } else {
                1
            }
        } else {
            0
        };
    count(1, base)?
        .checked_add(count(updates, if complex { 8 } else { 2 })?)
        .map_err(|error| LedgerError::from(error).into())
}

pub(super) fn count(elements: usize, per_element: u64) -> SolverResult<OperationCount> {
    OperationCount::from_operations(per_element)
        .checked_mul(u64::try_from(elements).map_err(|_| LedgerError::CountOverflow)?)
        .map_err(|error| LedgerError::from(error).into())
}
