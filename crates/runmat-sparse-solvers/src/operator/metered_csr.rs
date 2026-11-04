//! Operation admission for a borrowed CSR matrix-vector kernel. The immutable
//! pattern is quoted once; every application reserves before changing output.
use super::LinearOperator;
use crate::{CsrRef, Scalar, SolverError, SolverResult};
use runmat_simulation_usage::{
    CountOverflow, LedgerError, OperationCount, UsageCategory, UsageLedger,
};

/// A matrix-vector operator with explicit operation-ledger ownership. Matrix
/// arrays and the ledger are borrowed; construction allocates no payload copy.
/// Numerical execution uses the ordinary CSR kernel and its reduction order.
pub struct MeteredCsrOperator<'a, T: Scalar> {
    matrix: CsrRef<'a, T>,
    ledger: &'a UsageLedger,
    category: UsageCategory,
    operations: OperationCount,
}

impl<'a, T: Scalar> MeteredCsrOperator<'a, T> {
    pub fn new(
        matrix: CsrRef<'a, T>,
        ledger: &'a UsageLedger,
        category: UsageCategory,
    ) -> SolverResult<Self> {
        let operations = matrix
            .row_ptr()
            .windows(2)
            .try_fold(OperationCount::ZERO, |sum, row| {
                sum.checked_add(row_operations(row[1] - row[0], T::IS_COMPLEX)?)
            })
            .map_err(LedgerError::from)?;
        Ok(Self {
            matrix,
            ledger,
            category,
            operations,
        })
    }

    /// Direct row-dot reference count for one complete matrix application.
    pub fn operations_per_apply(&self) -> OperationCount {
        self.operations
    }
}

impl<T: Scalar> LinearOperator<T> for MeteredCsrOperator<'_, T> {
    fn reference_apply_work(&self) -> SolverResult<Option<OperationCount>> {
        Ok(Some(self.operations))
    }
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        if x.len() != self.ncols() || y.len() != self.nrows() {
            return Err(SolverError::Dimension("CSR matrix-vector operands".into()));
        }
        let mut reservation = self.ledger.reserve(self.category, self.operations)?;
        self.matrix.spmv(x, y);
        reservation.complete(self.operations)?;
        Ok(())
    }
}

fn row_operations(entries: usize, complex: bool) -> Result<OperationCount, CountOverflow> {
    let entries = u64::try_from(entries).map_err(|_| CountOverflow)?;
    if !complex {
        return OperationCount::direct_real_dot(entries);
    }
    if entries == 0 {
        return Ok(OperationCount::ZERO);
    }
    // A complex product uses four real multiplications and two additions;
    // accumulation adds two real components. The first product initializes
    // the reference accumulator, so a length-n row costs 6n + 2(n−1).
    OperationCount::from_operations(6)
        .checked_mul(entries)?
        .checked_add(OperationCount::from_operations(2).checked_mul(entries - 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{c32, c64};

    fn exercise<T: Scalar>() {
        let row_ptr = [0, 2, 2, 3];
        let col_idx = [0, 1, 2];
        let values = [
            T::from_parts(2., 1.),
            T::from_parts(3., -1.),
            T::from_parts(4., 2.),
        ];
        let matrix = CsrRef::try_new(3, 3, &row_ptr, &col_idx, &values).unwrap();
        let expected_operations =
            OperationCount::from_operations(if T::IS_COMPLEX { 20 } else { 4 });
        let cap = expected_operations.checked_mul(2).unwrap();
        let ledger = UsageLedger::new(cap, expected_operations).unwrap();
        let operator = MeteredCsrOperator::new(matrix, &ledger, UsageCategory::Solve).unwrap();
        assert_eq!(operator.operations_per_apply(), expected_operations);
        let x = [
            T::from_parts(1., 1.),
            T::from_parts(2., -1.),
            T::from_parts(3., 2.),
        ];
        let mut expected = [T::zero(); 3];
        matrix.spmv(&x, &mut expected);
        let untouched = [T::from_f64(-7.); 3];
        let mut y = untouched;
        assert!(matches!(
            operator.apply(&x[..2], &mut y),
            Err(SolverError::Dimension(_))
        ));
        assert_eq!(y, untouched);
        operator.apply(&x, &mut y).unwrap();
        assert_eq!(y, expected);
        y = untouched;
        assert!(matches!(
            operator.apply(&x, &mut y),
            Err(SolverError::Usage(LedgerError::MoreGrantRequired { .. }))
        ));
        assert_eq!(y, untouched);
        ledger.update_grant(cap).unwrap();
        operator.apply(&x, &mut y).unwrap();
        assert_eq!(y, expected);
        y = untouched;
        assert!(matches!(
            operator.apply(&x, &mut y),
            Err(SolverError::Usage(LedgerError::BudgetExceeded { .. }))
        ));
        assert_eq!(y, untouched);
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.solve, cap);
        assert_eq!(snapshot.completed.total().unwrap(), cap);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
        assert_eq!(ledger.finish().unwrap(), snapshot.completed);
    }

    #[test]
    fn csr_execution_reserves_before_effects_and_records_only_completed_applications() {
        exercise::<f32>();
        exercise::<f64>();
        exercise::<c32>();
        exercise::<c64>();
    }

    #[test]
    fn row_schedule_handles_empty_rows_and_wide_counts_exactly() {
        assert_eq!(row_operations(0, true).unwrap(), OperationCount::ZERO);
        assert_eq!(row_operations(1, true).unwrap().operations(), 6);
        if let Ok(wide) = usize::try_from(1u64 << 61) {
            assert_eq!(
                row_operations(wide, true).unwrap().operations(),
                u64::MAX - 1
            );
            assert!(row_operations(wide + 1, true).is_err());
            assert!(row_operations(usize::MAX, false).is_err());
        }
    }
}
