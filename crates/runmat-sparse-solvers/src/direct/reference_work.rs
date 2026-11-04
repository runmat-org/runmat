use super::{FactorKind, SymbolicFactorization, SymbolicInner};
use runmat_simulation_usage::{CountOverflow, OperationCount};

pub(super) enum LltPhase {
    Factorization,
    Solve { rhs_columns: usize },
}

/// One completed reference batch. Numerical buffers stay with their existing
/// solver owner; the reservation borrows the admitted run's shared ledger.
pub(super) fn with_llt_usage<R>(
    symbolic: &SymbolicFactorization,
    phase: LltPhase,
    work: impl FnOnce() -> crate::SolverResult<R>,
) -> crate::SolverResult<R> {
    use runmat_simulation_usage::{LedgerError, UsageCategory, UsageContext};
    let Some(context) = UsageContext::capture() else {
        return work();
    };
    let reference = symbolic.llt_reference_work().map_err(|error| match error {
        ReferenceWorkError::Overflow(error) => crate::SolverError::Usage(error.into()),
        error => crate::SolverError::Unsupported(error.to_string()),
    })?;
    let (category, operations) = match phase {
        LltPhase::Factorization => (
            UsageCategory::Factorization,
            reference.factorization_operations(),
        ),
        LltPhase::Solve { rhs_columns } => (
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

/// Reference arithmetic for a complete LLT factorization and its substitutions,
/// derived from symbolic columns rather than a floating-point planning estimate.
/// Numerical execution owners determine which work actually completed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LltReferenceWork {
    pub basic_operations: OperationCount,
    pub square_roots: u64,
    /// Forward and backward substitution for one right-hand side, including
    /// both diagonal divisions and every stored off-diagonal multiply/subtract.
    pub solve_operations_per_rhs: OperationCount,
}

impl LltReferenceWork {
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
    #[error("the LLT reference schedule does not apply to {0:?}")]
    FactorKind(FactorKind),
    #[error("invalid LLT reference column: the diagonal entry is absent")]
    InvalidColumn,
    #[error(transparent)]
    Overflow(#[from] CountOverflow),
}

impl SymbolicFactorization {
    /// LLT reference arithmetic over the stored symbolic fill. This count is
    /// independent of the floating-point memory/performance planning estimate.
    /// Square roots are retained separately for diagnostics and included by
    /// `LltReferenceWork::factorization_operations` in the measured total.
    pub fn llt_reference_work(&self) -> Result<LltReferenceWork, ReferenceWorkError> {
        if self.kind != FactorKind::Cholesky {
            return Err(ReferenceWorkError::FactorKind(self.kind));
        }
        let SymbolicInner::Cholesky(symbolic) = &self.inner else {
            return Err(ReferenceWorkError::FactorKind(self.kind));
        };
        match symbolic.raw() {
            faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Simplicial(factor) => llt_columns(
                factor.col_ptr().windows(2).map(|pair| pair[1] - pair[0]),
                self.complex,
            ),
            faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Supernodal(factor) => {
                let begin = factor.supernode_begin();
                let end = factor.supernode_end();
                llt_columns(
                    (0..factor.n_supernodes()).flat_map(|node| {
                        let width = end[node] - begin[node];
                        let trailing = factor.supernode(node).pattern().len();
                        (0..width).map(move |column| width - column + trailing)
                    }),
                    self.complex,
                )
            }
        }
    }
}

/// One column with p off-diagonal entries performs p divisions and p(p+1)/2
/// trailing lower-triangle multiply/subtract updates: p²+2p basic real ops.
/// For Hermitian complex values, real-diagonal division costs 2p; diagonal
/// updates cost 4p and strictly lower updates cost 8*p(p−1)/2, totaling 4p²+2p.
/// Each column also has one real square root, represented separately.
pub(super) fn llt_columns(
    columns: impl Iterator<Item = usize>,
    complex: bool,
) -> Result<LltReferenceWork, ReferenceWorkError> {
    let mut work = LltReferenceWork::default();
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
        work.square_roots = work.square_roots.checked_add(1).ok_or(CountOverflow)?;
        // Two triangular substitutions. LLT diagonals are real even for a
        // Hermitian complex factor: each complex/real division costs two ops.
        let solve_operations = p
            .checked_mul(if complex { 16 } else { 4 })
            .and_then(|value| value.checked_add(if complex { 4 } else { 2 }))
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
mod tests {
    use super::*;
    use crate::{
        control::SolveControl,
        csr::TripletBuilder,
        direct::{DirectOptions, DirectWorkspacePolicy, SymbolicFactorization},
        ordering::Ordering,
    };

    #[test]
    fn reference_arithmetic_counts_real_and_complex_primitive_updates() {
        // Count a direct right-looking reference independently of the formula.
        for complex in [false, true] {
            for heights in [vec![], vec![1], vec![3, 2, 1], vec![2, 2, 2, 1]] {
                let mut basic = 0u64;
                let mut solve = 0u64;
                for &height in &heights {
                    let trailing = height - 1;
                    for _ in 0..trailing {
                        basic += if complex { 2 } else { 1 };
                    }
                    for row in 0..trailing {
                        for column in 0..=row {
                            basic += if !complex {
                                2
                            } else if row == column {
                                4
                            } else {
                                8
                            };
                        }
                    }
                    for _ in 0..2 {
                        solve += if complex { 2 } else { 1 };
                        for _ in 0..trailing {
                            solve += if complex { 8 } else { 2 };
                        }
                    }
                }
                let measured = llt_columns(heights.iter().copied(), complex).unwrap();
                assert_eq!(measured.basic_operations.operations(), basic);
                assert_eq!(measured.square_roots, heights.len() as u64);
                assert_eq!(
                    measured.factorization_operations().unwrap().operations(),
                    basic + heights.len() as u64
                );
                assert_eq!(measured.solve_operations_per_rhs.operations(), solve);
                assert_eq!(measured.solve_operations(0).unwrap(), OperationCount::ZERO);
                assert_eq!(
                    measured.solve_operations(3).unwrap().operations(),
                    3 * solve
                );
            }
        }
        assert!(matches!(
            llt_columns([0].into_iter(), false),
            Err(ReferenceWorkError::InvalidColumn)
        ));
        assert!(LltReferenceWork {
            basic_operations: OperationCount::from_operations(u64::MAX),
            square_roots: 1,
            ..Default::default()
        }
        .factorization_operations()
        .is_err());
        if usize::BITS == 64 {
            assert!(matches!(
                llt_columns([usize::MAX].into_iter(), true),
                Err(ReferenceWorkError::Overflow(_))
            ));
        }
    }

    #[test]
    fn symbolic_factors_use_stored_fill_and_reject_other_families() {
        let mut entries = TripletBuilder::<f64>::new(3, 3);
        for row in 0..3 {
            for column in 0..3 {
                entries.push(row, column, if row == column { 4.0 } else { 1.0 });
            }
        }
        let matrix = entries.build();
        for workspace in [
            DirectWorkspacePolicy::CallerOwned,
            DirectWorkspacePolicy::BackendManaged,
        ] {
            let options = DirectOptions {
                ordering: Ordering::Identity,
                workspace,
                ..DirectOptions::new(FactorKind::Cholesky)
            };
            let mut symbolic = SymbolicFactorization::analyze::<f64>(
                matrix.as_ref().into(),
                &options,
                SolveControl::NONE,
            )
            .unwrap();
            // Exercise both storage variants even when AUTO would choose
            // simplicial storage for this small input.
            if workspace == DirectWorkspacePolicy::BackendManaged {
                use faer::sparse::linalg::{
                    cholesky::{
                        factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
                    },
                    SupernodalThreshold,
                };
                let raw = factorize_symbolic_cholesky(
                    matrix.as_ref().as_transpose_csc().faer_symbolic(),
                    faer::Side::Lower,
                    SymmetricOrdering::Identity,
                    CholeskySymbolicParams {
                        supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL,
                        ..Default::default()
                    },
                )
                .unwrap();
                std::sync::Arc::get_mut(&mut symbolic).unwrap().inner =
                    SymbolicInner::Cholesky(raw);
            }
            let work = symbolic.llt_reference_work().unwrap();
            assert_eq!(work.basic_operations.operations(), 11);
            assert_eq!(work.square_roots, 3);
            assert_eq!(work.factorization_operations().unwrap().operations(), 14);
            assert_eq!(work.solve_operations_per_rhs.operations(), 18);
            assert!(work.solve_operations(u64::MAX).is_err());
        }
        let options = DirectOptions::new(FactorKind::Ldlt);
        let symbolic = SymbolicFactorization::analyze::<f64>(
            matrix.as_ref().into(),
            &options,
            SolveControl::NONE,
        )
        .unwrap();
        assert!(matches!(
            symbolic.llt_reference_work(),
            Err(ReferenceWorkError::FactorKind(FactorKind::Ldlt))
        ));
    }

    #[test]
    fn admitted_llt_batches_preserve_buffers_on_refusal_and_count_factor_reuse() {
        use crate::{direct::DirectSolver, SolverError};
        use runmat_simulation_usage::{LedgerError, UsageContext, UsageLedger};
        use std::sync::Arc;
        let count = OperationCount::from_operations;
        let mut entries = TripletBuilder::<f64>::new(3, 3);
        for row in 0..3 {
            for column in 0..3 {
                entries.push(row, column, if row == column { 4.0 } else { 1.0 });
            }
        }
        let matrix = entries.build();
        for workspace in [
            DirectWorkspacePolicy::CallerOwned,
            DirectWorkspacePolicy::BackendManaged,
        ] {
            let ledger = Arc::new(UsageLedger::new(count(64), count(14)).unwrap());
            let context = UsageContext::new(Arc::clone(&ledger));
            context.run_sync(|| {
                let mut solver = DirectSolver::<f64>::new(
                    matrix.as_ref().into(),
                    DirectOptions {
                        workspace,
                        ordering: Ordering::Identity,
                        ..DirectOptions::new(FactorKind::Cholesky)
                    },
                    SolveControl::NONE,
                )
                .unwrap();
                let mut rhs = [6.0; 3];
                assert!(matches!(
                    solver.solve_in_place(&mut rhs, 1),
                    Err(SolverError::Usage(LedgerError::MoreGrantRequired { .. }))
                ));
                assert_eq!(rhs, [6.0; 3]);
                assert!(solver.solve_transpose_in_place(&mut rhs, 1).is_err());
                assert_eq!(rhs, [6.0; 3]);
                assert!(solver
                    .refactorize(matrix.as_ref().into(), SolveControl::NONE)
                    .is_err());
                assert!(matches!(
                    solver.solve_in_place(&mut [], usize::MAX),
                    Err(SolverError::IndexOverflow)
                ));
                assert!(matches!(
                    solver.solve_transpose_in_place(&mut [], usize::MAX),
                    Err(SolverError::IndexOverflow)
                ));
                ledger.update_grant(count(64)).unwrap();
                // The rejected refactorization must retain the existing factor.
                solver.solve_in_place(&mut rhs, 1).unwrap();
                assert!(rhs.iter().all(|value| (value - 1.0).abs() < 1e-12));
                solver
                    .refactorize(matrix.as_ref().into(), SolveControl::NONE)
                    .unwrap();
                rhs.fill(6.0);
                solver.solve_transpose_in_place(&mut rhs, 1).unwrap();
                assert!(rhs.iter().all(|value| (value - 1.0).abs() < 1e-12));
            });
            let usage = ledger.finish().unwrap();
            assert_eq!(usage.factorization, count(28));
            assert_eq!(usage.solve, count(36));
            assert_eq!(usage.total().unwrap(), count(64));
        }
    }
}
