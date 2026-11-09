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
    for kind in [FactorKind::Cholesky, FactorKind::Ldlt] {
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
                        if kind == FactorKind::Cholesky {
                            solve += if complex { 2 } else { 1 };
                        }
                        for _ in 0..trailing {
                            solve += if complex { 8 } else { 2 };
                        }
                    }
                    if kind == FactorKind::Ldlt {
                        solve += if complex { 2 } else { 1 };
                    }
                }
                let measured = hermitian_columns(heights.iter().copied(), complex, kind).unwrap();
                let roots = if kind == FactorKind::Cholesky {
                    heights.len() as u64
                } else {
                    0
                };
                assert_eq!(measured.basic_operations.operations(), basic);
                assert_eq!(measured.square_roots, roots);
                assert_eq!(
                    measured.factorization_operations().unwrap().operations(),
                    basic + roots
                );
                assert_eq!(measured.solve_operations_per_rhs.operations(), solve);
                assert_eq!(measured.solve_operations(0).unwrap(), OperationCount::ZERO);
                assert_eq!(
                    measured.solve_operations(3).unwrap().operations(),
                    3 * solve
                );
            }
        }
    }
    assert!(matches!(
        hermitian_columns([0].into_iter(), false, FactorKind::Cholesky),
        Err(ReferenceWorkError::InvalidColumn)
    ));
    assert!(DirectReferenceWork {
        basic_operations: OperationCount::from_operations(u64::MAX),
        square_roots: 1,
        ..Default::default()
    }
    .factorization_operations()
    .is_err());
    if usize::BITS == 64 {
        assert!(matches!(
            hermitian_columns([usize::MAX].into_iter(), true, FactorKind::Ldlt),
            Err(ReferenceWorkError::Overflow(_))
        ));
    }
}

#[test]
fn symbolic_factors_use_stored_fill_for_every_direct_family() {
    let mut entries = TripletBuilder::<f64>::new(3, 3);
    for row in 0..3 {
        for column in 0..3 {
            entries.push(row, column, if row == column { 4.0 } else { 1.0 });
        }
    }
    let matrix = entries.build();
    for kind in [FactorKind::Cholesky, FactorKind::Ldlt] {
        for workspace in [
            DirectWorkspacePolicy::CallerOwned,
            DirectWorkspacePolicy::BackendManaged,
        ] {
            let options = DirectOptions {
                ordering: Ordering::Identity,
                workspace,
                ..DirectOptions::new(kind)
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
            let work = symbolic.reference_work().unwrap();
            assert_eq!(work.basic_operations.operations(), 11);
            let (roots, factor, solve) = match kind {
                FactorKind::Cholesky => (3, 14, 18),
                FactorKind::Ldlt => (0, 11, 15),
                _ => unreachable!(),
            };
            assert_eq!(work.square_roots, roots);
            assert_eq!(
                work.factorization_operations().unwrap().operations(),
                factor
            );
            assert_eq!(work.solve_operations_per_rhs.operations(), solve);
            assert!(work.solve_operations(u64::MAX).is_err());
        }
    }
    let options = DirectOptions::new(FactorKind::Lu);
    let symbolic =
        SymbolicFactorization::analyze::<f64>(matrix.as_ref().into(), &options, SolveControl::NONE)
            .unwrap();
    // Dense 3x3: symmetrized fill columns of heights 3, 2, 1.
    let work = symbolic.reference_work().unwrap();
    assert_eq!(work.square_roots, 0);
    assert_eq!(work.factorization_operations().unwrap().operations(), 16);
    assert_eq!(work.solve_operations_per_rhs.operations(), 15);
    assert_eq!(
        symbolic.measurement_coverage().unwrap(),
        DirectMeasurementCoverage::Measured(work)
    );
}

#[test]
fn lu_reference_counts_pivot_scaling_and_square_trailing_updates() {
    for complex in [false, true] {
        let (pivot, scale, update) = if complex { (6, 6, 8) } else { (1, 1, 2) };
        for heights in [vec![], vec![1], vec![3, 2, 1], vec![2, 2, 2, 1]] {
            let mut factor = 0u64;
            let mut solve = 0u64;
            for &height in &heights {
                let p = height - 1;
                factor += pivot;
                for _ in 0..p {
                    factor += scale;
                }
                for _row in 0..p {
                    for _column in 0..p {
                        factor += update;
                    }
                }
                for _triangle in 0..2 {
                    for _ in 0..p {
                        solve += update;
                    }
                }
                solve += scale;
            }
            let measured = general_columns(heights.iter().copied(), complex).unwrap();
            assert_eq!(measured.square_roots, 0);
            assert_eq!(
                measured.factorization_operations().unwrap().operations(),
                factor
            );
            assert_eq!(measured.solve_operations_per_rhs.operations(), solve);
        }
    }
    assert!(matches!(
        general_columns([0].into_iter(), false),
        Err(ReferenceWorkError::InvalidColumn)
    ));
}

#[test]
fn excluded_sparse_pivot_kernels_keep_refined_execution_and_covered_batches() {
    use crate::direct::{DirectSolvePhase, DirectSolver};
    use runmat_simulation_usage::{UsageContext, UsageLedger};
    use std::sync::Arc;

    let count = OperationCount::from_operations;
    let mut entries = TripletBuilder::<f64>::new(3, 3);
    for row in 0..3 {
        for column in 0..3 {
            entries.push(row, column, if row == column { 4.0 } else { 1.0 });
        }
    }
    let matrix = entries.build();
    for (kind, exclusion) in [(
        FactorKind::BunchKaufman,
        ExcludedDirectKernel::SparseBunchKaufman,
    )] {
        let ledger = Arc::new(UsageLedger::new(count(100_000), count(100_000)).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            let solver = DirectSolver::<f64>::new(
                matrix.as_ref().into(),
                DirectOptions {
                    ordering: Ordering::Identity,
                    ..DirectOptions::new(kind)
                },
                SolveControl::NONE,
            )
            .unwrap();
            assert_eq!(
                solver.symbolic().measurement_coverage().unwrap(),
                DirectMeasurementCoverage::Excluded(exclusion)
            );
            let mut x = [0.0; 3];
            let mut phases = Vec::new();
            {
                let future = solver.solve_refined_with_owner(
                    matrix.as_ref().into(),
                    &[6.0; 3],
                    &mut x,
                    1,
                    |phase, operations| {
                        phases.push((phase, operations));
                        std::future::ready(Ok::<(), ()>(()))
                    },
                );
                let mut future = std::pin::pin!(future);
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                match std::future::Future::poll(future.as_mut(), &mut task) {
                    std::task::Poll::Ready(Ok(report)) => report,
                    std::task::Poll::Ready(Err(_)) => panic!("excluded direct solve failed"),
                    std::task::Poll::Pending => panic!("ready direct owner suspended"),
                };
            }
            assert!(x.iter().all(|value| (value - 1.0).abs() < 1e-12));
            assert!(phases
                .iter()
                .any(|(phase, _)| *phase == DirectSolvePhase::Residual));
            assert!(phases
                .iter()
                .any(|(phase, _)| *phase == DirectSolvePhase::Norm));
            assert!(phases.iter().all(|(phase, _)| !matches!(
                phase,
                DirectSolvePhase::InitialSubstitution | DirectSolvePhase::RefinementSubstitution
            )));
        });
        let usage = ledger.finish().unwrap();
        assert_eq!(usage.factorization, OperationCount::ZERO);
        assert!(usage.solve.operations() > 0);
    }
}

#[test]
fn admitted_direct_batches_preserve_buffers_on_refusal_and_count_factor_reuse() {
    exercise_admitted_batches::<f32>();
    exercise_admitted_batches::<f64>();
    exercise_admitted_batches::<crate::c32>();
    exercise_admitted_batches::<crate::c64>();
}

fn exercise_admitted_batches<T: super::super::FactorScalar>() {
    use crate::{direct::DirectSolver, SolverError};
    use runmat_simulation_usage::{LedgerError, UsageContext, UsageLedger};
    use std::sync::Arc;
    let count = OperationCount::from_operations;
    let mut entries = TripletBuilder::<T>::new(3, 3);
    for row in 0..3 {
        for column in 0..3 {
            entries.push(
                row,
                column,
                T::from_parts(if row == column { 4.0 } else { 1.0 }, 0.0),
            );
        }
    }
    let matrix = entries.build();
    for kind in [FactorKind::Cholesky, FactorKind::Ldlt] {
        let (factor, solve) = match (kind, T::IS_COMPLEX) {
            (FactorKind::Cholesky, false) => (14, 18),
            (FactorKind::Ldlt, false) => (11, 15),
            (FactorKind::Cholesky, true) => (29, 60),
            (FactorKind::Ldlt, true) => (26, 54),
            _ => unreachable!(),
        };
        let total = 2 * (factor + solve);
        for workspace in [
            DirectWorkspacePolicy::CallerOwned,
            DirectWorkspacePolicy::BackendManaged,
        ] {
            let ledger = Arc::new(UsageLedger::new(count(total), count(factor)).unwrap());
            let context = UsageContext::new(Arc::clone(&ledger));
            context.run_sync(|| {
                let mut solver = DirectSolver::<T>::new(
                    matrix.as_ref().into(),
                    DirectOptions {
                        workspace,
                        ordering: Ordering::Identity,
                        ..DirectOptions::new(kind)
                    },
                    SolveControl::NONE,
                )
                .unwrap();
                let initial = T::from_parts(6.0, if T::IS_COMPLEX { 6.0 } else { 0.0 });
                let expected = T::from_parts(1.0, if T::IS_COMPLEX { 1.0 } else { 0.0 });
                let mut rhs = [initial; 3];
                assert!(matches!(
                    solver.solve_in_place(&mut rhs, 1),
                    Err(SolverError::Usage(LedgerError::MoreGrantRequired { .. }))
                ));
                assert_eq!(rhs, [initial; 3]);
                assert!(solver.solve_transpose_in_place(&mut rhs, 1).is_err());
                assert_eq!(rhs, [initial; 3]);
                let mut refined = [T::zero(); 3];
                assert!(matches!(
                    solver.solve_refined(matrix.as_ref().into(), &rhs, &mut refined, 1),
                    Err(SolverError::Usage(LedgerError::MoreGrantRequired { .. }))
                ));
                assert_eq!(refined, [T::zero(); 3]);
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
                assert!(matches!(
                    solver.solve_refined(matrix.as_ref().into(), &[], &mut [], usize::MAX),
                    Err(SolverError::IndexOverflow)
                ));
                ledger.update_grant(count(total)).unwrap();
                // A refused refactorization retains the previously computed factor.
                solver.solve_in_place(&mut rhs, 1).unwrap();
                assert!(rhs
                    .iter()
                    .all(|&value| crate::Scalar::abs(value - expected) < 32.0 * T::EPSILON));
                solver
                    .refactorize(matrix.as_ref().into(), SolveControl::NONE)
                    .unwrap();
                rhs.fill(initial);
                solver.solve_transpose_in_place(&mut rhs, 1).unwrap();
                assert!(rhs
                    .iter()
                    .all(|&value| crate::Scalar::abs(value - expected) < 32.0 * T::EPSILON));
            });
            let usage = ledger.finish().unwrap();
            assert_eq!(usage.factorization, count(2 * factor));
            assert_eq!(usage.solve, count(2 * solve));
            assert_eq!(usage.total().unwrap(), count(total));
        }
    }
}
