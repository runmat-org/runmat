use super::*;
use crate::csr::{CsrMatrix, TripletBuilder};
use crate::par;
use crate::scalar::{c64, Scalar};
use crate::testutil::poisson3d;

struct Jacobi<T>(Vec<T>);
impl<T: Scalar> Preconditioner<T> for Jacobi<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        for i in 0..r.len() {
            z[i] = r[i] / self.0[i];
        }
        Ok(())
    }
}

fn residual<T: Scalar>(a: &CsrMatrix<T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::zero(); b.len()];
    a.as_ref().spmv(x, &mut ax);
    let r: Vec<T> = b.iter().zip(&ax).map(|(b, a)| *b - *a).collect();
    par::norm2(&r) / par::norm2(b)
}

#[test]
fn pcg_solves_poisson_and_is_thread_count_invariant() {
    let a = poisson3d::<f64>(12, 11, 10);
    let n = a.nrows();
    let b = par::seeded_vector::<f64>(n, 7);
    let jacobi = Jacobi(a.as_ref().diagonal());
    let options = KrylovOptions {
        rtol: 1e-10,
        max_iterations: 500,
        ..Default::default()
    };
    let mut x = vec![0.0; n];
    let report = pcg(&a, Some(&jacobi), &b, &mut x, &options, SolveControl::NONE).unwrap();
    assert!(report.converged(), "{report:?}");
    assert!(residual(&a, &x, &b) <= 1e-10);
    for threads in [2, 4] {
        let mut y = vec![0.0; n];
        let threaded = par::with_threads(threads, || {
            pcg(&a, Some(&jacobi), &b, &mut y, &options, SolveControl::NONE).unwrap()
        });
        assert_eq!(threaded.iterations, report.iterations);
        assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

#[test]
fn minres_solves_symmetric_indefinite() {
    let a = poisson3d::<f64>(8, 8, 8).with_shifted_diagonal(-1.3);
    let n = a.nrows();
    let b = par::seeded_vector::<f64>(n, 3);
    let mut x = vec![0.0; n];
    let options = KrylovOptions {
        rtol: 1e-9,
        max_iterations: 2000,
        ..Default::default()
    };
    let report = minres(&a, None, &b, &mut x, &options, SolveControl::NONE).unwrap();
    assert!(report.converged(), "{report:?}");
    assert!(residual(&a, &x, &b) <= 1e-9);
}

fn convection(n1: usize) -> CsrMatrix<f64> {
    // Nonsymmetric 2D convection–diffusion.
    let n = n1 * n1;
    let mut t = TripletBuilder::new(n, n);
    for j in 0..n1 {
        for i in 0..n1 {
            let r = j * n1 + i;
            t.push(r, r, 4.0);
            if i > 0 {
                t.push(r, r - 1, -1.0 - 0.4);
            }
            if i + 1 < n1 {
                t.push(r, r + 1, -1.0 + 0.4);
            }
            if j > 0 {
                t.push(r, r - n1, -1.0);
            }
            if j + 1 < n1 {
                t.push(r, r + n1, -1.0);
            }
        }
    }
    t.build()
}

#[test]
fn gmres_variants_and_bicgstab_solve_nonsymmetric() {
    let a = convection(30);
    let n = a.nrows();
    let b = par::seeded_vector::<f64>(n, 11);
    let jacobi = Jacobi(a.as_ref().diagonal());
    let options = KrylovOptions {
        rtol: 1e-10,
        max_iterations: 3000,
        ..Default::default()
    };
    for side in [
        PreconditionerSide::Left,
        PreconditionerSide::Right,
        PreconditionerSide::Flexible,
    ] {
        let mut applications = Vec::new();
        for residual_check in [ResidualCheck::EveryStep, ResidualCheck::OnEstimate] {
            let mut x = vec![0.0; n];
            let report = gmres(
                &a,
                Some(&jacobi),
                &b,
                &mut x,
                &options,
                &GmresOptions {
                    restart: 30,
                    side,
                    residual_check,
                },
                SolveControl::NONE,
            )
            .unwrap();
            assert!(report.converged(), "{side:?} {residual_check:?} {report:?}");
            assert!(residual(&a, &x, &b) <= 1e-9, "{side:?} {residual_check:?}");
            if side != PreconditionerSide::Left {
                // Right/flexible certify the true relative residual of the returned iterate.
                assert!(
                    (residual(&a, &x, &b) - report.relative_residual).abs() <= 1e-12,
                    "{side:?} {residual_check:?}"
                );
            }
            applications.push((report.iterations, report.operator_applications));
        }
        // Checking only on the estimate saves the per-step residual product.
        let (every, estimate) = (applications[0], applications[1]);
        assert!(estimate.1 < every.1, "{side:?} {applications:?}");
    }
    let mut x = vec![0.0; n];
    let report = bicgstab(&a, Some(&jacobi), &b, &mut x, &options, SolveControl::NONE).unwrap();
    assert!(report.converged(), "{report:?}");
    assert!(residual(&a, &x, &b) <= 1e-9);
}

#[test]
fn complex_gmres_and_hermitian_pcg() {
    // Complex symmetric Helmholtz-like matrix: Laplacian - k² + i·damping.
    let real = poisson3d::<f64>(7, 7, 7);
    let n = real.nrows();
    let (rows, cols, ptr, idx, vals) = real.into_parts();
    let values: Vec<c64> = vals
        .iter()
        .map(|&v| {
            if v > 0.0 {
                c64::new(v - 0.5, 0.3)
            } else {
                c64::new(v, 0.0)
            }
        })
        .collect();
    let a = CsrMatrix::try_new(rows, cols, ptr, idx, values).unwrap();
    let b = par::seeded_vector::<c64>(n, 5);
    let mut x = vec![c64::zero(); n];
    let options = KrylovOptions {
        rtol: 1e-10,
        max_iterations: 2000,
        ..Default::default()
    };
    let report = gmres(
        &a,
        None,
        &b,
        &mut x,
        &options,
        &GmresOptions::default(),
        SolveControl::NONE,
    )
    .unwrap();
    assert!(report.converged(), "{report:?}");
    assert!(residual(&a, &x, &b) <= 1e-9);
    // Hermitian positive definite: L + i(S - Sᵀ) with small skew part.
    let hermitian = {
        let base = poisson3d::<f64>(6, 6, 6);
        let mut t = TripletBuilder::new(base.nrows(), base.nrows());
        for row in 0..base.nrows() {
            let (c, v) = base.as_ref().row(row);
            for (&col, &val) in c.iter().zip(v) {
                let imag = if col > row {
                    0.2
                } else if col < row {
                    -0.2
                } else {
                    0.0
                };
                t.push(row, col, c64::new(val, imag));
            }
        }
        t.build()
    };
    assert!(hermitian.as_ref().is_symmetric(0.0, true));
    let b = par::seeded_vector::<c64>(hermitian.nrows(), 9);
    let mut x = vec![c64::zero(); hermitian.nrows()];
    let report = pcg(&hermitian, None, &b, &mut x, &options, SolveControl::NONE).unwrap();
    assert!(report.converged(), "{report:?}");
    assert!(residual(&hermitian, &x, &b) <= 1e-9);
}

#[test]
fn cancellation_and_iteration_limits_are_reported() {
    let a = poisson3d::<f64>(10, 10, 10);
    let n = a.nrows();
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    let cancel = || true;
    let control = SolveControl {
        cancelled: Some(&cancel),
        ..SolveControl::NONE
    };
    let error = pcg(&a, None, &b, &mut x, &KrylovOptions::default(), control).unwrap_err();
    assert_eq!(error, SolverError::Cancelled);
    let options = KrylovOptions {
        max_iterations: 3,
        return_best_iterate: true,
        ..Default::default()
    };
    let mut x = vec![0.0; n];
    let report = pcg(&a, None, &b, &mut x, &options, SolveControl::NONE).unwrap();
    assert_eq!(report.flag, StopFlag::MaxIterations);
    assert!((residual(&a, &x, &b) - report.relative_residual).abs() < 1e-12);
}

/// A truncated inner CG: a variable (nonlinear) preconditioner.
struct InnerCg<'a> {
    a: &'a CsrMatrix<f64>,
    jacobi: Jacobi<f64>,
    iterations: usize,
}
impl Preconditioner<f64> for InnerCg<'_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
        z.iter_mut().for_each(|v| *v = 0.0);
        let options = KrylovOptions {
            rtol: 1e-30,
            max_iterations: self.iterations,
            ..Default::default()
        };
        let _ = pcg(
            self.a,
            Some(&self.jacobi),
            r,
            z,
            &options,
            SolveControl::NONE,
        )?;
        Ok(())
    }
    fn is_variable(&self) -> bool {
        true
    }
}

#[test]
fn variable_preconditioners_select_flexible_cg_and_flexible_gmres() {
    let a = poisson3d::<f64>(14, 13, 12);
    let n = a.nrows();
    let b = par::seeded_vector::<f64>(n, 5);
    let inner = InnerCg {
        a: &a,
        jacobi: Jacobi(a.as_ref().diagonal()),
        iterations: 4,
    };
    let options = KrylovOptions {
        rtol: 1e-10,
        max_iterations: 300,
        ..Default::default()
    };
    let mut x = vec![0.0; n];
    let report = pcg(&a, Some(&inner), &b, &mut x, &options, SolveControl::NONE).unwrap();
    assert!(report.converged(), "flexible CG {report:?}");
    assert!(residual(&a, &x, &b) <= 1e-9);
    let mut y = vec![0.0; n];
    let report = gmres(
        &a,
        Some(&inner),
        &b,
        &mut y,
        &options,
        &GmresOptions {
            restart: 30,
            ..Default::default()
        },
        SolveControl::NONE,
    )
    .unwrap();
    assert!(report.converged(), "flexible GMRES {report:?}");
    assert!(residual(&a, &y, &b) <= 1e-9);
}

/// The browser's numerical worker quotes capacity only while no batch is outstanding. Owned PCG
/// releases its right-hand-side norm batch before the first iteration requests capacity.
#[test]
fn owned_pcg_requests_capacity_with_no_batch_outstanding() {
    use runmat_simulation_usage::{
        AsyncCapacityOwner, LedgerError, OperationCount, UsageContext, UsageLedger,
    };
    use std::{future::Future, pin::Pin, sync::Arc};
    struct ExclusiveOwner<'a>(&'a UsageLedger);
    impl AsyncCapacityOwner for ExclusiveOwner<'_> {
        fn ensure_capacity(
            &self,
            additional: OperationCount,
        ) -> Pin<Box<dyn Future<Output = Result<(), LedgerError>> + '_>> {
            let quoted = self.0.exclusive_work_requirement(additional).map(|_| ());
            Box::pin(async move { quoted })
        }
    }
    let a = poisson3d::<f64>(6, 5, 4);
    let n = a.nrows();
    let b = par::seeded_vector::<f64>(n, 7);
    let options = KrylovOptions {
        rtol: 1e-10,
        max_iterations: 400,
        ..Default::default()
    };
    let mut reference = vec![0.0; n];
    let jacobi = crate::precond::Jacobi::new(a.as_ref()).unwrap();
    let expected = pcg(
        &a,
        Some(&jacobi),
        &b,
        &mut reference,
        &options,
        SolveControl::NONE,
    )
    .unwrap();
    let limit = OperationCount::from_operations(1 << 40);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    let owner = ExclusiveOwner(&ledger);
    let metered = crate::operator::MeteredCsrOperator::new(
        a.as_ref(),
        &ledger,
        runmat_simulation_usage::UsageCategory::Solve,
    )
    .unwrap();
    let mut x = vec![0.0; n];
    let report = context.run_sync(|| {
        let mut future = std::pin::pin!(pcg_solve_with_capacity(
            &metered,
            Some(&jacobi),
            &b,
            &mut x,
            &options,
            SolveControl::NONE,
            &owner,
        ));
        let mut waker = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(result) = future.as_mut().poll(&mut waker) {
                break result;
            }
        }
    });
    let report = match report {
        Ok(report) => report,
        Err(PcgCapacityError::Admission(cause)) => panic!("capacity refused: {cause:?}"),
        Err(PcgCapacityError::Numerical(error)) => panic!("{error:?}"),
    };
    assert!(report.converged(), "{report:?}");
    assert_eq!(report.iterations, expected.iterations);
    assert!(residual(&a, &x, &b) <= 1e-10);
    let snapshot = ledger.snapshot().unwrap();
    assert!(snapshot.completed.solve > OperationCount::ZERO);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}
