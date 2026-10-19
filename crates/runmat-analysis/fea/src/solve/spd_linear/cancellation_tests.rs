use super::*;
use crate::{
    algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseMatrix, ScopedSpmvError,
    },
    planning::resource::LocalExecutionPlan,
    progress::{replace_fea_progress_context, FeaCancellationToken, FeaProgressContext},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

struct CancellingIdentity {
    cancellation: FeaCancellationToken,
    calls: AtomicUsize,
    cancel_on_call: usize,
}

impl SpdPreconditioner for CancellingIdentity {
    fn kind(&self) -> SpdPreconditionerKind {
        SpdPreconditionerKind::Jacobi
    }

    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        if self.calls.fetch_add(1, Ordering::SeqCst) + 1 == self.cancel_on_call {
            self.cancellation.cancel();
        }
        residual.to_vec()
    }

    fn apply_with_execution(
        &self,
        residual: &[f64],
        _plan: LocalExecutionPlan,
    ) -> Result<Vec<f64>, ScopedSpmvError> {
        Ok(self.apply(residual))
    }
}

fn system() -> AssembledRealSystem {
    let matrix =
        |values, role, properties| RealSparseMatrix::diagonal(values, role, properties).unwrap();
    let operators = CanonicalRealOperators::try_new(
        matrix(
            vec![2., 4.],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        ),
        matrix(
            vec![1.; 2],
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        ),
        matrix(
            vec![0.; 2],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        ),
    )
    .unwrap();
    AssembledRealSystem::try_new(operators, vec![false; 2], vec![0.; 2]).unwrap()
}

fn solve(preconditioner: &CancellingIdentity) -> SpdSolveAttempt {
    solve_spd_system_cg(
        &system(),
        &[1., 2.],
        SpdCgSolveOptions {
            max_iters: 8,
            tol: 1.0e-12,
            prepared_accelerator_system: None,
            cpu_preconditioner: Some(preconditioner),
            initial_guess: None,
        },
    )
}

#[test]
fn cancelled_admission_performs_no_preconditioner_work() {
    let cancellation = FeaCancellationToken::new();
    cancellation.cancel();
    let _context = replace_fea_progress_context(FeaProgressContext {
        cancellation: cancellation.clone(),
        ..Default::default()
    });
    let preconditioner = CancellingIdentity {
        cancellation,
        calls: AtomicUsize::new(0),
        cancel_on_call: usize::MAX,
    };
    let result = solve(&preconditioner);
    assert!(!result.converged);
    assert_eq!(result.vector, vec![0.; 2]);
    assert_eq!(preconditioner.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_between_iterations_preserves_the_last_iterate_without_success() {
    let cancellation = FeaCancellationToken::new();
    let context = replace_fea_progress_context(FeaProgressContext {
        cancellation: cancellation.clone(),
        ..Default::default()
    });
    let preconditioner = CancellingIdentity {
        cancellation,
        calls: AtomicUsize::new(0),
        cancel_on_call: 2,
    };
    let cancelled = solve(&preconditioner);
    assert!(!cancelled.converged);
    assert_eq!(cancelled.vector, vec![5. / 18., 10. / 18.]);
    assert_eq!(preconditioner.calls.load(Ordering::SeqCst), 2);
    drop(context);
    let resumed = solve(&preconditioner);
    assert!(resumed.converged);
    assert!((resumed.vector[0] - 0.5).abs() < 1.0e-12);
    assert!((resumed.vector[1] - 0.5).abs() < 1.0e-12);
}

#[test]
fn cancellation_from_progress_cannot_turn_a_finished_iterate_into_success() {
    for cancel_at in [
        crate::progress::FeaProgressStatus::Advanced,
        crate::progress::FeaProgressStatus::Completed,
    ] {
        let cancellation = FeaCancellationToken::new();
        let statuses = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&statuses);
        let notified_cancellation = cancellation.clone();
        let _context = replace_fea_progress_context(FeaProgressContext {
            cancellation: cancellation.clone(),
            handler: Some(Arc::new(move |event| {
                if event.status == cancel_at {
                    notified_cancellation.cancel();
                }
                recorded.lock().unwrap().push(event.status);
            })),
            ..Default::default()
        });
        let preconditioner = CancellingIdentity {
            cancellation,
            calls: AtomicUsize::new(0),
            cancel_on_call: usize::MAX,
        };
        let result = solve_spd_system_cg(
            &system(),
            &[1., 0.],
            SpdCgSolveOptions {
                max_iters: 8,
                tol: 1.0e-12,
                prepared_accelerator_system: None,
                cpu_preconditioner: Some(&preconditioner),
                initial_guess: None,
            },
        );
        assert_eq!(result.vector, vec![0.5, 0.]);
        assert!(!result.converged);
        assert_eq!(
            *statuses.lock().unwrap(),
            vec![
                crate::progress::FeaProgressStatus::Advanced,
                if cancel_at == crate::progress::FeaProgressStatus::Advanced {
                    crate::progress::FeaProgressStatus::Cancelled
                } else {
                    crate::progress::FeaProgressStatus::Completed
                },
            ]
        );
    }
}
