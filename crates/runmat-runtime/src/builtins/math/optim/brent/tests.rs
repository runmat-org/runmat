use super::*;

#[test]
fn active_brent_iteration_observes_invocation_cancellation_before_callback() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ScalarCooperation::with_budget(OptimSolver::Fzero, budget);
    let error = futures::executor::block_on(brent_zero(
        "fzero",
        &mut cooperation,
        &Value::FunctionHandle("sin".into()),
        BrentZeroBracket {
            a: -1.0,
            b: 1.0,
            fa: -1.0,
            fb: 1.0,
            evals: 2,
        },
        BrentParams {
            tol_x: 1.0e-8,
            max_iter: 8,
            max_fun_evals: 8,
        },
        false,
        None,
    ))
    .expect_err("active iteration must observe cancellation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn tiny_opposite_sign_values_remain_a_valid_bracket() {
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let positive = f64::MIN_POSITIVE;
    let negative = -positive;
    assert_eq!(positive * negative, -0.0);
    assert_ne!(positive.signum(), negative.signum());
    let result = futures::executor::block_on(brent_zero(
        "fzero",
        &mut cooperation,
        &Value::FunctionHandle("sin".into()),
        BrentZeroBracket {
            a: -1.0,
            b: 1.0,
            fa: negative,
            fb: positive,
            evals: 2,
        },
        BrentParams {
            tol_x: 1.0e-6,
            max_iter: 8,
            max_fun_evals: 8,
        },
        false,
        None,
    ));
    assert!(result.is_ok(), "tiny signed values form a valid bracket");
    struct NoReplay;
    impl ZeroFunction for NoReplay {
        type Receipt = std::rc::Rc<(f64, f64)>;
        async fn evaluate(
            &mut self,
            _: f64,
            _: &mut ScalarCooperation,
        ) -> BuiltinResult<(f64, Self::Receipt)> {
            panic!("exhausted bracket must return its active receipt without evaluation")
        }
    }
    let selected = futures::executor::block_on(zero_with_receipts(
        "fzero",
        &mut cooperation,
        NoReplay,
        BrentZeroBracket {
            a: -1.,
            b: 2.,
            fa: -1.,
            fb: 2.,
            evals: 2,
        },
        (std::rc::Rc::new((-1., -1.)), std::rc::Rc::new((2., 2.))),
        BrentParams {
            tol_x: 1e-8,
            max_iter: 4,
            max_fun_evals: 2,
        },
        None,
    ))
    .unwrap();
    assert_eq!(selected.result.x, -1.);
    assert_eq!(selected.result.func_count, 2);
    assert_eq!(*selected.receipt, (selected.result.x, selected.result.fval));
    assert_eq!(std::rc::Rc::strong_count(&selected.receipt), 1);
}

#[test]
fn opposite_extreme_finite_root_bracket_keeps_midpoint_finite() {
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let result = futures::executor::block_on(brent_zero(
        "fzero",
        &mut cooperation,
        &Value::FunctionHandle("sign".into()),
        BrentZeroBracket {
            a: -f64::MAX,
            b: f64::MAX,
            fa: -1.0,
            fb: 1.0,
            evals: 2,
        },
        BrentParams {
            tol_x: 1.0e-8,
            max_iter: 16,
            max_fun_evals: 16,
        },
        false,
        None,
    ))
    .unwrap();
    assert!(result.converged);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.fval, 0.0);
}

#[test]
fn bounded_minimum_on_opposite_extreme_finite_bounds_stays_finite() {
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fminbnd, None);
    let result = futures::executor::block_on(brent_min(
        "fminbnd",
        &mut cooperation,
        &Value::FunctionHandle("abs".into()),
        -f64::MAX,
        f64::MAX,
        BrentParams {
            tol_x: 1.0e-4,
            max_iter: 2000,
            max_fun_evals: 2000,
        },
        None,
    ))
    .unwrap();
    assert!(result.converged);
    assert!(result.x.is_finite());
    assert!(result.fval.is_finite());
    assert!(result.fval <= 1.0e-4, "{}", result.fval);
}

#[test]
fn large_same_sign_finite_bounds_do_not_overflow_tolerance_or_stop_after_one_sample() {
    let lower = 0.5 * f64::MAX;
    let upper = 0.75 * f64::MAX;
    let params = BrentParams {
        tol_x: 1.0e-4,
        max_iter: 2000,
        max_fun_evals: 2000,
    };
    assert!(brent_min_tolerance(upper, params).is_finite());
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fminbnd, None);
    let result = futures::executor::block_on(brent_min(
        "fminbnd",
        &mut cooperation,
        &Value::FunctionHandle("abs".into()),
        lower,
        upper,
        params,
        None,
    ))
    .unwrap();
    assert!(result.converged);
    assert!(
        result.func_count > 1,
        "the initial golden point is not the minimum"
    );
    assert!(result.x.is_finite() && result.fval.is_finite());
    assert!(result.x >= lower && result.x <= lower * (1.0 + 1.0e-6));
    assert_eq!(result.fval, result.x);
}

#[test]
fn interpolation_acceptance_uses_signed_q() {
    assert!(!interpolation_step_accepted(1.0, -2.0, 1.0, 0.1, 10.0));
    assert!(interpolation_step_accepted(1.0, -2.0, -1.0, 0.1, 10.0));
}

#[test]
fn brent_zero_rejects_nonzero_collapsed_bracket() {
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let err = futures::executor::block_on(brent_zero(
        "fzero",
        &mut cooperation,
        &Value::FunctionHandle("sin".into()),
        BrentZeroBracket {
            a: 1.0,
            b: 1.0,
            fa: 1.0,
            fb: 1.0,
            evals: 0,
        },
        BrentParams {
            tol_x: 1.0e-6,
            max_iter: 10,
            max_fun_evals: 10,
        },
        false,
        None,
    ))
    .unwrap_err();
    assert!(err.message().contains("invalid bracket"));
}

#[test]
fn with_sign_matches_fortran_semantics() {
    assert_eq!(with_sign(1.0, 0.5), 1.0);
    assert_eq!(with_sign(1.0, -0.5), -1.0);
    assert_eq!(with_sign(1.0, 0.0), 1.0);
    assert_eq!(with_sign(-1.0, -1.0), -1.0);
}
