use super::*;
use futures::executor::block_on;

struct CountingEvaluator {
    calls: usize,
}

impl LeastSquaresEvaluator for CountingEvaluator {
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        self.calls += 1;
        let residual = vec![x[0] - 1.0];
        Box::pin(async move { Ok(residual) })
    }
}

fn options(max_fun_evals: usize) -> LeastSquaresOptions {
    LeastSquaresOptions {
        criteria: None,
        tol_x: 1.0e-12,
        tol_fun: 1.0e-12,
        max_iter: 10,
        max_fun_evals,
        final_gradient: false,
        final_jacobian: false,
        scaling: Default::default(),
    }
}

#[test]
fn configured_axis_probes_use_signed_actual_steps_and_the_original_hard_budget() {
    struct Probes(Vec<Vec<f64>>);
    impl LeastSquaresEvaluator for Probes {
        fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
            self.0.push(x.to_vec());
            let f = x.iter().map(|x| x * x).sum();
            Box::pin(async move { Ok(vec![f]) })
        }
    }
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert(
        "FinDiffRelStep",
        runmat_value::Value::Tensor(runmat_value::Tensor::new(vec![0.1, 0.2], vec![1, 2]).unwrap()),
    );
    fields.insert(
        "TypicalX",
        runmat_value::Value::Tensor(runmat_value::Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
    );
    let fields = fields.finish();
    let scaling =
        crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields(
            "fsolve",
            &crate::builtins::math::optim::common::ScalarOptionFields::from_struct(Some(&fields)),
            true,
        )
        .unwrap();
    let mut evaluator = Probes(Vec::new());
    let mut calls = 1;
    let measured = block_on(finite_difference_jacobian(
        "fsolve",
        &mut evaluator,
        &[-1.0, 0.0],
        &LeastSquaresBounds::unbounded(2),
        &[1.0],
        &mut calls,
        3,
        &scaling,
    ))
    .unwrap();
    assert_eq!(calls, 3);
    assert_eq!(evaluator.0.len(), 2);
    assert!((evaluator.0[0][0] + 1.2).abs() < 1e-14);
    assert!((evaluator.0[1][1] - 0.6).abs() < 1e-14);
    assert!((measured.values[0] + 2.2).abs() < 1e-12);
    assert!((measured.values[1] - 0.6).abs() < 1e-12);
    let mut calls = 1;
    let partial = block_on(finite_difference_jacobian(
        "fsolve",
        &mut evaluator,
        &[-1.0, 0.0],
        &LeastSquaresBounds::unbounded(2),
        &[1.0],
        &mut calls,
        2,
        &scaling,
    ))
    .unwrap();
    assert_eq!(calls, 2);
    assert!(partial.exhausted_budget);
    assert!(partial.values[1].is_nan());
    let mut narrow = Probes(Vec::new());
    let mut calls = 1;
    let measured = block_on(finite_difference_jacobian(
        "fsolve",
        &mut narrow,
        &[5e-13],
        &LeastSquaresBounds {
            lower: vec![0.],
            upper: vec![1e-12],
        },
        &[2.5e-25],
        &mut calls,
        2,
        &Default::default(),
    ))
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(narrow.0.len(), 1);
    assert!(narrow.0[0][0] >= 0. && narrow.0[0][0] <= 1e-12);
    assert!(!measured.exhausted_budget && measured.values[0] > 0.);
}

#[test]
fn bounds_reject_infinite_projected_iterates() {
    let bounds = LeastSquaresBounds {
        lower: vec![f64::INFINITY],
        upper: vec![f64::INFINITY],
    };
    let err = bounds
        .validate("lsqcurvefit", 1)
        .expect_err("invalid bounds");
    assert!(err.message().contains("finite feasible iterate"));
}

#[test]
fn least_squares_workspace_guard_rejects_oversized_problem() {
    let err = checked_least_squares_work("lsqnonlin", MAX_LEAST_SQUARES_WORK_ELEMENTS, 2)
        .expect_err("oversized workspace");
    assert!(err.message().contains("workspace"));
}

#[test]
fn solve_stops_before_extra_callback_after_budget_exhausted() {
    let mut evaluator = CountingEvaluator { calls: 0 };
    let bounds = LeastSquaresBounds::unbounded(2);
    let result = block_on(solve_least_squares(
        "lsqcurvefit",
        &mut evaluator,
        vec![0.0, 0.0],
        &bounds,
        &options(2),
    ))
    .expect("result");

    assert_eq!(result.func_count, 2);
    assert_eq!(evaluator.calls, 2);
    reflective_controls();
    assert_eq!(result.exitflag, 0);
    let mut never_called = CountingEvaluator { calls: 0 };
    let mut zero = options(0);
    zero.max_iter = 0;
    zero.criteria = Some(Criteria {
        function: 0.,
        optimality: 0.,
        step: 0.,
    });
    let stopped = block_on(solve_selected(
        "lsqnonlin",
        &mut never_called,
        vec![3.],
        &LeastSquaresBounds {
            lower: vec![0.],
            upper: vec![2.],
        },
        &zero,
        Method::Reflective,
        runmat_builtins::OptimDisplay::Off,
    ))
    .unwrap();
    assert_eq!(never_called.calls, 0);
    assert_eq!(stopped.func_count, 0);
    assert_eq!(stopped.iterations, 0);
    assert_eq!(stopped.x, vec![2.]);
    assert_eq!(stopped.point_evidence, PointEvidence::Supplied);
    assert_eq!(
        stopped.method,
        Method::NotStarted(runmat_builtins::OptimAlgorithm::TrustRegionReflective)
    );
    assert_eq!(stopped.exitflag, 0);
    assert!(stopped.resnorm.is_nan() && stopped.first_order_optimality.is_nan());
    assert!(stopped.residual.is_empty());
    assert!(stopped.jacobian.row_major().unwrap().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn already_zero_residual_does_not_hide_final_jacobian_callback_failure() {
    struct FinalFailure {
        calls: usize,
    }
    impl LeastSquaresEvaluator for FinalFailure {
        fn residual<'a>(&'a mut self, _: &'a [f64]) -> ResidualFuture<'a> {
            self.calls += 1;
            let first = self.calls == 1;
            Box::pin(async move {
                if first {
                    Ok(vec![0.0])
                } else {
                    Err(crate::build_runtime_error("original callback failure")
                        .with_identifier("RunMat:test:OriginalCallback")
                        .with_source(std::io::Error::other("original cause"))
                        .build())
                }
            })
        }
    }
    let mut evaluator = FinalFailure { calls: 0 };
    let mut options = options(10);
    // Exercise the requested terminal-derivative boundary, not the valid
    // fast path that omits an unused final Jacobian.
    options.final_jacobian = true;
    let error = block_on(solve_least_squares(
        "fsolve",
        &mut evaluator,
        vec![0.0],
        &LeastSquaresBounds::unbounded(1),
        &options,
    ))
    .expect_err("not a synthetic zero Jacobian");
    assert_eq!(error.identifier(), Some("RunMat:test:OriginalCallback"));
    assert_eq!(error.source.as_ref().unwrap().to_string(), "original cause");
    assert_eq!(evaluator.calls, 2);
}
fn reflective_controls() {
    // A free component of the same 3-by-2 original residual matrix can
    // cancel although its absolute data contributions remain nonzero.
    let x = [0.5f64, 3.75];
    let residual = [x[0] - 2., x[1] - 3., x[0] + x[1] - 5.];
    let direction = [0.04f64, (-0.02f64).next_up()];
    let distance = x[1].sqrt();
    let g = distance * (residual[1] + residual[2]);
    let h = distance * (direction[1] + (direction[0] + direction[1]));
    let scale =
        crate::builtins::math::optim::common::trust_quadratic::residual::StationarityScale {
            rhs: distance * (residual[1].abs() + residual[2].abs()),
            curvature: distance * (direction[1].abs() + (direction[0] + direction[1]).abs()),
            diagonal: 0.,
        };
    assert_ne!(h, 0., "the component has actual floating formation error");
    assert!(scale.qualifies(g, h, 0., 2));
    assert!(
        !scale.qualifies(g, h + 1e-6, 0., 2),
        "a materially wrong product still fails"
    );
    assert!(
        !crate::builtins::math::optim::common::trust_quadratic::residual::StationarityScale {
            rhs: f64::INFINITY,
            curvature: 0.,
            diagonal: 0.,
        }
        .qualifies(g, h, 0., 2),
        "infinite bounds cannot erase residuals"
    );
    struct Residual {
        cooperation: ScalarCooperation,
        calls: usize,
        accepted: usize,
        target: f64,
        repeats: usize,
    }
    impl LeastSquaresEvaluator for Residual {
        fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
            self.calls += 1;
            let value = x.iter().sum::<f64>() - self.target;
            let repeats = self.repeats;
            Box::pin(async move { Ok(vec![value; repeats]) })
        }
        fn cooperation(&mut self) -> Option<&mut ScalarCooperation> {
            Some(&mut self.cooperation)
        }
        fn accept_residual(&mut self) {
            self.accepted += 1;
        }
    }
    let mut evaluator = Residual {
        cooperation: ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqnonlin, None),
        calls: 0,
        accepted: 0,
        target: 100.,
        repeats: 1,
    };
    let mut controls = options(3);
    controls.max_iter = 1;
    let result = block_on(solve_selected(
        "lsqnonlin",
        &mut evaluator,
        vec![0.],
        &LeastSquaresBounds::unbounded(1),
        &controls,
        Method::Reflective,
        runmat_builtins::OptimDisplay::Off,
    ))
    .unwrap();
    assert_eq!(result.method, Method::Reflective);
    assert!(
        result.x[0] > 0.9 && result.x[0] <= 1. + 1e-12,
        "actual radius-one step, not a damped LM alias"
    );
    assert_eq!(result.func_count, 3);
    assert_eq!(evaluator.calls, 3);
    assert_eq!(evaluator.accepted, 2);
    // At this accepted point the distance-scaled Jacobian is tiny relative
    // to its unchanged Coleman--Li diagonal. Preserve the strict original
    // normal-residual check without losing the small projected QR RHS.
    let mut bounded = Residual {
        cooperation: ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqnonlin, None),
        calls: 0,
        accepted: 0,
        target: 2.,
        repeats: 2,
    };
    let bounded_result = block_on(solve_selected(
        "lsqnonlin",
        &mut bounded,
        vec![0.9999847409781033],
        &LeastSquaresBounds {
            lower: vec![-1.],
            upper: vec![1.],
        },
        &controls,
        Method::Reflective,
        runmat_builtins::OptimDisplay::Off,
    ))
    .expect("near-active-bound rectangular factorization");
    assert!(bounded_result.x[0] > 1. - 1e-8 && bounded_result.x[0] < 1.);
    assert_eq!(bounded_result.func_count, bounded.calls);
    assert_eq!(bounded.calls, 3);
    assert_eq!(bounded.accepted, 2);
    let mut evaluator = Residual {
        cooperation: ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqnonlin, None),
        calls: 0,
        accepted: 0,
        target: 100.,
        repeats: 1,
    };
    controls.max_iter = 0;
    controls.max_fun_evals = 1;
    let result = block_on(solve_selected(
        "lsqnonlin",
        &mut evaluator,
        vec![0., 0.],
        &LeastSquaresBounds::unbounded(2),
        &controls,
        Method::Reflective,
        runmat_builtins::OptimDisplay::Off,
    ))
    .unwrap();
    assert_eq!(result.method, Method::LevenbergMarquardt);
    assert_eq!(
        evaluator.calls, 1,
        "fallback consumes the accepted first residual without replay"
    );
    assert_eq!(evaluator.accepted, 1);
    let mut evaluator = Residual {
        cooperation: ScalarCooperation::new(runmat_builtins::OptimSolver::Lsqnonlin, None),
        calls: 0,
        accepted: 0,
        target: 100.,
        repeats: 1,
    };
    let result = block_on(solve_selected(
        "lsqnonlin",
        &mut evaluator,
        vec![1.],
        &LeastSquaresBounds {
            lower: vec![1.],
            upper: vec![1.],
        },
        &controls,
        Method::Reflective,
        runmat_builtins::OptimDisplay::Off,
    ))
    .unwrap();
    assert_eq!(evaluator.calls, 1);
    assert_eq!(result.exitflag, 1);
    assert_eq!(result.first_order_optimality, 0.);
    assert!(
        result
            .jacobian
            .row_major()
            .expect("finite-difference workspace")
            .iter()
            .all(|v| v.is_nan()),
        "no invented fixed-coordinate derivative"
    );
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let mut evaluator = Residual {
        cooperation: ScalarCooperation::new(
            runmat_builtins::OptimSolver::Lsqnonlin,
            Some(&runtime),
        ),
        calls: 0,
        accepted: 0,
        target: 100.,
        repeats: 1,
    };
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(solve_selected(
        "lsqnonlin",
        &mut evaluator,
        vec![0.],
        &LeastSquaresBounds::unbounded(1),
        &controls,
        Method::Reflective,
        runmat_builtins::OptimDisplay::Off,
    ))
    .unwrap_err();
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(
        evaluator.calls, 0,
        "original cancellation precedes the first callback"
    );
}
