use super::*;
use futures::executor::block_on;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum Rhs {
    Growth,
    Quartic,
}

fn with_rhs(
    kind: Rhs,
    test: impl FnOnce(&crate::context::RuntimeContext, &Value, &Arc<AtomicUsize>),
) {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("refine_rhs", 975)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested| {
            assert_eq!(function, 975);
            assert_eq!(requested, 1);
            observed.fetch_add(1, Ordering::SeqCst);
            let (Value::Num(t), Value::Num(y)) = (&args[0], &args[1]) else {
                panic!("scalar time and state");
            };
            let derivative = match kind {
                Rhs::Growth => *y,
                Rhs::Quartic => 4.0 * t * t * t,
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    derivative,
                ))
            })
        },
    )))
    .unwrap();
    test(
        &runtime,
        &Value::FunctionHandle("refine_rhs".into()),
        &calls,
    );
}

fn input(runtime: &crate::context::RuntimeContext, times: &[f64], initial: f64) -> OdeInput {
    OdeInput {
        tspan: times.to_vec(),
        y0: vec![initial],
        y_shape: vec![1, 1],
        scalar_state: true,
        extensions: super::super::ode45::ODE_COMPATIBILITY_EXTENSIONS,
        runtime: Some(runtime.clone()),
    }
}

fn options(refine: Option<usize>) -> OdeOptions {
    OdeOptions {
        rel_tol: 1e-5,
        abs_tol: 1e-8,
        initial_step: Some(0.2),
        max_step: Some(0.3),
        max_steps: 1000,
        refine,
    }
}

#[test]
fn quartic_and_cubic_extensions_reproduce_independent_polynomial_solutions() {
    with_rhs(Rhs::Quartic, |runtime, function, calls| {
        let input = input(runtime, &[0.0, 1.0], 0.0);
        let step = block_on(runge_kutta::step_ode45(
            "ode45",
            function,
            0.0,
            &[0.0],
            1.0,
            &input,
            true,
        ))
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 7);
        let interpolant = step.interpolant.unwrap();
        for theta in [0.25_f64, 0.5, 0.75] {
            let row = block_on(interpolant.evaluate(
                "ode45",
                Some(runtime),
                &[0.0],
                &step.y_next,
                1.0,
                theta,
            ))
            .unwrap();
            assert!((row[0] - theta.powi(4)).abs() < 2e-13);
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            7,
            "interpolation adds no RHS calls"
        );
    });
    let cubic = StepInterpolant::BogackiShampine {
        first: vec![0.0],
        last: vec![3.0],
    };
    for theta in [0.25_f64, 0.5, 0.75] {
        let row = block_on(cubic.evaluate("ode23", None, &[0.0], &[1.0], 1.0, theta)).unwrap();
        assert!((row[0] - theta.powi(3)).abs() < 1e-15);
    }
}

#[test]
fn dormand_prince_midpoint_accuracy_has_quartic_dense_order_not_linear_interpolation() {
    with_rhs(Rhs::Growth, |runtime, function, _| {
        let input = input(runtime, &[0.0, 1.0], 1.0);
        let mut errors = Vec::new();
        for h in [0.25_f64, 0.125] {
            let step = block_on(runge_kutta::step_ode45(
                "ode45",
                function,
                0.0,
                &[1.0],
                h,
                &input,
                true,
            ))
            .unwrap();
            let row = block_on(step.interpolant.unwrap().evaluate(
                "ode45",
                Some(runtime),
                &[1.0],
                &step.y_next,
                h,
                0.5,
            ))
            .unwrap();
            errors.push((row[0] - (0.5 * h).exp()).abs());
        }
        assert!(errors[1] < 1e-8);
        let ratio = errors[0] / errors[1];
        // A quartic extension has O(h^5) local interpolation error. This
        // finite-h range excludes a linear/cubic interpolation substitute.
        assert!(
            ratio > 20.0 && ratio < 45.0,
            "observed local error ratio {ratio}"
        );
    });
}

#[test]
fn default_refinement_preserves_accepted_steps_and_derivative_counts_in_both_directions() {
    // Huge Refine cannot request duplicate ULP-rounded times even when its
    // scalar option value fits usize; reject before allocating RK stages.
    let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
    let gap = adjacent - 1.0;
    assert!(validate_output_times("ode45", 1.0, adjacent, gap, usize::MAX, 1.0).is_err());
    assert!(validate_output_times("ode45", adjacent, 1.0, -gap, usize::MAX, -1.0).is_err());
    with_rhs(Rhs::Growth, |runtime, function, calls| {
        for (times, initial) in [([0.0, 1.0], 1.0), ([1.0, 0.0], std::f64::consts::E)] {
            let input = input(runtime, &times, initial);
            calls.store(0, Ordering::SeqCst);
            let coarse = block_on(solve_ode(
                "ode45",
                OdeMethod::Ode45,
                function,
                &input,
                &options(Some(1)),
            ))
            .unwrap();
            let coarse_calls = calls.load(Ordering::SeqCst);
            calls.store(0, Ordering::SeqCst);
            let fine = block_on(solve_ode(
                "ode45",
                OdeMethod::Ode45,
                function,
                &input,
                &options(None),
            ))
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), coarse_calls);
            assert_eq!(fine.t.len(), 1 + 4 * (coarse.t.len() - 1));
            assert_eq!(fine.t.first(), Some(&times[0]));
            assert_eq!(fine.t.last(), Some(&times[1]));
            for (index, time) in coarse.t.iter().enumerate() {
                assert_eq!(fine.t[4 * index], *time);
                assert_eq!(fine.y_rows[4 * index], coarse.y_rows[index]);
            }
            let direction = times[1] - times[0];
            assert!(fine
                .t
                .windows(2)
                .all(|pair| (pair[1] - pair[0]) * direction > 0.0));
            for (time, row) in fine.t.iter().zip(&fine.y_rows) {
                assert!((row[0] - time.exp()).abs() < 5e-5);
            }
        }
    });
}

#[test]
fn explicit_nodes_and_sibling_refinement_have_distinct_truthful_contracts() {
    with_rhs(Rhs::Growth, |runtime, function, calls| {
        let explicit = input(runtime, &[0.0, 0.1, 0.7, 1.0], 1.0);
        calls.store(0, Ordering::SeqCst);
        let plain = block_on(solve_ode(
            "ode45",
            OdeMethod::Ode45,
            function,
            &explicit,
            &options(Some(1)),
        ))
        .unwrap();
        let plain_calls = calls.load(Ordering::SeqCst);
        calls.store(0, Ordering::SeqCst);
        let refined = block_on(solve_ode(
            "ode45",
            OdeMethod::Ode45,
            function,
            &explicit,
            &options(Some(7)),
        ))
        .unwrap();
        assert_eq!(refined.t, explicit.tspan);
        assert_eq!(refined.t, plain.t);
        assert_eq!(refined.y_rows, plain.y_rows);
        assert_eq!(calls.load(Ordering::SeqCst), plain_calls);

        let adaptive = input(runtime, &[0.0, 1.0], 1.0);
        calls.store(0, Ordering::SeqCst);
        let coarse = block_on(solve_ode(
            "ode23",
            OdeMethod::Ode23,
            function,
            &adaptive,
            &options(None),
        ))
        .unwrap();
        let coarse_calls = calls.load(Ordering::SeqCst);
        calls.store(0, Ordering::SeqCst);
        let fine = block_on(solve_ode(
            "ode23",
            OdeMethod::Ode23,
            function,
            &adaptive,
            &options(Some(3)),
        ))
        .unwrap();
        assert_eq!(fine.t.len(), 1 + 3 * (coarse.t.len() - 1));
        assert_eq!(calls.load(Ordering::SeqCst), coarse_calls);
        calls.store(0, Ordering::SeqCst);
        let error = block_on(solve_ode(
            "ode15s",
            OdeMethod::Ode15s,
            function,
            &adaptive,
            &options(Some(3)),
        ))
        .err()
        .expect("unsupported stiff refinement must not be ignored");
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<UnsupportedOutputRefinement>());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn cancelled_dense_work_retains_the_original_context_and_typed_control_source() {
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    runtime.set_error_namespace("RefineScope");
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let interpolant = StepInterpolant::BogackiShampine {
        first: vec![1.0],
        last: vec![2.0],
    };
    let error = block_on(interpolant.evaluate("ode23", Some(&runtime), &[1.0], &[2.0], 1.0, 0.5))
        .unwrap_err();
    assert_eq!(error.identifier(), Some("RefineScope:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert!(matches!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<crate::execution::ExecutionServiceError>(),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
}
