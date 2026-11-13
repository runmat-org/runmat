use super::*;
use futures::executor::block_on;
use std::sync::atomic::{AtomicUsize, Ordering};
#[test]
fn constraints_only_probe_never_invents_or_replays_objective_measurements() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("semi_objective", 2601), ("semi_constraints", 2602)],
    )
    .unwrap();
    let objective = Arc::new(AtomicUsize::new(0));
    let constraints = Arc::new(AtomicUsize::new(0));
    let oc = objective.clone();
    let cc = constraints.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            let Value::Num(x) = &args[0] else {
                panic!("original scalar x shape");
            };
            let x = *x;
            let outputs = if id == 2601 {
                oc.fetch_add(1, Ordering::SeqCst);
                assert_eq!(requested, 2);
                vec![Value::Num((x - 1.0).powi(2)), Value::Num(2.0 * (x - 1.0))]
            } else {
                cc.fetch_add(1, Ordering::SeqCst);
                assert_eq!(id, 2602);
                assert_eq!(requested, 4);
                let Value::Tensor(spacing) = &args[1] else {
                    panic!("sampling interval matrix");
                };
                assert_eq!(spacing.shape(), [1, 2]);
                let h = crate::builtins::common::tensor::tensor_value_f64(spacing, 0);
                let h = if h.is_nan() { 0.125 } else { h };
                let count = (1.0 / h).round() as usize + 1;
                let grid = (0..count)
                    .map(|i| -1.0 - (i as f64 * h - 0.37).powi(2) + 0.0 * x)
                    .collect();
                vec![
                    Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                    Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                    Value::Tensor(runmat_value::Tensor::new(grid, vec![1, count]).unwrap()),
                    Value::Tensor(runmat_value::Tensor::new(vec![h, 0.0], vec![1, 2]).unwrap()),
                ]
            };
            Box::pin(async move {
                crate::execution::RootedValueSequence::new(
                    runmat_value::ValueSequence::comma_separated(outputs)
                        .map_err(crate::sequence::sequence_error_to_runtime)?,
                )
                .map_err(|e| e.into_runtime_error())
            })
        },
    )))
    .unwrap();
    let function = Value::FunctionHandle("semi_objective".into());
    let callback = Value::FunctionHandle("semi_constraints".into());
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: "fseminf" },
        Some(&runtime),
    );
    let mut evaluator = Evaluator {
        internal: None,
        semi_infinite: Some(
            super::super::Domain::new(
                1,
                super::super::Controls {
                    check_values: true,
                    ..Default::default()
                },
            )
            .unwrap(),
        ),
        attainment: None,
        scaling: Default::default(),
        cooperation: &mut cooperation,
        objective: super::super::super::evaluation::Objective::User(&function),
        feasibility_limit: None,
        selected_constraints: None,
        constraints: Some(&callback),
        policy: Policy {
            owner: runmat_builtins::BuiltinCatalogIdentity { name: "fseminf" },
            invalid: None,
            numeric: Some(&runmat_builtins::FMINCON_NUMERIC_EXTENSION),
            single: Some(&runmat_builtins::FMINCON_SINGLE_EXTENSION),
            resident: Some(&runmat_builtins::FMINCON_RESIDENT_EXTENSION),
            sparse: None,
        },
        shape: vec![1, 1],
        scalar: true,
        objective_gradient: true,
        constraint_gradient: false,
        budget: evaluation::Budget::new(20),
        constraint_extents: None,
    };
    let point = match block_on(evaluator.evaluate(&[0.0])).unwrap() {
        Ok(point) => point,
        Err(_) => panic!("defined original point"),
    };
    assert_eq!(point.f, 1.0);
    let probe =
        match block_on(evaluator.evaluate_probe(&[0.125], evaluation::ProbeRole::ConstraintsOnly))
            .unwrap()
        {
            Ok(probe) => probe,
            Err(_) => panic!("defined constraint-only probe"),
        };
    assert!(matches!(&probe, evaluation::Probe::ConstraintsOnly { .. }));
    assert!(probe.actual_objective().is_err());
    assert_eq!(point.f, 1.0);
    assert_eq!(objective.load(Ordering::SeqCst), 1);
    assert_eq!(constraints.load(Ordering::SeqCst), 2);
    let counts = evaluator.budget.counts();
    assert_eq!(counts.points, 2);
    assert_eq!(counts.objective_calls, 1);
    assert_eq!(counts.constraint_calls, 2);

    let previous_stamp = super::super::stamp(&point).unwrap();
    let geometry = super::super::super::model::Geometry {
        variables: 1,
        inequality: super::super::super::model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 1)),
            rhs: Vec::new(),
        },
        equality: super::super::super::model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 1)),
            rhs: Vec::new(),
        },
        lower: vec![f64::NEG_INFINITY],
        upper: vec![f64::INFINITY],
    };
    let parameters = super::super::super::outcome::Parameters {
        max_iterations: 2,
        feasibility: 1e-6,
        optimality: 1e-6,
        step: 1e-8,
        function: Some(1e-4),
        differences: runmat_builtins::OptimDifferenceType::Forward,
        callback: None,
        display: runmat_builtins::OptimDisplay::Off,
    };
    let mut accepted = super::super::super::outcome::Outcome {
        algorithm: runmat_builtins::OptimAlgorithm::ActiveSet,
        x: vec![0.0],
        point: Some(point),
        constraints: None,
        multipliers: Vec::new(),
        multipliers_available: false,
        hessian: None,
        iterations: 0,
        counts,
        termination: super::super::super::outcome::Termination::Limit,
        violation: None,
        optimality: None,
        step: 0.0,
        alpha: None,
    };
    assert!(matches!(
        block_on(super::super::refresh(
            &mut accepted,
            &geometry,
            &parameters,
            &mut evaluator
        ))
        .unwrap(),
        super::super::Refresh::Rebuilt
    ));
    assert_eq!(accepted.point.as_ref().unwrap().f, 1.0);
    assert_eq!(objective.load(Ordering::SeqCst), 1);
    assert_eq!(constraints.load(Ordering::SeqCst), 4);
    let refreshed = evaluator.budget.counts();
    assert_eq!(refreshed.points, 4);
    assert_eq!(refreshed.objective_calls, 1);
    assert_eq!(refreshed.constraint_calls, 4);
    assert!(super::super::require_current(
        &evaluator,
        accepted.point.as_ref().unwrap(),
        accepted.constraints.as_ref().unwrap()
    )
    .is_ok());
    accepted.constraints.as_mut().unwrap().semi_infinite = Some(previous_stamp);
    assert!(super::super::require_current(
        &evaluator,
        accepted.point.as_ref().unwrap(),
        accepted.constraints.as_ref().unwrap()
    )
    .is_err());
}
