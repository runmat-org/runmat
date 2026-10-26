use super::*;
use futures::executor::block_on;
use runmat_value::Tensor;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
fn args(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn opts(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (k, v) in fields {
        out.insert(k, v);
    }
    Value::Struct(out.finish())
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(x) => vec![*x],
        Value::Tensor(x) => x.materialize_f64().clone(),
        _ => panic!("numeric"),
    }
}
fn outputs(value: Value) -> runmat_value::OutputListStorage {
    let Value::OutputList(out) = value else {
        panic!("outputs")
    };
    out
}
fn field(value: &Value, name: &str) -> f64 {
    let Value::Struct(value) = value else {
        panic!("metadata")
    };
    let Value::Num(out) = value.fields[name] else {
        panic!("count")
    };
    out
}
#[test]
fn general_rosenbrock_and_matrix_quadratic_preserve_shape_and_actual_accounting() {
    for rosenbrock in [true, false] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("nm_objective", 1101)],
        )
        .unwrap();
        let _entry = runtime.enter();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                assert_eq!(id, 1101);
                assert_eq!(requested, 1);
                counted.fetch_add(1, Ordering::SeqCst);
                let x = values(&args[0]);
                let f = if rosenbrock {
                    100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
                } else {
                    x.iter()
                        .enumerate()
                        .map(|(i, x)| (x - (i + 1) as f64).powi(2))
                        .sum()
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(f))
                })
            },
        )))
        .unwrap();
        let (x, shape, target) = if rosenbrock {
            (vec![-1.2, 1.0], vec![1, 2], vec![1.0, 1.0])
        } else {
            (vec![0.0; 4], vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        };
        let _output = crate::output_count::push_output_count(Some(4)).unwrap();
        let out = outputs(
            block_on(evaluate(
                Some(&runtime),
                args(vec![
                    Value::FunctionHandle("nm_objective".into()),
                    Value::Tensor(Tensor::new(x, shape.clone()).unwrap()),
                    opts([
                        ("Display", Value::from("off")),
                        ("TolX", Value::Num(1e-7)),
                        ("TolFun", Value::Num(1e-10)),
                        ("MaxFunEvals", Value::Num(4000.0)),
                        ("MaxIter", Value::Num(2000.0)),
                    ]),
                ]),
            ))
            .unwrap(),
        );
        let actual = values(&out[0]);
        assert!(
            actual.iter().zip(target).all(|(a, b)| (a - b).abs() < 1e-4),
            "{actual:?}"
        );
        assert!(matches!(out[2], Value::Num(1.0)));
        assert!(values(&out[1])[0] < 1e-8);
        let Value::Tensor(x) = &out[0] else {
            panic!("array")
        };
        assert_eq!(x.shape(), shape);
        assert_eq!(
            field(&out[3], "funcCount"),
            calls.load(Ordering::SeqCst) as f64
        );
    }
}
#[test]
fn hard_budgets_partial_simplex_and_nan_inf_never_fake_convergence() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("nm_limits", 1102)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let mode = Arc::new(AtomicUsize::new(0));
    let selected = mode.clone();
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(move |_, _, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            let f = match selected.load(Ordering::SeqCst) {
                0 => 1.0,
                1 => f64::NAN,
                _ => f64::INFINITY,
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(f))
            })
        })))
        .unwrap();
    let _output = crate::output_count::push_output_count(Some(4)).unwrap();
    for (kind, limit, iteration_limit, expected_iterations, expected_calls) in [
        (0, 1, 200, 0.0, 1),
        (0, 2, 200, 0.0, 2),
        (0, 4, 200, 1.0, 4),
        (1, 8, 200, 2.0, 8),
        (2, 8, 200, 2.0, 8),
        (0, 100, 1, 1.0, 3),
        (0, 100, 0, 0.0, 1),
    ] {
        mode.store(kind, Ordering::SeqCst);
        calls.store(0, Ordering::SeqCst);
        let display = if kind == 0 && limit == 4 {
            "iter"
        } else {
            "off"
        };
        let capture = crate::console::begin_capture().unwrap();
        let out = outputs(
            block_on(evaluate(
                Some(&runtime),
                args(vec![
                    Value::FunctionHandle("nm_limits".into()),
                    Value::Tensor(
                        Tensor::new(
                            vec![if limit == 1 { f64::MAX } else { 1.0 }, 2.0],
                            vec![2, 1],
                        )
                        .unwrap(),
                    ),
                    opts([
                        ("Display", Value::from(display)),
                        ("MaxFunEvals", Value::Num(limit as f64)),
                        ("MaxIter", Value::Num(iteration_limit as f64)),
                    ]),
                ]),
            ))
            .unwrap(),
        );
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
        assert_eq!(field(&out[3], "funcCount"), expected_calls as f64);
        assert_eq!(field(&out[3], "iterations"), expected_iterations);
        assert!(matches!(out[2], Value::Num(0.0)));
        let text = capture.finish().unwrap();
        if display == "iter" {
            assert_eq!(text.matches("Func-count").count(), 1);
            let rows: Vec<_> = text
                .lines()
                .filter_map(|line| {
                    let fields: Vec<_> = line.split_whitespace().collect();
                    Some((
                        fields.first()?.parse::<usize>().ok()?,
                        fields.get(1)?.parse::<usize>().ok()?,
                    ))
                })
                .collect();
            assert_eq!(rows, [(0, 1), (1, 3)]);
            assert!(text.contains("FuncCount: 4"));
        } else {
            assert!(text.is_empty());
        }
    }
    mode.store(1, Ordering::SeqCst);
    let error = block_on(evaluate(
        Some(&runtime),
        args(vec![
            Value::FunctionHandle("nm_limits".into()),
            Value::Num(1.0),
            opts([
                ("Display", Value::from("off")),
                ("FunValCheck", Value::from("on")),
            ]),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:fminsearch:InvalidFunctionValue")
    );
}
#[test]
fn original_observer_states_stop_error_and_cancel_do_not_replay_objective() {
    for action in 0..4 {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::user_functions::test_support::exact_fixture_runtime()
            .with_cancellation_token(cancellation.token());
        runtime.set_error_namespace("NelderInvocation");
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("nm_observed", 1103), ("nm_observer", 1104)],
        )
        .unwrap();
        let _entry = runtime.enter();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let states = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = states.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, _| {
                let value = if id == 1103 {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok(Value::Num(values(&args[0])[0].powi(2)))
                } else {
                    assert_eq!(id, 1104);
                    let state =
                        crate::builtins::common::validation::value_text_scalar_string(&args[2])
                            .unwrap();
                    let Value::Struct(metadata) = &args[1] else {
                        panic!("values")
                    };
                    assert_eq!(metadata.fields.len(), 4);
                    assert_eq!(
                        field(&args[1], "funccount"),
                        counted.load(Ordering::SeqCst) as f64
                    );
                    seen.lock().unwrap().push(state.clone());
                    if action == 1 {
                        Err(build_runtime_error("original observer error")
                            .with_identifier("Original:NMObserver")
                            .with_source(std::io::Error::other("owned source"))
                            .build())
                    } else if action == 2 {
                        cancellation.cancel(runmat_execution::CancellationReason::User);
                        Ok(Value::Bool(false))
                    } else {
                        Ok(Value::Bool(
                            state == if action == 3 { "iter" } else { "init" },
                        ))
                    }
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(value?)
                })
            },
        )))
        .unwrap();
        let _output = crate::output_count::push_output_count(Some(4)).unwrap();
        let result = block_on(evaluate(
            Some(&runtime),
            args(vec![
                Value::FunctionHandle("nm_observed".into()),
                Value::Num(1.0),
                opts([
                    ("Display", Value::from("off")),
                    ("OutputFcn", Value::FunctionHandle("nm_observer".into())),
                ]),
            ]),
        ));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            if action == 3 { 2 } else { 1 }
        );
        match action {
            0 => {
                let out = outputs(result.unwrap());
                assert!(matches!(out[2], Value::Num(-1.0)));
                assert_eq!(*states.lock().unwrap(), ["init", "done"]);
            }
            1 => {
                let error = result.unwrap_err();
                assert_eq!(error.identifier(), Some("Original:NMObserver"));
                assert!(std::error::Error::source(&error)
                    .unwrap()
                    .is::<std::io::Error>());
                assert_eq!(states.lock().unwrap().len(), 1);
            }
            2 => {
                let error = result.unwrap_err();
                assert_eq!(error.identifier(), Some("NelderInvocation:Cancelled"));
                assert!(crate::runtime_error::error_has_control_cancellation_origin(
                    &error
                ));
                assert_eq!(states.lock().unwrap().len(), 1);
            }
            _ => {
                let out = outputs(result.unwrap());
                assert!(matches!(out[2], Value::Num(-1.0)));
                assert_eq!(field(&out[3], "iterations"), 1.0);
                assert_eq!(*states.lock().unwrap(), ["init", "iter", "done"]);
            }
        }
    }
}
#[test]
fn problem_defaults_and_legacy_constructor_do_not_enable_optimoptions() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("nm_problem", 1105)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, args, _| {
            let f = (values(&args[0])[0] - 2.0).abs();
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(f))
            })
        })))
        .unwrap();
    let problem = opts([
        ("objective", Value::FunctionHandle("nm_problem".into())),
        ("x0", Value::Num(0.0)),
        ("solver", Value::from("fminsearch")),
        ("options", opts([("Display", Value::from("off"))])),
    ]);
    let x = block_on(evaluate(Some(&runtime), args(vec![problem]))).unwrap();
    assert!((values(&x)[0] - 2.0).abs() < 1e-3);
    assert!(runmat_builtins::OptimSolver::parse("fminsearch").is_none());
    assert!(block_on(options::Options::parse(
        Some(&opts([("Tol", Value::Num(1.0))])),
        1,
        None
    ))
    .is_err());
}
#[test]
fn public_boundary_preserves_original_finite_and_cancel_owners_before_callbacks() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Finite(ManagedHostEffectLedger);
    impl RuntimeParallelService for Finite {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let x = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(x.free_bytes()),
                epoch: x.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    let error = block_on(evaluate(Some(&runtime), args(vec![Value::Num(1.0)]))).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(source.token());
    runtime.set_error_namespace("NelderInvocation");
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(evaluate(Some(&runtime), args(vec![]))).unwrap_err();
    assert_eq!(error.identifier(), Some("NelderInvocation:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
