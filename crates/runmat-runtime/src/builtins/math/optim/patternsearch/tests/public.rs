//! Normal public callbacks use the existing exact semantic fixture, not a
//! replacement mathematical oracle inside the pattern engine.
use super::super::*;
mod batching;
mod mads;
mod latin;
mod custom;
mod nonuniform;
mod nelder;
mod rbf;
use futures::executor::block_on;
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
fn arguments(values: Vec<Value>) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(values)
}
fn array(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}
fn empty() -> Value {
    array(Vec::new(), vec![0, 0])
}
fn structure(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut output = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        output.insert(name, value);
    }
    Value::Struct(output.finish())
}
fn numbers(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(value) => value.materialize_f64(),
        _ => panic!("numeric"),
    }
}
fn returned(values: Vec<Value>) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::execution::RootedValueSequence::new(
        runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?,
    )
    .map_err(|error| error.into_runtime_error())
}
fn request(options: Value) -> crate::call::BuiltinArguments {
    arguments(vec![
        Value::FunctionHandle("pattern_objective".into()),
        Value::Num(0.0),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        options,
    ])
}

#[test]
fn classic_public_shapes_linear_equalities_and_nonlinear_counts_are_actual() {
    for mode in 0..4 {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29201), ("pattern_constraint", 29202)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let constraints = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let constrained = constraints.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                let x = numbers(&args[0]);
                let values = if id == 29201 {
                    assert_eq!(width, 1);
                    counted.fetch_add(1, Ordering::SeqCst);
                    vec![Value::Num((x[0] - 0.25).powi(2) + (x[1] - 0.75).powi(2))]
                } else {
                    assert_eq!(id, 29202);
                    assert_eq!(width, 2);
                    constrained.fetch_add(1, Ordering::SeqCst);
                    vec![Value::Num(x[0] * x[0] + x[1] * x[1] - 1.0), empty()]
                };
                Box::pin(async move { returned(values) })
            },
        )))
        .unwrap();
        let opts = structure([
            ("Display", Value::from("off")),
            ("MaxFunctionEvaluations", Value::Num(5000.0)),
        ]);
        let values = vec![
            Value::FunctionHandle("pattern_objective".into()),
            array(
                if mode == 3 {
                    vec![0.25, 0.5]
                } else {
                    vec![0.5, 0.5]
                },
                vec![1, 2],
            ),
            empty(),
            empty(),
            if mode == 1 {
                array(vec![1.0, 1.0], vec![1, 2])
            } else if mode == 3 {
                array(vec![1.0, 0.0], vec![1, 2])
            } else {
                empty()
            },
            if mode == 1 {
                Value::Num(1.0)
            } else if mode == 3 {
                Value::Num(0.25)
            } else {
                empty()
            },
            array(vec![0.0, 0.0], vec![1, 2]),
            array(vec![1.0, 1.0], vec![1, 2]),
            if mode == 2 {
                Value::FunctionHandle("pattern_constraint".into())
            } else {
                empty()
            },
            opts,
        ];
        let args = if mode == 0 {
            arguments(vec![structure([
                ("objective", values[0].clone()),
                ("x0", values[1].clone()),
                ("Aineq", values[2].clone()),
                ("bineq", values[3].clone()),
                ("Aeq", values[4].clone()),
                ("beq", values[5].clone()),
                ("lb", values[6].clone()),
                ("ub", values[7].clone()),
                ("nonlcon", values[8].clone()),
                ("options", values[9].clone()),
                ("solver", Value::from("patternsearch")),
            ])])
        } else {
            arguments(values)
        };
        let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
        let output = block_on(execute(Some(&runtime), args)).unwrap();
        let values = output.as_slice();
        let x = numbers(&values[0]);
        assert!(matches!(&values[0], Value::Tensor(value) if value.shape()==[1,2]));
        assert!(
            (x[0] - 0.25).abs() < 1e-3 && (x[1] - 0.75).abs() < 1e-3,
            "mode={mode}, x={x:?}"
        );
        assert!(numbers(&values[1])[0] < 1e-6);
        assert!(numbers(&values[2])[0] > 0.0);
        assert!(x.iter().all(|&x| (0.0..=1.0).contains(&x)));
        if mode == 1 {
            assert!((x[0] + x[1] - 1.0).abs() < 1e-6);
        }
        if mode == 3 {
            assert_eq!(x[0], 0.25);
        }
        if mode == 2 {
            assert_eq!(
                calls.load(Ordering::SeqCst),
                constraints.load(Ordering::SeqCst)
            );
        }
    }
}

#[test]
fn complete_poll_preserves_partial_improvement_and_original_callback_error() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29203)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let fail = Arc::new(AtomicUsize::new(0));
    let mode = fail.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            assert_eq!(id, 29203);
            assert_eq!(width, 1);
            let call = observed.fetch_add(1, Ordering::SeqCst) + 1;
            let value = Value::Num((numbers(&args[0])[0] - 1.0).powi(2));
            let failure = mode.load(Ordering::SeqCst) > 0 && call == 2;
            Box::pin(async move {
                if failure {
                    Err(crate::build_runtime_error("original objective failure")
                        .with_identifier("RunMat:PatternFixture:Original")
                        .build())
                } else {
                    returned(vec![value])
                }
            })
        },
    )))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let opts = || {
        structure([
            ("Display", Value::from("off")),
            ("UseCompletePoll", Value::Bool(true)),
            ("MaxFunctionEvaluations", Value::Num(2.0)),
        ])
    };
    let original_request = request(opts());
    let output = {
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _foreign = foreign.enter();
        let _foreign_outputs = crate::output_count::push_output_count(Some(1)).unwrap();
        block_on(execute(Some(&runtime), original_request)).unwrap()
    };
    assert_eq!(output.len(), 3, "the original output projection must win");
    assert_eq!(numbers(&output.as_slice()[0]), [1.0]);
    assert_eq!(numbers(&output.as_slice()[1]), [0.0]);
    assert_eq!(numbers(&output.as_slice()[2]), [0.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    calls.store(0, Ordering::SeqCst);
    fail.store(1, Ordering::SeqCst);
    let error = block_on(registered(Some(runtime.clone()), request(opts()))).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:PatternFixture:Original"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    calls.store(0, Ordering::SeqCst);
    let zero = structure([
        ("Display", Value::from("off")),
        ("MaxFunctionEvaluations", Value::Num(0.0)),
    ]);
    let output = block_on(execute(Some(&runtime), request(zero))).unwrap();
    assert_eq!(numbers(&output.as_slice()[0]), [0.0]);
    assert!(numbers(&output.as_slice()[1]).is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn mutable_three_output_observer_updates_budget_without_objective_replay() {
    use std::sync::Mutex;
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let clock = std::rc::Rc::new(
        crate::context::DeterministicTimingService::new(
            runmat_execution::host::NativeClockInputs {
                schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
                initial_monotonic_nanos: 0,
                initial_wall_unix_seconds: 0,
                initial_wall_nanosecond: 0,
                read_step_nanos: 1_000_000,
                local_offset_seconds: 0,
                process_cpu: None,
            },
            runtime.cancellation(),
        )
        .unwrap(),
    );
    let ports = runtime.service_ports().clone().with_timing(clock);
    let runtime = runtime.with_service_ports(ports);
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29204), ("pattern_observer", 29205)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let phases = Arc::new(Mutex::new(Vec::new()));
    let recorded = phases.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            let values = if id == 29204 {
                assert_eq!(width, 1);
                observed.fetch_add(1, Ordering::SeqCst);
                vec![Value::Num(numbers(&args[0])[0].powi(2))]
            } else {
                assert_eq!(id, 29205);
                assert_eq!(width, 3);
                assert_eq!(args.len(), 3);
                let phase = crate::builtins::common::validation::value_text_scalar_string(&args[2])
                    .unwrap();
                recorded.lock().unwrap().push(phase.clone());
                let Value::Struct(current) = &args[1] else {
                    panic!("options")
                };
                let mut changed = runmat_value::UnboundedStructValueBuilder::new();
                for (name, value) in &current.fields {
                    changed.insert(
                        name,
                        if name == "MaxFunctionEvaluations" {
                            Value::Num(1.0)
                        } else {
                            value.clone()
                        },
                    );
                }
                vec![
                    Value::Bool(false),
                    Value::Struct(changed.finish()),
                    Value::Bool(phase == "init"),
                ]
            };
            Box::pin(async move { returned(values) })
        },
    )))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let opts = structure([
        ("Display", Value::from("off")),
        (
            "OutputFcn",
            Value::FunctionHandle("pattern_observer".into()),
        ),
    ]);
    let output = block_on(execute(Some(&runtime), request(opts))).unwrap();
    assert_eq!(numbers(&output.as_slice()[2]), [0.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(*phases.lock().unwrap(), ["init", "done"]);
}

#[test]
fn cache_suppresses_actual_duplicate_callbacks_without_fabricating_machine_stopping() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29206)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            assert_eq!((id, width), (29206, 1));
            observed.fetch_add(1, Ordering::SeqCst);
            let f = (numbers(&args[0])[0] - 1.0).powi(2);
            Box::pin(async move { returned(vec![Value::Num(f)]) })
        },
    )))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let mut measured = Vec::new();
    for enabled in [false, true] {
        calls.store(0, Ordering::SeqCst);
        let options = structure([
            ("Display", Value::from("off")),
            ("Cache", Value::Bool(enabled)),
            ("CacheSize", Value::Num(100.0)),
            ("CacheTol", Value::Num(1e-12)),
            ("MeshTolerance", Value::Num(0.01)),
        ]);
        let result = block_on(execute(Some(&runtime), request(options))).unwrap();
        assert_eq!(numbers(&result.as_slice()[0]), [1.0]);
        assert_eq!(numbers(&result.as_slice()[1]), [0.0]);
        assert!(numbers(&result.as_slice()[2])[0] > 0.0);
        measured.push(calls.load(Ordering::SeqCst));
    }
    assert!(
        measured[1] < measured[0],
        "on={:?}, off={:?}",
        measured[1],
        measured[0]
    );
    calls.store(0, Ordering::SeqCst);
    let options = structure([
        ("Display", Value::from("off")),
        ("Cache", Value::Bool(true)),
        ("CacheTol", Value::Num(100.0)),
        ("MeshTolerance", Value::Num(0.01)),
    ]);
    let result = block_on(execute(Some(&runtime), request(options))).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        numbers(&result.as_slice()[2]),
        [1.0],
        "cached represented points are not a machine-precision step failure"
    );
}
