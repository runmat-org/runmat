use super::*;
use futures::executor::block_on;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[test]
fn observer_fault_and_cancellation_keep_original_owner_and_never_replay_observers() {
    for cancelled in [false, true] {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::user_functions::test_support::exact_fixture_runtime()
            .with_cancellation_token(cancellation.token());
        runtime.set_error_namespace("ObserverScope");
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("observed_objective", 985), ("failing_observer", 986)],
        )
        .unwrap();
        let _entry = runtime.enter();
        let observers = Arc::new(AtomicUsize::new(0));
        let observed = observers.clone();
        let objectives = Arc::new(AtomicUsize::new(0));
        let counted = objectives.clone();
        let at_observer = Arc::new(AtomicUsize::new(0));
        let at_call = at_observer.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, _| {
                let output = if id == 985 {
                    counted.fetch_add(1, Ordering::SeqCst);
                    let Value::Num(x) = args[0] else {
                        panic!("scalar")
                    };
                    Ok(Value::Num((x - 3.0).powi(2)))
                } else {
                    assert_eq!(id, 986);
                    observed.fetch_add(1, Ordering::SeqCst);
                    at_call.store(counted.load(Ordering::SeqCst), Ordering::SeqCst);
                    if cancelled {
                        cancellation.cancel(runmat_execution::CancellationReason::User);
                        Ok(Value::Bool(false))
                    } else {
                        Err(crate::build_runtime_error("original observer failure")
                            .with_identifier("Observer:Original")
                            .with_source(std::io::Error::other("original observer source"))
                            .build())
                    }
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(output?)
                })
            },
        )))
        .unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert(
            "OutputFcn",
            Value::FunctionHandle("failing_observer".into()),
        );
        let args =
            crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
        let error = block_on(fminunc_in_runtime(
            Some(&runtime),
            Value::FunctionHandle("observed_objective".into()),
            Value::Num(0.0),
            args.into(),
        ))
        .unwrap_err();
        assert_eq!(observers.load(Ordering::SeqCst), 1);
        assert_eq!(
            objectives.load(Ordering::SeqCst),
            at_observer.load(Ordering::SeqCst)
        );
        if cancelled {
            assert_eq!(error.identifier(), Some("ObserverScope:Cancelled"));
            assert!(crate::runtime_error::error_has_control_cancellation_origin(
                &error
            ));
        } else {
            assert_eq!(error.identifier(), Some("Observer:Original"));
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some());
        }
    }
}

#[test]
fn output_observer_sees_initial_accepted_and_final_states_and_stops_without_extra_objectives() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[
            ("observer_objective", 981),
            ("iteration_observer", 982),
            ("iteration_observer_two", 984),
        ],
    )
    .unwrap();
    let _entry = runtime.enter();
    let observations = Arc::new(Mutex::new(Vec::new()));
    let observed = observations.clone();
    let objectives = Arc::new(AtomicUsize::new(0));
    let objective_calls = objectives.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(requested, 1);
            let output = match id {
                981 => {
                    objective_calls.fetch_add(1, Ordering::SeqCst);
                    let Value::Num(x) = args[0] else {
                        panic!("scalar objective")
                    };
                    Value::Num((x - 3.0).powi(2))
                }
                982 | 984 => {
                    let Value::Num(x) = args[0] else {
                        panic!("scalar iterate")
                    };
                    let Value::Struct(ref values) = args[1] else {
                        panic!("iteration values")
                    };
                    let Value::String(runmat_value::StringElement::Text(ref state)) = args[2] else {
                        panic!("state")
                    };
                    let Value::Num(iteration) = values.fields["iteration"] else {
                        panic!("iteration")
                    };
                    let Value::Num(count) = values.fields["funccount"] else {
                        panic!("count")
                    };
                    let Value::Num(fval) = values.fields["fval"] else {
                        panic!("fval")
                    };
                    assert_eq!(count as usize, objective_calls.load(Ordering::SeqCst));
                    assert!((fval - (x - 3.0).powi(2)).abs() < 1e-12);
                    let Value::Tensor(ref direction) = values.fields["searchdirection"] else { panic!("direction") };
                    if state.as_str() == "init" {
                        assert!(direction.is_empty());
                        assert!(matches!(values.fields["lssteplength"], Value::Tensor(ref value) if value.is_empty()));
                    } else {
                        let Value::Num(alpha) = values.fields["lssteplength"] else { panic!("line length") };
                        assert!((x-alpha*direction.materialize_f64()[0]).abs()<1e-10);
                        assert!(matches!(values.fields["directionalderivative"],Value::Num(value) if value.is_finite()));
                    }
                    observed
                        .lock()
                        .unwrap()
                        .push((state.as_str().to_owned(), iteration, x, count));
                    Value::Bool(id == 982 && state.as_str() == "iter")
                }
                _ => panic!("unexpected callback"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(output)
            })
        },
    )))
    .unwrap();
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert(
        "OutputFcn",
        Value::Cell(
            runmat_value::CellArray::new(
                vec![
                    Value::FunctionHandle("iteration_observer".into()),
                    Value::FunctionHandle("iteration_observer_two".into()),
                ],
                1,
                2,
            )
            .unwrap(),
        ),
    );
    options.insert("MaxIterations", Value::Num(10.0));
    options.insert("Display", Value::from("iter"));
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let capture = crate::console::begin_capture().unwrap();
    let output = block_on(runtime.scope(fminunc_builtin(
        Value::FunctionHandle("observer_objective".into()),
        Value::Num(0.0),
        vec![Value::Struct(options.finish())],
    )))
    .unwrap();
    let Value::OutputList(outputs) = output else {
        panic!("outputs")
    };
    assert!(matches!(outputs[2], Value::Num(-1.0)));
    let observations = observations.lock().unwrap();
    assert_eq!(
        observations
            .iter()
            .map(|row| row.0.as_str())
            .collect::<Vec<_>>(),
        ["init", "init", "iter", "iter", "done", "done"]
    );
    assert_eq!(observations[0].1, 0.0);
    assert_eq!(observations[2].1, 1.0);
    assert_eq!(observations[2].2, observations[4].2);
    assert_eq!(observations[2].3, observations[4].3);
    let text = capture.finish().unwrap();
    assert_eq!(text.matches("First-order optimality").count(), 1);
    let rows: Vec<_> = text
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            fields
                .first()?
                .parse::<usize>()
                .ok()
                .map(|iteration| (iteration, fields))
        })
        .collect();
    assert_eq!(rows.len(), 2, "{text}");
    for ((iteration, fields), observed) in rows.iter().zip([&observations[0], &observations[2]]) {
        assert_eq!(*iteration, observed.1 as usize);
        assert_eq!(fields[1].parse::<usize>().unwrap(), observed.3 as usize);
        assert!((fields[2].parse::<f64>().unwrap() - (observed.2 - 3.0).powi(2)).abs() < 1e-6);
    }
    assert_eq!(rows[0].1[3], "-");
    assert!(rows[1].1[3].parse::<f64>().unwrap() > 0.0);
    assert_eq!(text.matches("fminunc: fval").count(), 1);
    assert_eq!(
        objectives.load(Ordering::SeqCst),
        observations[4].3 as usize
    );
}
