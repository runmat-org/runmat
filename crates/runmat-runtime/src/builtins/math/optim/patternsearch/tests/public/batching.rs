//! Vectorized shape, point limits, original errors, and placement selection.
use super::*;

#[test]
fn complete_vectorized_polls_preserve_row_column_constraints_and_point_limits() {
    for columns in [false, true] {
        for nonlinear in [false, true] {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let _entry = runtime.enter();
            let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
                &runtime,
                &[("pattern_objective", 29211), ("pattern_constraint", 29212)],
            )
            .unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let points = Arc::new(AtomicUsize::new(0));
            let constraints = Arc::new(AtomicUsize::new(0));
            let (observed, measured, constrained) =
                (calls.clone(), points.clone(), constraints.clone());
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |id, args, outputs| {
                    let Value::Tensor(argument) = &args[0] else {
                        panic!("vectorized numeric matrix")
                    };
                    let shape = argument.shape();
                    let width = if columns {
                        assert_eq!(shape[0], 2);
                        shape[1]
                    } else {
                        assert_eq!(shape[1], 2);
                        shape[0]
                    };
                    let data = argument.as_f64_slice().unwrap();
                    let mut values = Vec::new();
                    for point in 0..width {
                        let (x, y) = if columns {
                            (data[2 * point], data[2 * point + 1])
                        } else {
                            (data[point], data[width + point])
                        };
                        assert!((0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y));
                        values.push(if id == 29211 {
                            (x - 1.0).powi(2) + (y - 0.5).powi(2)
                        } else {
                            x + y - 3.0
                        });
                    }
                    let shape = if columns {
                        vec![1, width]
                    } else {
                        vec![width, 1]
                    };
                    let values = if id == 29211 {
                        assert_eq!(outputs, 1);
                        observed.fetch_add(1, Ordering::SeqCst);
                        measured.fetch_add(width, Ordering::SeqCst);
                        vec![array(values, shape)]
                    } else {
                        assert_eq!((id, outputs), (29212, 2));
                        constrained.fetch_add(1, Ordering::SeqCst);
                        vec![array(values, shape), empty()]
                    };
                    Box::pin(async move { returned(values) })
                }),
            ))
            .unwrap();
            let options = structure([
                ("Display", Value::from("off")),
                ("UseVectorized", Value::Bool(true)),
                ("UseCompletePoll", Value::Bool(true)),
                ("MaxFunctionEvaluations", Value::Num(3.0)),
                // Vectorization wins before reading this dormant selector.
                ("UseParallel", Value::FunctionHandle("not_a_switch".into())),
            ]);
            let shape = if columns { vec![2, 1] } else { vec![1, 2] };
            let args = arguments(vec![
                Value::FunctionHandle("pattern_objective".into()),
                array(vec![0.5, 0.5], shape.clone()),
                empty(),
                empty(),
                empty(),
                empty(),
                array(vec![0.0, 0.0], shape.clone()),
                array(vec![1.0, 1.0], shape.clone()),
                if nonlinear {
                    Value::FunctionHandle("pattern_constraint".into())
                } else {
                    empty()
                },
                options,
            ]);
            let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
            let output = {
                let foreign = crate::user_functions::test_support::exact_fixture_runtime();
                let _foreign = foreign.enter();
                let _one = crate::output_count::push_output_count(Some(1)).unwrap();
                block_on(execute(Some(&runtime), args)).unwrap()
            };
            assert_eq!(output.len(), 3);
            assert!(matches!(&output.as_slice()[0],Value::Tensor(value) if value.shape()==shape));
            let x = numbers(&output.as_slice()[0]);
            assert_eq!(
                numbers(&output.as_slice()[1]),
                [(x[0] - 1.0).powi(2) + (x[1] - 0.5).powi(2)]
            );
            assert_eq!(numbers(&output.as_slice()[2]), [0.0]);
            assert_eq!(points.load(Ordering::SeqCst), 3);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                2,
                "one initial and one actual two-point invocation"
            );
            assert_eq!(
                constraints.load(Ordering::SeqCst),
                if nonlinear { 2 } else { 0 }
            );
        }
    }
}

#[test]
fn selected_parallel_admission_precedes_callbacks_and_auto_has_only_known_fallback() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29213)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            assert_eq!((id, width), (29213, 1));
            observed.fetch_add(1, Ordering::SeqCst);
            let f = (numbers(&args[0])[0] - 1.0).powi(2);
            Box::pin(async move { returned(vec![Value::Num(f)]) })
        },
    )))
    .unwrap();
    let opts = |mode| {
        structure([
            ("Display", Value::from("off")),
            ("UseParallel", Value::from(mode)),
            (
                "Cache",
                Value::FunctionHandle("ignored_cache_switch".into()),
            ),
            (
                "CacheSize",
                Value::FunctionHandle("ignored_cache_size".into()),
            ),
            ("MaxFunctionEvaluations", Value::Num(2.0)),
        ])
    };
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let error = block_on(registered(Some(runtime.clone()), request(opts("on")))).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Execution:ParallelCallbackUnavailable")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let output = block_on(execute(Some(&runtime), request(opts("auto")))).unwrap();
    assert_eq!(numbers(&output.as_slice()[0]), [1.0]);
    assert_eq!(numbers(&output.as_slice()[2]), [0.0]);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn vectorized_callback_errors_and_orientation_refusals_do_not_replay() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29214)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let mode = Arc::new(AtomicUsize::new(0));
    let (observed, fail) = (calls.clone(), mode.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            assert_eq!((id, width), (29214, 1));
            let call = observed.fetch_add(1, Ordering::SeqCst) + 1;
            let fail = call == 2 && fail.load(Ordering::SeqCst) == 1;
            let values = numbers(&args[0])
                .into_iter()
                .map(|x| (x - 1.0).powi(2))
                .collect::<Vec<_>>();
            let len = values.len();
            let value = array(values, vec![1, len]);
            Box::pin(async move {
                if fail {
                    Err(crate::build_runtime_error("original vectorized failure")
                        .with_identifier("RunMat:PatternFixture:Vectorized")
                        .build())
                } else {
                    returned(vec![value])
                }
            })
        },
    )))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    for original_failure in [false, true] {
        calls.store(0, Ordering::SeqCst);
        mode.store(usize::from(original_failure), Ordering::SeqCst);
        let options = structure([
            ("Display", Value::from("off")),
            ("UseVectorized", Value::Bool(true)),
            ("UseCompletePoll", Value::Bool(true)),
            ("MaxFunctionEvaluations", Value::Num(3.0)),
        ]);
        let error = block_on(registered(Some(runtime.clone()), request(options))).unwrap_err();
        if original_failure {
            assert_eq!(error.identifier(), Some("RunMat:PatternFixture:Vectorized"));
        } else {
            assert!(error.message().contains("orientation"));
        }
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn random_order_drops_its_stream_turn_before_serial_and_vectorized_callbacks() {
    for vectorized in [false, true] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29215)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!((id, width), (29215, 1));
                let runtime = crate::context::legacy::active();
                let turn = crate::builtins::common::random::try_acquire_turn(
                    runtime.as_ref(),
                    "pattern_callback",
                )
                .unwrap();
                drop(turn);
                observed.fetch_add(1, Ordering::SeqCst);
                let data = numbers(&args[0]);
                let values: Vec<_> = data.iter().map(|&x| (x - 1.0).powi(2)).collect();
                let output = if vectorized {
                    let len = values.len();
                    array(values, vec![len, 1])
                } else {
                    Value::Num(values[0])
                };
                Box::pin(async move { returned(vec![output]) })
            },
        )))
        .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
        let options = structure([
            ("Display", Value::from("off")),
            ("PollOrderAlgorithm", Value::from("Random")),
            ("UseCompletePoll", Value::Bool(true)),
            ("UseVectorized", Value::Bool(vectorized)),
            ("MaxFunctionEvaluations", Value::Num(3.0)),
        ]);
        let output = block_on(execute(Some(&runtime), request(options))).unwrap();
        let x = numbers(&output.as_slice()[0])[0];
        assert_eq!(numbers(&output.as_slice()[1]), [(x - 1.0).powi(2)]);
        assert_eq!(x, 1.0);
        assert_eq!(calls.load(Ordering::SeqCst), if vectorized { 2 } else { 3 });
    }
}

#[test]
fn success_order_reuses_the_actual_previous_winning_ray_without_duplicate_poll_calls() {
    for (complete, bounded) in [(false, false), (true, false), (false, true)] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29216)],
        )
        .unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Vec<f64>>::new()));
        let recorded = seen.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!((id, width), (29216, 1));
                let x = numbers(&args[0]);
                let f = (x[0] - if bounded { 1.0 } else { 0.0 }).powi(2) + (x[1] - 3.0).powi(2);
                recorded.lock().unwrap().push(x);
                Box::pin(async move { returned(vec![Value::Num(f)]) })
            },
        )))
        .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
        let options = structure([
            ("Display", Value::from("off")),
            ("PollOrderAlgorithm", Value::from("Success")),
            ("ScaleMesh", Value::Bool(false)),
            ("UseCompletePoll", Value::Bool(complete)),
            (
                "MaxFunctionEvaluations",
                Value::Num(if bounded {
                    3.0
                } else if complete {
                    9.0
                } else {
                    4.0
                }),
            ),
        ]);
        let args = arguments(vec![
            Value::FunctionHandle("pattern_objective".into()),
            array(vec![0.0, 0.0], vec![1, 2]),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            if bounded {
                array(vec![1.0, f64::INFINITY], vec![1, 2])
            } else {
                empty()
            },
            empty(),
            options,
        ]);
        let output = block_on(execute(Some(&runtime), args)).unwrap();
        if bounded {
            assert_eq!(numbers(&output.as_slice()[0]), [1.0, 2.0]);
            assert_eq!(numbers(&output.as_slice()[1]), [1.0]);
            let actual = seen.lock().unwrap();
            assert_eq!(
                actual.as_slice(),
                [vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 2.0]],
                "an unavailable old ray cannot replace the current feasible cone"
            );
            continue;
        }
        assert_eq!(numbers(&output.as_slice()[0]), [0.0, 3.0]);
        assert_eq!(numbers(&output.as_slice()[1]), [0.0]);
        let actual = seen.lock().unwrap();
        assert_eq!(actual[1], [1.0, 0.0]);
        assert_eq!(actual[2], [0.0, 1.0]);
        let next = if complete { 5 } else { 3 };
        assert_eq!(
            actual[next],
            [0.0, 3.0],
            "the actual winning e2 ray precedes consecutive e1"
        );
        assert_eq!(
            actual[next..]
                .iter()
                .filter(|x| x.as_slice() == [0.0, 3.0])
                .count(),
            1,
            "the matching consecutive proposal is the same measured point, not another callback"
        );
        assert_eq!(actual.len(), if complete { 9 } else { 4 });
    }
}
