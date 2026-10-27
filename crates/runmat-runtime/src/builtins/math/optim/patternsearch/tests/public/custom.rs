//! Actual public SearchFcn -> scoped merit -> original objective/constraint route.
use super::*;
use std::sync::Mutex;
#[test]
fn custom_search_uses_actual_merit_counts_original_owner_and_no_replay() {
    for mode in 0..7 {
        let multiple_points = mode >= 5;
        let columns = mode == 6;
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let foreign = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[
                ("pattern_objective", 29301),
                ("actual_search", 29302),
                ("actual_constraints", 29303),
            ],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let scores = Arc::new(Mutex::new(Vec::new()));
        let observed = scores.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                if id == 29301 {
                    assert_eq!(width, 1);
                    let x = numbers(&args[0]);
                    let point_count = if multiple_points {
                        let Value::Tensor(input) = &args[0] else {
                            panic!("vectorized objective requires its actual matrix")
                        };
                        let shape = input.shape();
                        assert_eq!(shape.len(), 2);
                        assert_eq!(shape[if columns { 0 } else { 1 }], 2);
                        shape[if columns { 1 } else { 0 }]
                    } else {
                        1
                    };
                    let ordinal = counted.fetch_add(point_count, Ordering::SeqCst);
                    let value = if multiple_points {
                        let mut values = Vec::new();
                        for point in 0..point_count {
                            let (first, second) = if columns {
                                (x[2 * point], x[2 * point + 1])
                            } else {
                                (x[point], x[point + point_count])
                            };
                            values.push((first - 0.25).powi(2) + (second - 0.5).powi(2));
                        }
                        array(
                            values,
                            if columns {
                                vec![1, point_count]
                            } else {
                                vec![point_count, 1]
                            },
                        )
                    } else if mode == 1 {
                        assert!(matches!(&args[0],Value::Tensor(value) if value.shape()==[1,1]));
                        array(vec![(x[0] - 0.25).powi(2)], vec![1, 1])
                    } else {
                        Value::Num((x[0] - 0.25).powi(2))
                    };
                    return Box::pin(async move {
                        if mode == 3 && ordinal > 0 {
                            Err(crate::build_runtime_error("original objective failed")
                                .with_identifier("CustomSearch:Original")
                                .build())
                        } else {
                            returned(vec![value])
                        }
                    });
                }
                if id == 29303 {
                    assert_eq!(width, 2);
                    return Box::pin(std::future::ready(returned(vec![
                        empty(),
                        Value::Num(numbers(&args[0])[0] - 0.5),
                    ])));
                }
                assert_eq!((id, width), (29302, 4));
                assert_eq!(args.len(), 10);
                assert!(matches!(&args[0], Value::ScopedFunctionHandle(_)));
                let fun = args[0].clone();
                let original =
                    crate::context::legacy::active().expect("selected original poll scope");
                let observed = observed.clone();
                Box::pin(async move {
                    let trial_points = if multiple_points {
                        if columns {
                            array(vec![0.25, 0.5, 1.0, 1.0, -1.0, -1.0], vec![2, 3])
                        } else {
                            array(vec![0.25, 1.0, -1.0, 0.5, 1.0, -1.0], vec![3, 2])
                        }
                    } else {
                        Value::Num(0.25)
                    };
                    let measured =
                        crate::builtins::math::optim::common::call_function_sequence_in_context(
                            Some(&original),
                            &fun,
                            [trial_points],
                            1,
                        )
                        .await?
                        .try_into_single()?;
                    let measured_scores = numbers(measured.value());
                    if multiple_points {
                        let Value::Tensor(value) = measured.value() else {
                            panic!("the merit batch retains its point axis")
                        };
                        assert_eq!(value.shape(), if columns { &[1, 3] } else { &[3, 1] });
                        assert_eq!(measured_scores, [0.0, 0.8125, 3.8125]);
                    }
                    let score = measured_scores[0];
                    observed.lock().unwrap().push(score);
                    returned(vec![
                        Value::Bool(true),
                        if multiple_points {
                            array(
                                vec![0.25, 0.5],
                                if columns { vec![2, 1] } else { vec![1, 2] },
                            )
                        } else {
                            Value::Num(0.25)
                        },
                        Value::Num(score),
                        Value::Num(if mode == 4 {
                            0.0
                        } else if multiple_points {
                            3.0
                        } else {
                            1.0
                        }),
                    ])
                })
            },
        )))
        .unwrap();
        let options = structure([
            ("Display", Value::from("off")),
            ("SearchFcn", Value::FunctionHandle("actual_search".into())),
            ("MaxIterations", Value::Num(1.0)),
            (
                "MaxFunctionEvaluations",
                Value::Num(if multiple_points { 4.0 } else { 2.0 }),
            ),
            ("UseVectorized", Value::Bool(mode == 1 || multiple_points)),
        ]);
        let args = arguments(vec![
            Value::FunctionHandle("pattern_objective".into()),
            if multiple_points {
                array(
                    vec![0.0, 0.0],
                    if columns { vec![2, 1] } else { vec![1, 2] },
                )
            } else {
                Value::Num(0.0)
            },
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            if mode == 2 {
                Value::FunctionHandle("actual_constraints".into())
            } else {
                empty()
            },
            options,
        ]);
        let _width = crate::output_count::push_output_count(Some(4)).unwrap();
        let _ambient = foreign.enter();
        let output = block_on(registered(Some(runtime.clone()), args));
        if mode == 4 {
            let error = output.unwrap_err();
            assert_eq!(
                error.identifier(),
                runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT.identifier
            );
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        } else if mode == 3 {
            assert_eq!(
                output.unwrap_err().identifier(),
                Some("CustomSearch:Original")
            );
        } else {
            let output = output.unwrap();
            assert_eq!(
                numbers(&output.as_slice()[0]),
                if multiple_points {
                    vec![0.25, 0.5]
                } else {
                    vec![0.25]
                },
            );
            assert_eq!(numbers(&output.as_slice()[1]), [0.0]);
            assert_eq!(scores.lock().unwrap().len(), 1);
            if mode == 2 {
                assert!(
                    scores.lock().unwrap()[0] > 0.0,
                    "the issued function is AL merit, not raw f"
                );
            } else {
                assert_eq!(scores.lock().unwrap()[0], 0.0);
            }
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            if multiple_points { 4 } else { 2 },
            "one initial point plus only the actual search trials"
        );
    }
}
