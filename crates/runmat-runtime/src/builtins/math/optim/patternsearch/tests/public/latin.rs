use super::*;
use std::collections::BTreeSet;
use std::sync::Mutex;

#[test]
fn latin_search_preserves_originating_callback_failure_without_poll_or_replay() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29223)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, _, width| {
            assert_eq!((id, width), (29223, 1));
            let ordinal = observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if ordinal == 0 {
                    returned(vec![Value::Num(0.0)])
                } else {
                    Err(
                        crate::build_runtime_error("original search callback failure")
                            .with_identifier("LatinOriginal:Failure")
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build(),
                    )
                }
            })
        },
    )))
    .unwrap();
    let options = structure([
        ("Display", Value::from("off")),
        ("SearchFcn", Value::from("searchlhs")),
    ]);
    let args = arguments(vec![
        Value::FunctionHandle("pattern_objective".into()),
        Value::Num(0.0),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::Num(0.0),
        Value::Num(1.0),
        empty(),
        options,
    ]);
    let error = block_on(registered(Some(runtime.clone()), args)).unwrap_err();
    assert_eq!(error.identifier(), Some("LatinOriginal:Failure"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
#[test]
fn latin_search_returns_actual_best_samples_and_honors_complete_and_hard_limits() {
    for (complete, vectorized) in [(false, false), (true, false), (true, true)] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29221)],
        )
        .unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let callbacks = Arc::new(AtomicUsize::new(0));
        let invocations = callbacks.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!((id, width), (29221, 1));
                let values = numbers(&args[0]);
                assert!(values.iter().all(|x| (0.0..=1.0).contains(x)));
                invocations.fetch_add(1, Ordering::SeqCst);
                observed.lock().unwrap().extend_from_slice(&values);
                let result = if vectorized {
                    let Value::Tensor(argument) = &args[0] else {
                        panic!("actual vectorized carrier")
                    };
                    assert_eq!(argument.shape(), [values.len(), 1]);
                    let n = values.len();
                    array(values.into_iter().map(|x| -x).collect(), vec![n, 1])
                } else {
                    Value::Num(-values[0])
                };
                Box::pin(async move { returned(vec![result]) })
            },
        )))
        .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
        let limit = if complete { 8.0 } else { 2.0 };
        let search = Value::Cell(
            runmat_value::CellArray::new(
                vec![Value::from("searchlhs"), Value::Num(1.0), Value::Num(7.0)],
                1,
                3,
            )
            .unwrap(),
        );
        let options = structure([
            ("Display", Value::from("off")),
            ("SearchFcn", search),
            ("UseCompleteSearch", Value::Bool(complete)),
            ("UseVectorized", Value::Bool(vectorized)),
            ("MaxFunctionEvaluations", Value::Num(limit)),
        ]);
        let args = arguments(vec![
            Value::FunctionHandle("pattern_objective".into()),
            Value::Num(0.0),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(0.0),
            Value::Num(1.0),
            empty(),
            options,
        ]);
        let output = block_on(execute(Some(&runtime), args)).unwrap();
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), limit as usize);
        assert_eq!(
            callbacks.load(Ordering::SeqCst),
            if vectorized { 2 } else { limit as usize }
        );
        let expected = calls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(numbers(&output.as_slice()[0]), [expected]);
        assert_eq!(numbers(&output.as_slice()[1]), [-expected]);
        assert_eq!(numbers(&output.as_slice()[2]), [0.0]);
        if complete {
            let strata = calls[1..]
                .iter()
                .map(|x| (x * 7.0).floor() as usize)
                .collect::<BTreeSet<_>>();
            assert_eq!(strata, (0..7).collect());
        }
        let Value::Struct(info) = &output.as_slice()[3] else {
            panic!("output metadata")
        };
        assert_eq!(numbers(info.fields.get("funccount").unwrap()), [limit]);
    }
}

#[test]
fn one_sided_latin_search_preserves_original_bounds_and_reports_measured_values() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29222)],
    )
    .unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let observed = count.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            assert_eq!((id, width), (29222, 1));
            let x = numbers(&args[0])[0];
            assert!((2.0..=4.0).contains(&x));
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { returned(vec![Value::Num(-x)]) })
        },
    )))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let options = structure([
        ("Display", Value::from("off")),
        ("SearchFcn", Value::from("searchlhs")),
        ("UseCompleteSearch", Value::Bool(true)),
        ("MaxFunctionEvaluations", Value::Num(16.0)),
    ]);
    let args = arguments(vec![
        Value::FunctionHandle("pattern_objective".into()),
        Value::Num(2.5),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::Num(2.0),
        empty(),
        empty(),
        options,
    ]);
    let output = block_on(execute(Some(&runtime), args)).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 16);
    let x = numbers(&output.as_slice()[0])[0];
    assert!(x > 3.8 && x <= 4.0);
    assert_eq!(numbers(&output.as_slice()[1]), [-x]);
    assert_eq!(numbers(&output.as_slice()[2]), [0.0]);
}
