use super::*;

#[test]
fn nonuniform_algorithms_use_original_callbacks_budgets_and_ignore_dormant_poll_controls() {
    for algorithm in ["nups", "nups-gps", "nups-mads"] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29225)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!((id, width), (29225, 1));
                counted.fetch_add(1, Ordering::SeqCst);
                let x = numbers(&args[0])[0];
                assert!((-1e-6..=1.0 + 1e-6).contains(&x));
                Box::pin(async move { returned(vec![Value::Num((x - 2.0).powi(2))]) })
            },
        )))
        .unwrap();
        let options = structure([
            ("Algorithm", Value::from(algorithm)),
            ("Display", Value::from("off")),
            ("ScaleMesh", Value::Bool(false)),
            ("MaxFunctionEvaluations", Value::Num(12.0)),
            (
                "PollMethod",
                structure([("unread", Value::from("not a method"))]),
            ),
            ("PollOrderAlgorithm", Value::Num(f64::NAN)),
            ("UseCompletePoll", Value::Num(f64::NAN)),
            ("MeshExpansionFactor", Value::Num(f64::NAN)),
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
        let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
        let output = block_on(execute(Some(&runtime), args)).unwrap();
        assert_eq!(output.len(), 4);
        let x = numbers(&output.as_slice()[0])[0];
        assert!((x - 1.0).abs() < 1e-6);
        assert_eq!(numbers(&output.as_slice()[1]), [(x - 2.0).powi(2)]);
        let Value::Struct(info) = &output.as_slice()[3] else {
            panic!("original output metadata")
        };
        assert_eq!(
            numbers(info.fields.get("funccount").unwrap()),
            [calls.load(Ordering::SeqCst) as f64]
        );
        assert!(calls.load(Ordering::SeqCst) <= 12);
        assert_eq!(
            info.fields.get("pollmethod").unwrap(),
            &Value::from(algorithm)
        );
    }
}

#[test]
fn random_nonuniform_equalities_and_poll_search_are_refused_before_objective_calls() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    for (algorithm, equality, search) in [
        ("nups-mads", true, empty()),
        ("nups-gps", false, Value::from("GPSPositiveBasis2N")),
    ] {
        let options = structure([
            ("Algorithm", Value::from(algorithm)),
            ("Display", Value::from("off")),
            ("SearchFcn", search),
        ]);
        let args = arguments(vec![
            Value::FunctionHandle("not_called".into()),
            Value::Num(0.0),
            empty(),
            empty(),
            if equality { Value::Num(1.0) } else { empty() },
            if equality { Value::Num(0.0) } else { empty() },
            empty(),
            empty(),
            empty(),
            options,
        ]);
        let error = block_on(registered(Some(runtime.clone()), args)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:patternsearch:UnsupportedOption")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
}
