use super::*;

#[test]
fn nonlinear_lattice_callbacks_share_the_original_hard_point_budget() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29219), ("pattern_constraint", 29220)],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let constraints = Arc::new(AtomicUsize::new(0));
    let objective_calls = calls.clone();
    let constraint_calls = constraints.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, width| {
            let x = numbers(&args[0])[0];
            let values = match id {
                29219 => {
                    assert_eq!(width, 1);
                    objective_calls.fetch_add(1, Ordering::SeqCst);
                    vec![Value::Num((x - 1.0).powi(2))]
                }
                29220 => {
                    assert_eq!(width, 2);
                    constraint_calls.fetch_add(1, Ordering::SeqCst);
                    vec![Value::Num(x * x - 0.25), empty()]
                }
                _ => panic!("original selected callback identity"),
            };
            Box::pin(async move { returned(values) })
        },
    )))
    .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
    let options = structure([
        ("Display", Value::from("off")),
        ("PollMethod", Value::from("MADSPositiveBasisNp1")),
        ("ScaleMesh", Value::Bool(false)),
        ("MaxFunctionEvaluations", Value::Num(12.0)),
    ]);
    let args = arguments(vec![
        Value::FunctionHandle("pattern_objective".into()),
        Value::Num(0.0),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        empty(),
        Value::FunctionHandle("pattern_constraint".into()),
        options,
    ]);
    let output = block_on(execute(Some(&runtime), args)).unwrap();
    let Value::Struct(info) = &output.as_slice()[3] else {
        panic!("output metadata")
    };
    assert!(calls.load(Ordering::SeqCst) > 1 && calls.load(Ordering::SeqCst) <= 12);
    assert_eq!(
        constraints.load(Ordering::SeqCst),
        calls.load(Ordering::SeqCst)
    );
    assert_eq!(
        numbers(info.fields.get("funccount").unwrap()),
        [calls.load(Ordering::SeqCst) as f64]
    );
    assert_eq!(
        numbers(info.fields.get("constraintcallcount").unwrap()),
        [constraints.load(Ordering::SeqCst) as f64]
    );
    let x = numbers(&output.as_slice()[0])[0];
    assert_eq!(numbers(&output.as_slice()[1]), [(x - 1.0).powi(2)]);
}

#[test]
fn lattice_methods_use_whole_feasible_points_and_poll_measure_stopping() {
    for method in [
        "MADSPositiveBasis2N",
        "MADSPositiveBasisNp1",
        "OrthoMADSPositiveBasis2N",
        "OrthoMADSPositiveBasisNp1",
    ] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("pattern_objective", 29217)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, width| {
                assert_eq!((id, width), (29217, 1));
                let x = numbers(&args[0])[0];
                assert!(
                    x - 1.0 <= 1e-9,
                    "only original-row feasible mesh points reach the objective"
                );
                observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move { returned(vec![Value::Num((x - 2.0).powi(2))]) })
            },
        )))
        .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(4)).unwrap();
        let opts = structure([
            ("Display", Value::from("off")),
            ("PollMethod", Value::from(method)),
            ("ScaleMesh", Value::Bool(false)),
            ("MeshTolerance", Value::Num(0.001)),
            ("ConstraintTolerance", Value::Num(1e-9)),
            ("MaxFunctionEvaluations", Value::Num(1000.0)),
            // Detailed primary applicability makes these dormant for MADS.
            (
                "StepTolerance",
                Value::FunctionHandle("dormant_step".into()),
            ),
            (
                "FunctionTolerance",
                Value::FunctionHandle("dormant_function".into()),
            ),
            (
                "PollOrderAlgorithm",
                Value::FunctionHandle("dormant_order".into()),
            ),
            (
                "MeshExpansionFactor",
                Value::FunctionHandle("dormant_expansion".into()),
            ),
        ]);
        let args = arguments(vec![
            Value::FunctionHandle("pattern_objective".into()),
            Value::Num(0.0),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            Value::Num(1.0),
            empty(),
            opts,
        ]);
        let output = block_on(execute(Some(&runtime), args)).unwrap();
        let x = numbers(&output.as_slice()[0])[0];
        assert!(x - 1.0 <= 1e-9 && x > 0.99);
        assert_eq!(numbers(&output.as_slice()[1]), [(x - 2.0).powi(2)]);
        assert_eq!(numbers(&output.as_slice()[2]), [1.0]);
        let Value::Struct(info) = &output.as_slice()[3] else {
            panic!("output metadata")
        };
        let mesh = numbers(info.fields.get("meshsize").unwrap())[0];
        assert!(mesh.sqrt() <= 0.001);
        assert_eq!(
            numbers(info.fields.get("funccount").unwrap()),
            [calls.load(Ordering::SeqCst) as f64]
        );
    }
}

#[test]
fn mads_equalities_refuse_before_callbacks_without_relabeling_a_gps_method() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("pattern_objective", 29218)],
    )
    .unwrap();
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("unsupported linear equality must fail before objective")
        })))
        .unwrap();
    let options = structure([
        ("PollMethod", Value::from("MADSPositiveBasis2N")),
        ("Display", Value::from("off")),
    ]);
    let args = arguments(vec![
        Value::FunctionHandle("pattern_objective".into()),
        array(vec![0.0, 0.0], vec![1, 2]),
        empty(),
        empty(),
        array(vec![1.0, 0.0], vec![1, 2]),
        Value::Num(0.0),
        empty(),
        empty(),
        empty(),
        options,
    ]);
    let error = block_on(registered(Some(runtime.clone()), args)).unwrap_err();
    assert!(error.message().contains("linear equality"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}
