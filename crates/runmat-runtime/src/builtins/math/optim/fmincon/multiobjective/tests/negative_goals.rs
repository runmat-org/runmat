use super::*;

#[test]
fn mixed_negative_zero_and_first_equality_goals_preserve_public_derivatives() {
    for case in 0..3 {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        runtime.set_runmat_extensions_enabled(true);
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[
                ("signed_goal_objective", 2244),
                ("signed_goal_observer", 2245),
            ],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let observers = observed.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                if id == 2245 {
                    assert_eq!(requested, 1);
                    let Value::Struct(info) = &args[1] else {
                        panic!("original observer metadata")
                    };
                    if let Value::SparseTensor(jacobian) = &info.fields["gradient"] {
                        let expected = match case {
                            0 => vec![1.0, 1.0, 1.0],
                            1 => vec![1.0, -1.0],
                            _ => vec![1.0],
                        };
                        assert_eq!(jacobian.rows, 1);
                        assert_eq!(jacobian.cols, expected.len());
                        assert_eq!(
                            jacobian
                                .numeric_host_buffer()
                                .unwrap()
                                .as_f64_slice()
                                .unwrap(),
                            expected.as_slice()
                        );
                        observers.fetch_add(1, Ordering::SeqCst);
                    }
                    return Box::pin(async { tuple(vec![Value::Bool(false)]) });
                }
                assert_eq!(id, 2244);
                assert_eq!(requested, 2);
                assert_eq!(numbers(&args[0]).len(), 1, "callback never receives gamma");
                counted.fetch_add(1, Ordering::SeqCst);
                let x = numbers(&args[0])[0];
                let (f, j) = match case {
                    0 => (vec![x, x, x], vec![1.0, 1.0, 1.0]),
                    1 => (vec![x, 2.0 - x], vec![1.0, -1.0]),
                    _ => (vec![x], vec![1.0]),
                };
                let m = f.len();
                Box::pin(async move { tuple(vec![array(f, vec![1, m]), array(j, vec![1, m])]) })
            },
        )))
        .unwrap();
        let (goals, weights, expected_x, expected_gamma) = match case {
            0 => (vec![1.0, 3.0, 2.0], vec![2.0, -1.0, 0.0], 2.0, 1.0),
            1 => (vec![1.0, 0.0], vec![-2.0, 1.0], 5.0 / 3.0, 1.0 / 3.0),
            _ => (vec![1.0], vec![-2.0], 3.0, -1.0),
        };
        let m = goals.len();
        let settings = options(
            true,
            vec![
                (
                    "OutputFcn",
                    Value::FunctionHandle("signed_goal_observer".into()),
                ),
                (
                    "EqualityGoalCount",
                    Value::Num(if case == 1 { 1.0 } else { 0.0 }),
                ),
            ],
        );
        let _outputs = crate::output_count::push_output_count(Some(6)).unwrap();
        let out = block_on(execute(
            Some(&runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::FunctionHandle("signed_goal_objective".into()),
                Value::Num(1.5),
                array(goals, vec![m, 1]),
                array(weights, vec![m, 1]),
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                inputs::empty(),
                Value::Num(0.0),
                Value::Num(3.0),
                inputs::empty(),
                settings,
            ]),
            definition(Kind::GoalAttain),
        ))
        .unwrap();
        let out = out.as_slice();
        assert!(numbers(&out[3])[0] > 0.0, "{:?}", out[4]);
        assert!((numbers(&out[0])[0] - expected_x).abs() < 1e-4);
        assert!((numbers(&out[2])[0] - expected_gamma).abs() < 1e-4);
        assert_eq!(
            crate::builtins::math::optim::fmincon::numeric::shape(&out[1]),
            Some(vec![1, m])
        );
        if case == 0 {
            assert!(numbers(&out[1])[2] <= 2.0 + 1e-7, "zero weight stays hard");
        }
        let Value::Struct(info) = &out[4] else {
            panic!("information")
        };
        assert_eq!(
            numbers(&info.fields["objectiveCallCount"])[0] as usize,
            calls.load(Ordering::SeqCst)
        );
        assert_eq!(numbers(&info.fields["constraintCallCount"]), vec![0.0]);
        assert!(observed.load(Ordering::SeqCst) > 0);
        let Value::Struct(lambda) = &out[5] else {
            panic!("original multipliers")
        };
        assert!(
            numbers(&lambda.fields["ineqnonlin"]).is_empty(),
            "generated rows do not alter public lambda units"
        );
        assert_eq!(numbers(&lambda.fields["lower"]).len(), 1);
        if case == 2 {
            assert!(
                (numbers(&lambda.fields["upper"])[0] - 0.5).abs() < 1e-4,
                "original upper-bound dual retains gamma-objective units"
            );
        }
    }
}
