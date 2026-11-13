use super::*;
fn reflective_options(source: usize, direct: bool) -> Value {
    let mut fields = vec![
        ("Algorithm", Value::from("trust-region-reflective")),
        ("Display", Value::from("off")),
        ("SpecifyObjectiveGradient", Value::Bool(true)),
        ("OptimalityTolerance", Value::Num(1e-7)),
        ("ConstraintTolerance", Value::Num(1e-7)),
        ("StepTolerance", Value::Num(1e-10)),
        ("FunctionTolerance", Value::Num(0.0)),
    ];
    if source == 0 {
        fields.push((
            "HessPattern",
            Value::SparseTensor(
                runmat_value::SparseTensor::new_logical(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
            ),
        ));
    } else if source == 1 {
        fields.push(("HessianFcn", Value::from("objective")));
        fields.push(("HessianMultiplyFcn", Value::from("dormant, not a callback")));
        fields.push(("HessPattern", Value::from("dormant, not a matrix")));
    } else {
        fields.push((
            "HessianMultiplyFcn",
            Value::FunctionHandle("reflective_product".into()),
        ));
    }
    if direct {
        fields.push(("SubproblemAlgorithm", Value::from("factorization")));
    }
    structure(fields)
}

#[test]
fn equality_qr_optional_context_uses_the_same_factor_and_original_rows() {
    use crate::builtins::math::linalg::ops::sparse_factor::equality_factor::EqualityFactor;
    let runtime = runtime();
    for context in [None, Some(&runtime)] {
        block_on(async {
            // Redundant rows [1 1; 2 2], represented once in CSC.
            let factor = EqualityFactor::factor(
                2,
                2,
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &[1.0, 2.0, 1.0, 2.0],
                context,
                IDENTITY,
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(factor.rank(), 1);
            let x = factor.solve(&[2.0, 4.0], false).await.unwrap().unwrap();
            assert!((x[0] + x[1] - 2.0).abs() < 1e-12);
            assert!((2.0 * x[0] + 2.0 * x[1] - 4.0).abs() < 1e-12);
            let multipliers = factor.solve(&[3.0, 3.0], true).await.unwrap().unwrap();
            assert!((multipliers[0] + 2.0 * multipliers[1] - 3.0).abs() < 1e-12);
        });
    }
}

#[test]
fn reflective_bounds_equalities_sparse_models_and_original_hinfo_are_real() {
    runmat_gc::gc_test_context(|| {
        for (bounded, source, direct) in [
            (true, 0, false),
            (true, 1, true),
            (true, 2, false),
            (false, 1, false),
            (false, 1, true),
            (false, 2, false),
        ] {
            let runtime = runtime();
            runtime.set_runmat_extensions_enabled(false);
            let _entry = runtime.enter();
            let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
                &runtime,
                &[("reflective_objective", 2231), ("reflective_product", 2232)],
            )
            .unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let products = Arc::new(AtomicUsize::new(0));
            let counted = calls.clone();
            let product_counted = products.clone();
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |id, args, requested| {
                    if id == 2232 {
                        product_counted.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(requested, 1);
                        let Value::Struct(info) = &args[0] else {
                            panic!("original Hinfo")
                        };
                        assert_eq!(numbers(&info.fields["diagonal"]), vec![2.0, 2.0]);
                        let y = numbers(&args[1]);
                        let out = array(y.into_iter().map(|y| 2.0 * y).collect(), vec![2, 1]);
                        return Box::pin(async move { result(vec![out]) });
                    }
                    assert_eq!(id, 2231);
                    assert_eq!(requested, if source == 0 { 2 } else { 3 });
                    assert!(matches!(&args[0],Value::Tensor(value)if value.shape()==[1,2]));
                    counted.fetch_add(1, Ordering::SeqCst);
                    let x = numbers(&args[0]);
                    let mut out = vec![
                        Value::Num((x[0] - 3.0).powi(2) + (x[1] - 2.0).powi(2)),
                        array(vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 2.0)], vec![2, 1]),
                    ];
                    let allocation = if source == 2 {
                        Some(
                            runmat_gc::gc_allocate_rooted(Value::from(
                                "original reflective Hinfo descendant",
                            ))
                            .unwrap(),
                        )
                    } else {
                        None
                    };
                    if source == 1 {
                        out.push(Value::SparseTensor(
                            runmat_value::SparseTensor::new(
                                2,
                                2,
                                vec![0, 1, 2],
                                vec![0, 1],
                                vec![2.0, 2.0],
                            )
                            .unwrap(),
                        ));
                    }
                    if source == 2 {
                        let mut info = runmat_value::UnboundedStructValueBuilder::new();
                        info.insert("diagonal", array(vec![2.0, 2.0], vec![2, 1]));
                        info.insert("at", array(x, vec![1, 2]));
                        info.insert(
                            "descendant",
                            Value::HandleObject(runmat_value::HandleRef {
                                class_name: "ReflectiveHinfoFixture".into(),
                                target: allocation.as_ref().unwrap().handle(),
                                valid: true,
                            }),
                        );
                        out.push(Value::Struct(info.finish()));
                    }
                    Box::pin(async move {
                        let returned = result(out);
                        drop(allocation);
                        returned
                    })
                }),
            ))
            .unwrap();
            let options = reflective_options(source, direct);
            let input = if bounded {
                args(vec![
                    Value::FunctionHandle("reflective_objective".into()),
                    array(vec![-2.0, 3.0], vec![1, 2]),
                    empty(),
                    empty(),
                    empty(),
                    empty(),
                    array(vec![0.0, 0.0], vec![2, 1]),
                    array(vec![1.0, 4.0], vec![2, 1]),
                    empty(),
                    options,
                ])
            } else {
                // Redundant ORIGINAL equality rows retain both public dual identities.
                args(vec![structure([
                    ("solver", Value::from("fmincon")),
                    (
                        "objective",
                        Value::FunctionHandle("reflective_objective".into()),
                    ),
                    ("x0", array(vec![-2.0, 3.0], vec![1, 2])),
                    (
                        "Aeq",
                        Value::SparseTensor(
                            runmat_value::SparseTensor::new(
                                2,
                                2,
                                vec![0, 2, 4],
                                vec![0, 1, 0, 1],
                                vec![1.0, 2.0, 1.0, 2.0],
                            )
                            .unwrap(),
                        ),
                    ),
                    ("beq", array(vec![2.0, 4.0], vec![2, 1])),
                    ("options", options),
                ])])
            };
            let _count = crate::output_count::push_output_count(Some(7)).unwrap();
            let out = block_on(execute(Some(&runtime), input)).unwrap();
            let x = numbers(&out.as_slice()[0]);
            let expected = if bounded { [1.0, 2.0] } else { [1.5, 0.5] };
            assert!(
                x.iter()
                    .zip(expected)
                    .all(|(&x, expected)| (x - expected).abs() < 1e-4),
                "{bounded}/{source}/{direct}: {x:?}"
            );
            assert!(matches!(&out.as_slice()[0],Value::Tensor(value)if value.shape()==[1,2]));
            assert!(numbers(&out.as_slice()[2])[0] > 0.0);
            let Value::Struct(info) = &out.as_slice()[3] else {
                panic!("information")
            };
            assert_eq!(
                info.fields["algorithm"],
                Value::from("trust-region-reflective")
            );
            assert_eq!(
                info.fields["funcCount"],
                Value::Num(calls.load(Ordering::SeqCst) as f64)
            );
            assert_eq!(info.fields["objectiveCallCount"], info.fields["funcCount"]);
            assert_eq!(info.fields["constraintCallCount"], Value::Num(0.0));
            if direct {
                assert_eq!(info.fields["cgiterations"], Value::Num(0.0));
            }
            let Value::Struct(lambda) = &out.as_slice()[4] else {
                panic!("actual original lambda")
            };
            if bounded {
                assert!((numbers(&lambda.fields["upper"])[0] - 4.0).abs() < 1e-4);
                assert!(numbers(&lambda.fields["lower"])
                    .iter()
                    .all(|value| value.abs() < 1e-5));
            } else {
                let dual = numbers(&lambda.fields["eqlin"]);
                assert_eq!(dual.len(), 2);
                assert!((dual[0] + 2.0 * dual[1] - 3.0).abs() < 1e-5);
            }
            if source == 2 {
                assert!(products.load(Ordering::SeqCst) > 0);
                runmat_gc::gc_collect_major().unwrap();
                let Value::Struct(info) = &out.as_slice()[6] else {
                    panic!("last-used Hinfo")
                };
                let Value::HandleObject(descendant) = &info.fields["descendant"] else {
                    panic!("opaque descendant")
                };
                assert_eq!(
                    runmat_gc::gc_clone_value(&descendant.target).unwrap(),
                    Value::from("original reflective Hinfo descendant")
                );
                assert!(
                    numbers(&info.fields["at"])
                        .iter()
                        .zip(&x)
                        .any(|(&old, &new)| (old - new).abs() > 1e-9),
                    "do not return the unused final-point Hinfo generation"
                );
            } else {
                assert!(matches!(&out.as_slice()[6], Value::SparseTensor(_)));
            }
        }
    });
}

#[test]
fn reflected_trajectory_is_not_a_clipped_unconstrained_step() {
    let geometry = model::Geometry {
        variables: 2,
        inequality: model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 2)),
            rhs: Vec::new(),
        },
        equality: model::LinearConstraints {
            matrix: clarabel::algebra::CscMatrix::zeros((0, 2)),
            rhs: Vec::new(),
        },
        lower: vec![0.0, 0.0],
        upper: vec![1.0, 1.0],
    };
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, None);
    let p = block_on(reflective::reflected(
        &[0.75, 0.25],
        &[1.0, 0.2],
        &geometry,
        &mut cooperation,
    ))
    .unwrap()
    .unwrap();
    assert!((p[0] + 0.5).abs() < 1e-14 && (p[1] - 0.2).abs() < 1e-14);
}

#[test]
fn reflective_restrictions_hard_budget_and_original_failures_are_not_replayed() {
    let runtime = runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[
            ("reflective_failure", 2233),
            ("reflective_info", 2234),
            ("reflective_bad_product", 2235),
        ],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let product_calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let product_counted = product_calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, arguments, requested| {
            if id == 2235 {
                product_counted.fetch_add(1, Ordering::SeqCst);
                return Box::pin(async {
                    Err(crate::build_runtime_error("original product failure")
                        .with_identifier("ReflectiveFixture:ProductFailure")
                        .build())
                });
            }
            counted.fetch_add(1, Ordering::SeqCst);
            if id == 2233 {
                return Box::pin(async {
                    Err(crate::build_runtime_error("original objective failure")
                        .with_identifier("ReflectiveFixture:ObjectiveFailure")
                        .build())
                });
            }
            assert_eq!(id, 2234);
            assert_eq!(requested, 3);
            let x = numbers(&arguments[0])[0];
            Box::pin(async move {
                result(vec![
                    Value::Num((x - 1.0).powi(2)),
                    Value::Num(2.0 * (x - 1.0)),
                    structure([("opaque", Value::from("original Hinfo"))]),
                ])
            })
        },
    )))
    .unwrap();
    let plain = || {
        structure([
            ("Algorithm", Value::from("trust-region-reflective")),
            ("SpecifyObjectiveGradient", Value::Bool(true)),
            ("Display", Value::from("off")),
        ])
    };
    let input = |options| {
        vec![
            Value::FunctionHandle("reflective_failure".into()),
            Value::Num(0.0),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            empty(),
            options,
        ]
    };
    let mut cases = vec![input(structure([(
        "Algorithm",
        Value::from("trust-region-reflective"),
    )]))];
    let mut linear = input(plain());
    linear[2] = Value::Num(1.0);
    linear[3] = Value::Num(1.0);
    cases.push(linear);
    let mut nonlinear = input(plain());
    nonlinear[8] = Value::FunctionHandle("reflective_failure".into());
    cases.push(nonlinear);
    let mut fixed = input(plain());
    fixed[6] = Value::Num(0.0);
    fixed[7] = Value::Num(0.0);
    cases.push(fixed);
    let mut mixed = input(plain());
    mixed[4] = Value::Num(1.0);
    mixed[5] = Value::Num(0.0);
    mixed[6] = Value::Num(-1.0);
    mixed[7] = Value::Num(1.0);
    cases.push(mixed);
    cases.push(input(structure([
        ("Algorithm", Value::from("trust-region-reflective")),
        ("SpecifyObjectiveGradient", Value::Bool(true)),
        ("SubproblemAlgorithm", Value::from("factorization")),
        (
            "HessianMultiplyFcn",
            Value::FunctionHandle("reflective_bad_product".into()),
        ),
    ])));
    cases.push(input(structure([
        ("SpecifyObjectiveGradient", Value::Bool(true)),
        ("HessianFcn", Value::from("objective")),
    ])));
    let _count = crate::output_count::push_output_count(Some(7)).unwrap();
    for input in cases {
        let error = block_on(execute(Some(&runtime), args(input))).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:fmincon:InvalidArgument"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(product_calls.load(Ordering::SeqCst), 0);
    }
    let stopped = block_on(execute(
        Some(&runtime),
        args(input(structure([
            ("Algorithm", Value::from("trust-region-reflective")),
            ("SpecifyObjectiveGradient", Value::Bool(true)),
            ("Display", Value::from("off")),
            ("MaxFunEvals", Value::Num(0.0)),
        ]))),
    ))
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(numbers(&stopped.as_slice()[2]), vec![0.0]);
    for index in [1, 4, 5, 6] {
        assert!(matches!(&stopped.as_slice()[index],Value::Tensor(value)if value.is_empty()));
    }
    let error = block_on(execute(Some(&runtime), args(input(plain())))).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("ReflectiveFixture:ObjectiveFailure")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut product = input(structure([
        ("Algorithm", Value::from("trust-region-reflective")),
        ("SpecifyObjectiveGradient", Value::Bool(true)),
        ("Display", Value::from("off")),
        (
            "HessianMultiplyFcn",
            Value::FunctionHandle("reflective_bad_product".into()),
        ),
    ]));
    product[0] = Value::FunctionHandle("reflective_info".into());
    let error = block_on(execute(Some(&runtime), args(product))).unwrap_err();
    assert_eq!(error.identifier(), Some("ReflectiveFixture:ProductFailure"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(product_calls.load(Ordering::SeqCst), 1);
}
