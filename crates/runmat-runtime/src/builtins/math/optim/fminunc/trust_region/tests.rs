use super::*;
use futures::executor::block_on;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn array(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(value) => value.materialize_f64().clone(),
        _ => panic!("real array"),
    }
}
fn outputs(values: Vec<Value>) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let sequence = crate::sequence::ValueSequence::comma_separated(values)
        .map_err(crate::sequence::sequence_error_to_runtime)?;
    crate::execution::RootedValueSequence::new(sequence)
        .map_err(|source| source.into_runtime_error())
}
fn settings() -> runmat_value::UnboundedStructValueBuilder {
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert("Algorithm", Value::from("trust-region"));
    options.insert("SpecifyObjectiveGradient", Value::Bool(true));
    options.insert("OptimalityTolerance", Value::Num(1e-10));
    options.insert("FunctionTolerance", Value::Num(0.0));
    options.insert("StepTolerance", Value::Num(0.0));
    options.insert("MaxPCGIter", Value::Num(2.0));
    options
}

#[test]
fn actual_objective_hessian_sparse_direct_and_colored_gradient_models_reach_quadratic_minimum() {
    runmat_gc::gc_test_context(|| {
        for mode in 0..5 {
            let runtime = crate::user_functions::test_support::exact_fixture_runtime();
            let _entry = runtime.enter();
            let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
                &runtime,
                &[("tr_quadratic", 1601), ("tr_quadratic_product", 1605)],
            )
            .unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let counted = calls.clone();
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                Arc::new(move |id, args, requested| {
                    if id == 1605 {
                        assert_eq!(requested, 1);
                        let Value::Struct(info) = &args[0] else {
                            panic!("original Hinfo")
                        };
                        let diagonal = values(&info.fields["diagonal"]);
                        let y = values(&args[1]);
                        let value = array(vec![diagonal[0] * y[0], diagonal[1] * y[1]], vec![2, 1]);
                        return Box::pin(async move { outputs(vec![value]) });
                    }
                    assert_eq!(id, 1601);
                    assert_eq!(requested, if mode == 2 { 2 } else { 3 });
                    counted.fetch_add(1, Ordering::SeqCst);
                    let x = values(&args[0]);
                    let mut result = vec![
                        Value::Num((x[0] - 1.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2)),
                        array(vec![2.0 * (x[0] - 1.0), 4.0 * (x[1] + 1.0)], vec![2, 1]),
                    ];
                    if mode == 0 {
                        result.push(array(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]));
                    }
                    if mode == 1 {
                        result.push(Value::SparseTensor(
                            runmat_value::SparseTensor::new(
                                2,
                                2,
                                vec![0, 1, 2],
                                vec![0, 1],
                                vec![2.0, 4.0],
                            )
                            .unwrap(),
                        ));
                    }
                    let allocation = if mode >= 3 {
                        Some(
                            runmat_gc::gc_allocate_rooted(Value::from("original Hinfo descendant"))
                                .unwrap(),
                        )
                    } else {
                        None
                    };
                    if mode >= 3 {
                        let mut info = runmat_value::UnboundedStructValueBuilder::new();
                        info.insert("diagonal", array(vec![2.0, 4.0], vec![2, 1]));
                        info.insert(
                            "descendant",
                            Value::HandleObject(runmat_value::HandleRef {
                                class_name: "TrustHinfoFixture".into(),
                                target: allocation.as_ref().unwrap().handle(),
                                valid: true,
                            }),
                        );
                        result.push(Value::Struct(info.finish()));
                    }
                    Box::pin(async move {
                        let result = outputs(result);
                        drop(allocation);
                        result
                    })
                }),
            ))
            .unwrap();
            let mut options = settings();
            if mode < 2 {
                options.insert("HessianFcn", Value::from("objective"));
            }
            if mode == 1 || mode == 4 {
                options.insert("SubproblemAlgorithm", Value::from("factorization"));
            }
            if mode >= 3 {
                options.insert(
                    "HessianMultiplyFcn",
                    Value::FunctionHandle("tr_quadratic_product".into()),
                );
            }
            if mode == 2 {
                options.insert(
                    "HessPattern",
                    Value::SparseTensor(
                        runmat_value::SparseTensor::new_logical(2, 2, vec![0, 1, 2], vec![0, 1])
                            .unwrap(),
                    ),
                );
            }
            let _requested = crate::output_count::push_output_count(Some(6)).unwrap();
            let rest =
                crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
            let result = block_on(fminunc_in_runtime(
                Some(&runtime),
                Value::FunctionHandle("tr_quadratic".into()),
                array(vec![-2.0, 3.0], vec![1, 2]),
                rest.into(),
            ))
            .unwrap();
            if mode >= 3 {
                runmat_gc::gc_collect_major().unwrap();
                let Value::Struct(info) = &result.as_slice()[5] else {
                    panic!("rooted Hinfo result")
                };
                let Value::HandleObject(descendant) = &info.fields["descendant"] else {
                    panic!("original GC descendant")
                };
                assert_eq!(
                    runmat_gc::gc_clone_value(&descendant.target).expect(
                        "successor rooted sequence retains Hinfo after original model retirement"
                    ),
                    Value::from("original Hinfo descendant")
                );
            }
            let result = legacy_test_value(result).unwrap();
            let Value::OutputList(result) = result else {
                panic!("six outputs")
            };
            let x = values(&result[0]);
            assert!(
                (x[0] - 1.0).abs() < 1e-7 && (x[1] + 1.0).abs() < 1e-7,
                "mode {mode}: {x:?}"
            );
            assert!(matches!(&result[0],Value::Tensor(value)if value.shape()==[1,2]));
            assert_eq!(result[2], Value::Num(1.0));
            let Value::Struct(info) = &result[3] else {
                panic!("output")
            };
            assert_eq!(
                info.fields["funcCount"],
                Value::Num(calls.load(Ordering::SeqCst) as f64)
            );
            assert_eq!(info.fields["algorithm"], Value::from("trust-region"));
            assert!(info.fields.contains_key("cgiterations"));
            assert!(!info.fields.contains_key("lssteplength"));
            match &result[5] {
                Value::Struct(info) if mode >= 3 => {
                    assert_eq!(values(&info.fields["diagonal"]), vec![2.0, 4.0])
                }
                Value::Tensor(h) => assert!(
                    (crate::builtins::common::tensor::tensor_value_f64(h, 0) - 2.0).abs() < 1e-6
                ),
                Value::SparseTensor(h) => assert!(
                    (h.numeric_host_buffer()
                        .unwrap()
                        .value_at(0)
                        .unwrap()
                        .materialize_f64()
                        - 2.0)
                        .abs()
                        < 1e-6
                ),
                _ => panic!("actual last model"),
            }
        }
    });
}

#[test]
fn negative_curvature_and_direct_hard_case_satisfy_original_trust_ball_model() {
    block_on(async {
        let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Fminunc, None);
        let owner =
            crate::execution::RootedValue::new(array(vec![-2.0, 0.0, 0.0, 1.0], vec![2, 2]))
                .unwrap();
        let mut model = model::Model::matrix(owner, 2, &mut cooperation)
            .await
            .unwrap();
        let mut products = model.products(&mut cooperation).await.unwrap();
        let preconditioner = cg::Preconditioner::new(&mut products, 2, 0, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        let controls = Controls {
            source: ModelSource::Objective,
            maximum_cg_iterations: 2,
            cg_tolerance: 0.1,
            bandwidth: 0,
            subproblem: runmat_builtins::OptimSubproblemAlgorithm::Cg,
        };
        let step = cg::step(
            &mut products,
            &preconditioner,
            &[1.0, 0.0],
            1.0,
            &controls,
            &mut cooperation,
        )
        .await
        .unwrap()
        .unwrap();
        assert!(step.boundary);
        assert!((step.displacement[0] + 1.0).abs() < 1e-12);
        assert_eq!(step.displacement[1], 0.0);
        let direct = factorization::Factorization::new(&mut products, 2, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        let step = direct
            .step(&mut products, &[0.0, 1.0], 1.0, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        assert!(step.boundary);
        assert!((norm(&step.displacement) - 1.0).abs() < 1e-12);
        assert!((step.displacement[1] + 1.0 / 3.0).abs() < 1e-12);
        let p = &step.displacement;
        let q = -p[0] * p[0] + 0.5 * p[1] * p[1] + p[1];
        assert!((q + 7.0 / 6.0).abs() < 1e-12);
        let owner =
            crate::execution::RootedValue::new(array(vec![0.0, 1e16, 1e16, 0.0], vec![2, 2]))
                .unwrap();
        let mut scaled = model::Model::matrix(owner, 2, &mut cooperation)
            .await
            .unwrap();
        let mut scaled_products = scaled.products(&mut cooperation).await.unwrap();
        let band = cg::Preconditioner::new(&mut scaled_products, 2, 1, &mut cooperation)
            .await
            .unwrap()
            .expect("off-diagonal scaled SPD preconditioner");
        let step = cg::step(
            &mut scaled_products,
            &band,
            &[1.0, -1.0],
            1.0,
            &controls,
            &mut cooperation,
        )
        .await
        .unwrap()
        .unwrap();
        assert!(step.boundary);
        assert!((norm(&step.displacement) - 1.0).abs() < 1e-12);
        assert!(step.displacement[0] * step.displacement[1] < 0.0);
        for small in [1e-100, 0.0] {
            let owner =
                crate::execution::RootedValue::new(array(vec![small, 0.0, 0.0, 1e100], vec![2, 2]))
                    .unwrap();
            let mut wide = model::Model::matrix(owner, 2, &mut cooperation)
                .await
                .unwrap();
            let mut wide_products = wide.products(&mut cooperation).await.unwrap();
            let direct = factorization::Factorization::new(&mut wide_products, 2, &mut cooperation)
                .await
                .unwrap()
                .unwrap();
            let step = direct
                .step(&mut wide_products, &[1.0, 1e100], 2.0, &mut cooperation)
                .await
                .unwrap()
                .expect("validated wide-spectrum secular boundary");
            assert!(step.boundary);
            assert!((norm(&step.displacement) - 2.0).abs() < 1e-11);
            assert!((step.displacement[0] + 3.0f64.sqrt()).abs() < 1e-10);
            assert!((step.displacement[1] + 1.0).abs() < 1e-12);
            let multiplier = -1.0 / step.displacement[0] - small;
            assert!((1.0 + (small + multiplier) * step.displacement[0]).abs() < 1e-12);
            assert!((1.0 + (1.0 + multiplier / 1e100) * step.displacement[1]).abs() < 1e-12);
        }
        // One existing borrowed CSC proof is reused for multiple cooperative
        // products. No validator-count hook exists; verify the actual view
        // aliases original coefficients instead of introducing instrumentation.
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperative =
            ScalarCooperation::new(runmat_builtins::OptimSolver::Fminunc, Some(&runtime));
        let n = 4096;
        let sparse = runmat_value::SparseTensor::new(
            n,
            n,
            (0..=n).collect::<Vec<_>>(),
            (0..n).collect::<Vec<_>>(),
            vec![2.0; n],
        )
        .unwrap();
        let owner = crate::execution::RootedValue::in_optional_context(
            Value::SparseTensor(sparse),
            Some(&runtime),
        )
        .unwrap();
        let mut large = model::Model::matrix(owner, n, &mut cooperative)
            .await
            .unwrap();
        {
            let mut borrowed = large.products(&mut cooperative).await.unwrap();
            {
                let model::Products::Matrix {
                    source,
                    operator:
                        crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator::Sparse(
                            view,
                        ),
                    ..
                } = &borrowed
                else {
                    panic!("retained canonical sparse view")
                };
                let Value::SparseTensor(source) = source else {
                    panic!("original sparse backing")
                };
                assert_eq!(
                    view.as_slice().as_ptr(),
                    source.as_f64_slice().unwrap().as_ptr()
                );
            }
            let input = vec![1.0; n];
            let mut output = vec![0.0; n];
            for _ in 0..3 {
                borrowed
                    .apply(&input, &mut output, &mut cooperative)
                    .await
                    .unwrap();
                assert!(output.iter().all(|&value| value == 2.0));
            }
        }
        assert!(matches!(large.into_value().value(),Value::SparseTensor(value)if value.nnz()==n));
    });
}

#[test]
fn model_not_constructed_at_zero_iterations_or_exhausted_gradient_budget_is_empty_without_replay() {
    for zero_iterations in [false, true] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("tr_limited", 1602)],
        )
        .unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let counted = count.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_, args, requested| {
                assert_eq!(requested, 2);
                counted.fetch_add(1, Ordering::SeqCst);
                let x = values(&args[0]);
                Box::pin(
                    async move { outputs(vec![Value::Num(x[0] * x[0]), Value::Num(2.0 * x[0])]) },
                )
            },
        )))
        .unwrap();
        let mut options = settings();
        options.insert(
            if zero_iterations {
                "MaxIterations"
            } else {
                "MaxFunctionEvaluations"
            },
            Value::Num(if zero_iterations { 0.0 } else { 1.0 }),
        );
        let rest =
            crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
        let _requested = crate::output_count::push_output_count(Some(6)).unwrap();
        let result = block_on(fminunc_in_runtime(
            Some(&runtime),
            Value::FunctionHandle("tr_limited".into()),
            Value::Num(1.0),
            rest.into(),
        ))
        .unwrap();
        let result = legacy_test_value(result).unwrap();
        let Value::OutputList(result) = result else {
            panic!("six outputs")
        };
        assert_eq!(result[2], Value::Num(0.0));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(matches!(&result[5],Value::Tensor(value)if value.is_empty()));
        let mut malformed = settings();
        let mut payload = runmat_value::UnboundedStructValueBuilder::new();
        payload.insert("not_a_numeric_matrix", Value::from("opaque aggregate"));
        malformed.insert("HessPattern", Value::Struct(payload.finish()));
        let rest =
            crate::execution::RootedValues::new(vec![Value::Struct(malformed.finish())]).unwrap();
        let error = block_on(fminunc_in_runtime(
            Some(&runtime),
            Value::FunctionHandle("tr_limited".into()),
            Value::Num(1.0),
            rest.into(),
        ))
        .unwrap_err();
        assert!(error.message().contains("HessPattern"));
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "a nonnumeric pattern rejects before objective execution"
        );
    }
}

#[test]
fn opaque_hinfo_product_failure_or_cancellation_preserves_original_error_and_never_replays() {
    for (cancelled, malformed) in [(false, false), (true, false), (false, true)] {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::user_functions::test_support::exact_fixture_runtime()
            .with_cancellation_token(cancellation.token());
        runtime.set_error_namespace("TrustInvocation");
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("tr_opaque", 1603), ("tr_product", 1604)],
        )
        .unwrap();
        let objectives = Arc::new(AtomicUsize::new(0));
        let evaluated = objectives.clone();
        let products = Arc::new(AtomicUsize::new(0));
        let multiplied = products.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                let result = if id == 1603 {
                    assert_eq!(requested, 3);
                    evaluated.fetch_add(1, Ordering::SeqCst);
                    let x = values(&args[0]);
                    let mut info = runmat_value::UnboundedStructValueBuilder::new();
                    info.insert("opaque", Value::from("original Hinfo"));
                    Ok(vec![
                        Value::Num(x.iter().map(|x| x * x).sum()),
                        array(x.iter().map(|x| 2.0 * x).collect(), vec![x.len(), 1]),
                        Value::Struct(info.finish()),
                    ])
                } else {
                    assert_eq!(id, 1604);
                    assert_eq!(requested, 1);
                    multiplied.fetch_add(1, Ordering::SeqCst);
                    let Value::Struct(info) = &args[0] else {
                        panic!("opaque carrier was not normalized")
                    };
                    assert_eq!(info.fields["opaque"], Value::from("original Hinfo"));
                    if malformed {
                        Ok(vec![array(vec![2.0, 2.0], vec![1, 2])])
                    } else if cancelled {
                        cancellation.cancel(runmat_execution::CancellationReason::User);
                        Ok(vec![array(vec![2.0], vec![1, 1])])
                    } else {
                        Err(build_runtime_error("original product failure")
                            .with_identifier("Hessian:Original")
                            .with_source(std::io::Error::other("product source"))
                            .build())
                    }
                };
                Box::pin(async move { outputs(result?) })
            },
        )))
        .unwrap();
        let mut options = settings();
        options.insert(
            "HessianMultiplyFcn",
            Value::FunctionHandle("tr_product".into()),
        );
        let rest =
            crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
        let error = block_on(fminunc_in_runtime(
            Some(&runtime),
            Value::FunctionHandle("tr_opaque".into()),
            if malformed {
                array(vec![1.0, 1.0], vec![2, 1])
            } else {
                Value::Num(1.0)
            },
            rest.into(),
        ))
        .unwrap_err();
        assert_eq!(objectives.load(Ordering::SeqCst), 1);
        assert_eq!(products.load(Ordering::SeqCst), 1);
        if malformed {
            assert_eq!(error.identifier(), Some("RunMat:fminunc:InvalidInput"));
            assert!(error.message().contains("n-by-1 product"));
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        } else if cancelled {
            assert_eq!(error.identifier(), Some("TrustInvocation:Cancelled"));
            assert!(crate::runtime_error::error_has_control_cancellation_origin(
                &error
            ));
        } else {
            assert_eq!(error.identifier(), Some("Hessian:Original"));
            assert_eq!(
                error.gpu_gather_retry,
                crate::GpuGatherRetry::Legacy,
                "original callback error retry policy is not blanket-reclassified"
            );
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some());
        }
    }
}

#[test]
fn optimality_flag_compares_actual_reported_gradient_to_requested_tolerance() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("tr_scaled_exp", 1606)],
    )
    .unwrap();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |id, args, requested| {
            assert_eq!(id, 1606);
            assert_eq!(requested, 3);
            let x = values(&args[0])[0];
            let f = 1e12 * x.exp();
            Box::pin(async move { outputs(vec![Value::Num(f), Value::Num(f), Value::Num(f)]) })
        },
    )))
    .unwrap();
    let mut options = settings();
    options.insert("HessianFcn", Value::from("objective"));
    options.insert("OptimalityTolerance", Value::Num(1e-6));
    options.insert("MaxIterations", Value::Num(20.0));
    let rest = crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
    let _requested = crate::output_count::push_output_count(Some(5)).unwrap();
    let result = block_on(fminunc_in_runtime(
        Some(&runtime),
        Value::FunctionHandle("tr_scaled_exp".into()),
        Value::Num(0.0),
        rest.into(),
    ))
    .unwrap();
    assert_eq!(result.as_slice()[2], Value::Num(0.0));
    let gradient = values(&result.as_slice()[4])[0];
    assert!(gradient>1e-6 && gradient<1e6,"raw gradient {gradient} is not an optimality certificate merely because the initial gradient was large");
    let Value::Struct(info) = &result.as_slice()[3] else {
        panic!("output")
    };
    assert_eq!(info.fields["firstorderopt"], Value::Num(gradient));
    assert_eq!(info.fields["iterations"], Value::Num(20.0));
    assert_eq!(info.fields["funcCount"], Value::Num(21.0));
}
