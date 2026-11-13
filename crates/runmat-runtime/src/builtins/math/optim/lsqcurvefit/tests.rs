use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::IntegerStorage;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn one_output_does_not_publish_unused_gradient_or_jacobian() {
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let result = LeastSquaresResult {
        point_evidence: crate::builtins::math::optim::least_squares::PointEvidence::Evaluated,
        gradient: None,
        cg_iterations: 0,
        jacobian_source:
            crate::builtins::math::optim::least_squares::JacobianSource::FiniteDifference,
        method: crate::builtins::math::optim::least_squares::Method::LevenbergMarquardt,
        x: vec![2.0],
        residual: vec![0.0],
        jacobian: crate::builtins::math::optim::least_squares::PublishedJacobian::RowMajor(vec![]),
        residual_len: 1,
        variable_len: 1,
        resnorm: 0.0,
        exitflag: 1,
        iterations: 0,
        func_count: 1,
        first_order_optimality: 0.0,
        step_size: 0.0,
        message: String::new(),
    };
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqcurvefit,
        None,
    );
    let value = block_on(finalize(
        result,
        &[1, 1],
        true,
        &[1, 1],
        &LeastSquaresBounds::unbounded(1),
        ALGORITHM,
        &mut cooperation,
    ))
    .unwrap();
    assert!(
        matches!(value, Value::OutputList(values) if matches!(values.as_slice(), [Value::Num(number)] if *number == 2.0))
    );
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn selected_model(
    function: usize,
    name: &str,
) -> (
    crate::context::RuntimeContext,
    Value,
    crate::user_functions::BoundFunctionInvokerGuard,
) {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, function);
    let entry = runtime.enter();
    let forwarder = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    drop(entry);
    (
        runtime,
        crate::user_functions::test_support::exact_local_value(name, binding),
        forwarder,
    )
}

#[test]
fn lsqcurvefit_real_array_reads_typed_integer_storage_exactly() {
    let input =
        Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![1, 3]).expect("integer");

    let parsed = block_on(real_array("xdata", Value::Tensor(input))).expect("real array");

    assert_eq!(parsed.values, vec![1.0, 2.0, 3.0]);
    assert_eq!(parsed.shape, vec![1, 3]);
}

#[test]
fn lsqcurvefit_bound_vector_reads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let input = Tensor::new_integer(IntegerStorage::I16(vec![-1, 2]), vec![1, 2]).expect("integer");

    let parsed = block_on(bound_vector(
        "lower bounds",
        Value::Tensor(input),
        2,
        f64::NEG_INFINITY,
    ))
    .expect("bounds");

    assert_eq!(parsed, vec![-1.0, 2.0]);
}

#[test]
fn lsqcurvefit_strict_mode_rejects_integer_initial_point() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let x0 = Tensor::new_integer(IntegerStorage::I32(vec![0]), vec![1, 1]).unwrap();

    let error = block_on(lsqcurvefit_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(x0),
        Value::Num(0.0),
        Value::Num(0.0),
        Vec::new(),
    ))
    .expect_err("integer x0 is a RunMat-only extension");

    assert_eq!(
        error.identifier(),
        LSQCURVEFIT_INTEGER_X0_EXTENSION.error_identifier
    );
}

#[test]
fn lsqcurvefit_runmat_mode_rejects_wide_integer_initial_point() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x0 = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();

    let error = block_on(lsqcurvefit_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(x0),
        Value::Num(0.0),
        Value::Num(0.0),
        Vec::new(),
    ))
    .expect_err("wide integer x0 cannot cross exactly");

    assert!(error.message().contains("exactly representable"));
}

#[test]
fn lsqcurvefit_passes_wide_integer_xdata_to_callback_exactly() {
    let (runtime, model, _bound) = selected_model(901, "wide_xdata");
    let _entry = runtime.enter();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let expected = u64::MAX;
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, _requested_outputs| {
            assert_eq!(function, 901);
            let Value::Tensor(xdata) = &args[1] else {
                panic!("expected exact integer xdata")
            };
            assert!(matches!(
                xdata.numeric_value_at(0),
                Some(runmat_value::NumericScalar::U64(value)) if value == expected
            ));
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(0.0))
            })
        },
    )))
    .expect("call-port scope");
    let xdata = Tensor::new_integer(IntegerStorage::U64(vec![expected]), vec![1, 1]).unwrap();

    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        Value::Num(0.0),
        Value::Tensor(xdata),
        Value::Num(0.0),
        Vec::new(),
    )))
    .expect("exact xdata pass-through");

    assert!(matches!(result, Value::Num(value) if value == 0.0));
}

#[test]
fn lsqcurvefit_automatic_resident_input_gathers_but_explicit_input_is_gated() {
    let (runtime, model, _bound) = selected_model(904, "zero_model");
    let _entry = runtime.enter();
    test_support::with_test_provider(|provider| {
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            |function, _args, _requested_outputs| {
                assert_eq!(function, 904);
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        0.0,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        let values = [0.0];
        let shape = [1, 1];
        let automatic = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("automatic upload");
        let automatic = automatic
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("admit provenance view");
        let result = block_on(runtime.scope(lsqcurvefit_builtin(
            model,
            Value::GpuTensor(automatic),
            Value::Num(0.0),
            Value::Num(0.0),
            Vec::new(),
        )))
        .expect("automatic resident initial point gathers");
        assert!(matches!(
            result,
            Value::Tensor(ref tensor)
                if tensor.shape() == [1, 1]
                    && matches!(tensor.numeric_value_at(0), Some(runmat_value::NumericScalar::F64(value)) if value == 0.0)
        ));

        let explicit = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("explicit upload");
        let explicit = explicit
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("admit provenance view");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(runtime.scope(lsqcurvefit_builtin(
            Value::FunctionHandle("unused".into()),
            Value::GpuTensor(explicit),
            Value::Num(0.0),
            Value::Num(0.0),
            Vec::new(),
        )))
        .expect_err("explicit resident input is gated before fallback");
        assert_eq!(
            error.identifier(),
            LSQCURVEFIT_RESIDENT_INPUT_EXTENSION.error_identifier
        );
    });
}

#[test]
fn lsqcurvefit_linear_model_recovers_parameters() {
    let (runtime, model, _bound) = selected_model(1, "line");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 1);
            assert_eq!(requested_outputs, 1);
            let p = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected params, got {other:?}"),
            };
            let xdata = match &args[1] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected xdata, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    xdata.iter().map(|x| p[0] * x + p[1]).collect::<Vec<_>>(),
                    vec![1, xdata.len()],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        tensor(vec![0.0, 0.0], vec![2, 1]),
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
        tensor(vec![1.0, 3.0, 5.0, 7.0], vec![1, 4]),
        Vec::new(),
    )))
    .expect("lsqcurvefit");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert!((t.materialize_f64()[0] - 2.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[1] - 1.0).abs() < 1.0e-5);
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqcurvefit_exponential_model_recovers_parameters() {
    let (runtime, model, _bound) = selected_model(2, "exp_decay");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 2);
            let p = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected params, got {other:?}"),
            };
            let xdata = match &args[1] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected xdata, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    xdata
                        .iter()
                        .map(|x| p[0] * (-p[1] * x).exp())
                        .collect::<Vec<_>>(),
                    vec![1, xdata.len()],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let xdata = (0..=10).map(|i| i as f64 * 0.2).collect::<Vec<_>>();
    let ydata = xdata
        .iter()
        .map(|x| 2.5 * (-0.7 * x).exp())
        .collect::<Vec<_>>();
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        tensor(vec![1.0, 0.1], vec![2, 1]),
        tensor(xdata.clone(), vec![1, xdata.len()]),
        tensor(ydata, vec![1, xdata.len()]),
        Vec::new(),
    )))
    .expect("lsqcurvefit");
    match result {
        Value::Tensor(t) => {
            assert!((t.materialize_f64()[0] - 2.5).abs() < 1.0e-4);
            assert!((t.materialize_f64()[1] - 0.7).abs() < 1.0e-4);
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqcurvefit_respects_bounds() {
    let (runtime, model, _bound) = selected_model(3, "bounded_line");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 3);
            let p = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected params, got {other:?}"),
            };
            let xdata = match &args[1] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected xdata, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    xdata.iter().map(|x| p[0] * x).collect::<Vec<_>>(),
                    vec![1, xdata.len()],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        tensor(vec![0.5], vec![1, 1]),
        tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
        tensor(vec![2.0, 4.0, 6.0], vec![1, 3]),
        vec![Value::Tensor(Tensor::zeros(vec![0, 0])), Value::Num(1.0)],
    )))
    .expect("lsqcurvefit");
    match result {
        Value::Tensor(t) => assert!((t.materialize_f64()[0] - 1.0).abs() < 1.0e-8),
        Value::Num(n) => assert!((n - 1.0).abs() < 1.0e-8),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqcurvefit_multi_output_includes_diagnostics_lambda_and_jacobian() {
    let (runtime, model, _bound) = selected_model(4, "line_multi");
    let _entry = runtime.enter();
    let _guard = crate::output_count::push_output_count(Some(7)).expect("output-count scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 4);
            let p = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected params, got {other:?}"),
            };
            let xdata = match &args[1] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected xdata, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    xdata.iter().map(|x| p[0] * x + p[1]).collect(),
                    vec![1, xdata.len()],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        tensor(vec![0.0, 0.0], vec![1, 2]),
        tensor(vec![0.0, 1.0, 2.0], vec![1, 3]),
        tensor(vec![1.0, 3.0, 5.0], vec![1, 3]),
        Vec::new(),
    )))
    .expect("lsqcurvefit");
    match result {
        Value::OutputList(outputs) => {
            assert_eq!(outputs.len(), 7);
            assert!(matches!(&outputs[1], Value::Num(resnorm) if *resnorm < 1.0e-10));
            assert!(matches!(&outputs[2], Value::Tensor(t) if t.shape() == [1, 3]));
            assert!(matches!(&outputs[3], Value::Num(flag) if *flag > 0.0));
            assert!(
                matches!(&outputs[4], Value::Struct(output) if output.fields.contains_key("funcCount"))
            );
            assert!(
                matches!(&outputs[5], Value::Struct(lambda) if lambda.fields.contains_key("lower") && lambda.fields.contains_key("upper"))
            );
            match &outputs[6] {
                Value::Tensor(j) => {
                    assert_eq!(j.shape(), &[3, 2]);
                    let expected = [0.0, 1.0, 2.0, 1.0, 1.0, 1.0];
                    for (actual, expected) in j.materialize_f64().iter().zip(expected) {
                        assert!((actual - expected).abs() < 1.0e-6);
                    }
                }
                other => panic!("expected jacobian tensor, got {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn optimoptions_long_evaluation_name_limits_lsqcurvefit_callbacks() {
    let (runtime, model, _bound) = selected_model(162, "budgeted_line");
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 162);
            assert_eq!(requested_outputs, 1);
            callback_calls.fetch_add(1, Ordering::SeqCst);
            let Value::Num(parameter) = &args[0] else {
                panic!("expected scalar parameter, got {:?}", args[0]);
            };
            let parameter = *parameter;
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![parameter, parameter * 2.0],
                    vec![1, 2],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let option_arguments = vec![
        Value::from("lsqcurvefit"),
        Value::from("MaxFunctionEvaluations"),
        Value::Num(1.0),
    ];
    let admitted =
        block_on(runtime.scope(crate::call_builtin_async("optimoptions", &option_arguments)))
            .expect("long option name is admitted");
    let options = admitted.value().clone();
    let Value::Struct(fields) = &options else {
        panic!("optimoptions must return a struct");
    };
    assert!(matches!(
        fields.fields.get("MaxFunEvals"),
        Some(Value::Num(1.0))
    ));
    assert!(!fields.fields.contains_key("MaxFunctionEvaluations"));

    let _outputs = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        Value::Num(0.0),
        tensor(vec![1.0, 2.0], vec![1, 2]),
        tensor(vec![2.0, 4.0], vec![1, 2]),
        vec![
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            options,
        ],
    )))
    .expect("budgeted lsqcurvefit completes with a diagnostic");
    let Value::OutputList(outputs) = result else {
        panic!("expected five lsqcurvefit outputs, got {result:?}");
    };
    assert!(matches!(outputs.get(3), Some(Value::Num(0.0))));
    let Some(Value::Struct(report)) = outputs.get(4) else {
        panic!("expected output diagnostics");
    };
    assert!(matches!(
        report.fields.get("funcCount"),
        Some(Value::Num(1.0))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn lsqcurvefit_output_reports_selected_algorithm() {
    let (runtime, model, _bound) = selected_model(45, "line_algorithm");
    let _entry = runtime.enter();
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 45);
            let p = match &args[0] {
                Value::Num(p) => *p,
                other => panic!("expected scalar param, got {other:?}"),
            };
            let xdata = match &args[1] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected xdata, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    xdata.iter().map(|x| p * x).collect::<Vec<_>>(),
                    vec![1, xdata.len()],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert("Algorithm", Value::from("trust-region-reflective"));
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        Value::Num(0.0),
        tensor(vec![1.0, 2.0], vec![1, 2]),
        tensor(vec![2.0, 4.0], vec![1, 2]),
        vec![
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Struct(options.finish()),
        ],
    )))
    .expect("lsqcurvefit");
    match result {
        Value::OutputList(outputs) => match &outputs[4] {
            Value::Struct(output) => match output.fields.get("algorithm") {
                Some(Value::String(algorithm)) => {
                    assert_eq!(algorithm, "trust-region-reflective")
                }
                other => panic!("expected algorithm string, got {other:?}"),
            },
            other => panic!("expected output struct, got {other:?}"),
        },
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqcurvefit_underdetermined_case_runs() {
    let (runtime, model, _bound) = selected_model(5, "underdetermined");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 5);
            let p = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected params, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    p[0] + p[1],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        tensor(vec![0.0, 0.0], vec![2, 1]),
        Value::Num(0.0),
        Value::Num(3.0),
        Vec::new(),
    )))
    .expect("lsqcurvefit");
    match result {
        Value::Tensor(t) => {
            assert!((t.materialize_f64()[0] + t.materialize_f64()[1] - 3.0).abs() < 1.0e-6)
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqcurvefit_rejects_model_ydata_shape_mismatch() {
    let (runtime, model, _bound) = selected_model(6, "bad_shape");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, _args, _requested_outputs| {
            assert_eq!(function, 6);
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![1.0, 2.0],
                    vec![1, 2],
                ))
            })
        },
    )))
    .expect("call-port scope");
    let err = block_on(runtime.scope(lsqcurvefit_builtin(
        model,
        Value::Num(0.0),
        Value::Num(0.0),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Vec::new(),
    )))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lsqcurvefit:InvalidInput"));
}

#[test]
fn lsqcurvefit_rejects_inconsistent_bounds() {
    let err = block_on(lsqcurvefit_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(0.0),
        Value::Num(0.0),
        Value::Num(0.0),
        vec![Value::Num(2.0), Value::Num(1.0)],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lsqcurvefit:InvalidArgument"));
}

#[test]
fn lsqcurvefit_rejects_more_than_seven_outputs() {
    let _guard = crate::output_count::push_output_count(Some(8)).expect("output-count scope");
    let err = block_on(lsqcurvefit_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(0.0),
        Value::Num(0.0),
        Value::Num(0.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lsqcurvefit:TooManyOutputs"));
}
