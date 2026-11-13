use super::*;
use futures::executor::block_on;
use runmat_value::Tensor;
use std::sync::{Arc, Mutex};

#[test]
fn one_output_does_not_publish_unused_jacobian() {
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
    let outcome = FsolveOutcome {
        result,
        residual_shape: vec![1, 1],
        residual_scalar: true,
    };
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Fsolve,
        None,
    );
    let value = block_on(finalize(outcome, &[1, 1], true, &mut cooperation)).unwrap();
    assert!(
        matches!(value, Value::OutputList(values) if matches!(values.as_slice(), [Value::Num(number)] if *number == 2.0))
    );
}

#[test]
fn residual_shape_cannot_change_at_constant_element_count() {
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Fsolve,
        None,
    );
    let function = Value::Num(0.0);
    let mut evaluator = FsolveEvaluator {
        cooperation: &mut cooperation,
        function: &function,
        shape: vec![2, 1],
        scalar: false,
        residual_shape: None,
        residual_scalar: false,
        specify_objective_gradient: false,
        jacobian: None,
        accepted_jacobian: None,
    };
    evaluator.accept_residual_shape(vec![1, 2], false).unwrap();
    assert!(evaluator.accept_residual_shape(vec![2, 1], false).is_err());
    assert_eq!(evaluator.residual_shape, Some(vec![1, 2]));
}

#[test]
fn fsolve_scalar_builtin_handle() {
    let root = block_on(fsolve_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(3.0),
        Vec::new(),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-5),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fsolve_five_output_form_returns_residual_status_diagnostics_and_jacobian() {
    let _outputs = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let result = block_on(fsolve_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(3.0),
        Vec::new(),
    ))
    .unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected five outputs");
    };
    assert_eq!(outputs.len(), 5);
    assert!(matches!(outputs[0], Value::Num(_)));
    assert!(matches!(outputs[1], Value::Num(_)));
    assert!(matches!(outputs[2], Value::Num(_)));
    assert!(matches!(outputs[3], Value::Struct(_)));
    assert!(matches!(&outputs[4], Value::Tensor(tensor) if tensor.shape() == [1, 1]));
}

#[test]
fn fsolve_returns_stationary_non_root_with_solver_status() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 44);
    let entry = runtime.enter();
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 44);
            let x = match &args[0] {
                Value::Num(value) => *value,
                other => panic!("expected scalar numeric argument, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    x * x + 1.0,
                ))
            })
        },
    )))
    .expect("call-port scope");
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    drop(entry);
    let result = block_on(runtime.scope(fsolve_builtin(
        crate::user_functions::test_support::exact_local_value("no_real_root", binding),
        Value::Num(0.0),
        Vec::new(),
    )))
    .unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected outputs")
    };
    assert!(matches!(outputs[2], Value::Num(flag) if flag <= 0.0));
}

#[test]
fn fsolve_vector_system_via_semantic_resolver() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("system", 41)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
        std::sync::Arc::new(|function, args, _requested_outputs| {
            assert_eq!(function, 41);
            let x = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                _ => panic!("expected tensor input"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    Tensor::new(
                        vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] * x[1] - 1.0],
                        vec![2, 1],
                    )
                    .unwrap(),
                ))
            })
        }),
    ))
    .expect("call-port scope");
    drop(entry);
    // On the symmetric diagonal both Jacobian rows are parallel, and a
    // stationary least-squares point need not be a root of this system.
    // The separate stationary-non-root case covers that terminal behavior.
    let x0 = Tensor::new(vec![1.5, 0.5], vec![2, 1]).unwrap();
    let root = block_on(runtime.scope(fsolve_builtin(
        Value::FunctionHandle("system".into()),
        Value::Tensor(x0),
        Vec::new(),
    )))
    .unwrap();
    match root {
        Value::Tensor(t) => {
            assert!(
                (t.materialize_f64()[0] * t.materialize_f64()[0]
                    + t.materialize_f64()[1] * t.materialize_f64()[1]
                    - 4.0)
                    .abs()
                    < 1.0e-5
            );
            assert!((t.materialize_f64()[0] * t.materialize_f64()[1] - 1.0).abs() < 1.0e-5);
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fsolve_preserves_row_vector_shape_for_callback() {
    let seen_shapes = Arc::new(Mutex::new(Vec::new()));
    let seen_shapes_for_invoker = Arc::clone(&seen_shapes);
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("row_system", 42)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, _requested_outputs| {
            assert_eq!(function, 42);
            let (x, shape) = match &args[0] {
                Value::Tensor(t) => (t.materialize_f64().clone(), t.shape().to_vec()),
                other => panic!("expected tensor input, got {other:?}"),
            };
            assert_eq!(shape, vec![1, 2]);
            seen_shapes_for_invoker.lock().unwrap().push(shape.clone());
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    Tensor::new(vec![x[0] - 3.0, x[1] - 4.0], shape).unwrap(),
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let x0 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
    let root = block_on(runtime.scope(fsolve_builtin(
        Value::FunctionHandle("row_system".into()),
        Value::Tensor(x0),
        Vec::new(),
    )))
    .unwrap();
    match root {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[1, 2]);
            assert!((t.materialize_f64()[0] - 3.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[1] - 4.0).abs() < 1.0e-5);
        }
        other => panic!("unexpected value {other:?}"),
    }
    assert!(!seen_shapes.lock().unwrap().is_empty());
}

#[test]
fn fsolve_preserves_matrix_shape_for_callback() {
    let seen_shapes = Arc::new(Mutex::new(Vec::new()));
    let seen_shapes_for_invoker = Arc::clone(&seen_shapes);
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("matrix_system", 45)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, _requested_outputs| {
            assert_eq!(function, 45);
            let (x, shape) = match &args[0] {
                Value::Tensor(t) => (t.materialize_f64().clone(), t.shape().to_vec()),
                other => panic!("expected tensor input, got {other:?}"),
            };
            assert_eq!(shape, vec![2, 2]);
            seen_shapes_for_invoker.lock().unwrap().push(shape.clone());
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    Tensor::new(vec![x[0] - 1.0, x[1] - 2.0, x[2] - 3.0, x[3] - 4.0], shape)
                        .unwrap(),
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let x0 = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
    let root = block_on(runtime.scope(fsolve_builtin(
        Value::FunctionHandle("matrix_system".into()),
        Value::Tensor(x0),
        Vec::new(),
    )))
    .unwrap();
    match root {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert!((t.materialize_f64()[0] - 1.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[1] - 2.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[2] - 3.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[3] - 4.0).abs() < 1.0e-5);
        }
        other => panic!("unexpected value {other:?}"),
    }
    assert!(!seen_shapes.lock().unwrap().is_empty());
}

#[test]
fn fsolve_accepts_semantic_function_handle_callback() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 43);
    let entry = runtime.enter();
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 43);
            assert_eq!(requested_outputs, 1);
            let x = match &args[0] {
                Value::Num(value) => *value,
                other => panic!("expected scalar numeric argument, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    x - 3.0,
                ))
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let root = block_on(runtime.scope(fsolve_builtin(
        crate::user_functions::test_support::exact_local_value("system_function", binding),
        Value::Num(1.0),
        Vec::new(),
    )))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - 3.0).abs() < 1.0e-5),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fsolve_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FSOLVE_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "x = fsolve(fun, x0)",
            "x = fsolve(fun, x0, options)",
            "[x,fval] = fsolve(fun, x0)",
            "[x,fval] = fsolve(fun, x0, options)",
            "[x,fval,exitflag] = fsolve(fun, x0)",
            "[x,fval,exitflag] = fsolve(fun, x0, options)",
            "[x,fval,exitflag,output] = fsolve(fun, x0)",
            "[x,fval,exitflag,output] = fsolve(fun, x0, options)",
            "[x,fval,exitflag,output,jacobian] = fsolve(fun, x0)",
            "[x,fval,exitflag,output,jacobian] = fsolve(fun, x0, options)",
        ]
    );

    let codes: Vec<&str> = FSOLVE_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert_eq!(
        codes,
        vec!["RM.FSOLVE.INVALID_ARGUMENT", "RM.FSOLVE.INVALID_INPUT"]
    );
}

#[test]
fn fsolve_too_many_args_uses_stable_identifier() {
    let err = block_on(fsolve_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(1.0),
        vec![
            Value::Struct(StructValue::new()),
            Value::Struct(StructValue::new()),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:fsolve:InvalidArgument"));
}
