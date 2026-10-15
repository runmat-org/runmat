use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;
use std::sync::Arc;

#[test]
fn one_output_does_not_publish_unused_gradient_or_jacobian() {
    let _outputs = crate::output_count::push_output_count(Some(1));
    let result = LeastSquaresResult {
        x: vec![2.0],
        residual: vec![0.0],
        jacobian: vec![],
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
        runmat_builtins::OptimSolver::Lsqnonlin,
        None,
    );
    let value = block_on(finalize(
        result,
        &[1, 1],
        true,
        &[1, 1],
        &LeastSquaresBounds::unbounded(1),
        DEFAULT_ALGORITHM,
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

fn selected_residual(
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
fn lsqnonlin_real_array_reads_typed_integer_storage_exactly() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).expect("integer");

    let parsed = block_on(real_array("x0", Value::Tensor(input))).expect("real array");

    assert_eq!(parsed.values, vec![1.0, 2.0]);
    assert_eq!(parsed.shape, vec![2, 1]);
}

#[test]
fn lsqnonlin_bound_vector_reads_typed_integer_storage_exactly() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![-1, 2]), vec![2, 1]).expect("integer");

    let parsed = block_on(bound_vector(
        "lower bounds",
        Value::Tensor(input),
        3,
        f64::NEG_INFINITY,
    ))
    .expect("bounds");

    assert_eq!(parsed, vec![-1.0, 2.0, f64::NEG_INFINITY]);
}

#[test]
fn lsqnonlin_vector_residual_recovers_parameters() {
    let (runtime, objective, _bound) = selected_residual(1, "residual");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 1);
            assert_eq!(requested_outputs, 1);
            let x = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected x tensor, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![x[0] - 2.0, x[1] - 3.0, x[0] + x[1] - 5.0],
                    vec![3, 1],
                ))
            })
        },
    )));
    let result = block_on(runtime.scope(lsqnonlin_builtin(
        objective,
        vec![tensor(vec![0.0, 0.0], vec![2, 1])],
    )))
    .expect("lsqnonlin");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert!((t.materialize_f64()[0] - 2.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[1] - 3.0).abs() < 1.0e-5);
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqnonlin_respects_bounds_and_empty_constraint_placeholders() {
    let (runtime, objective, _bound) = selected_residual(2, "bounded");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 2);
            let x = match &args[0] {
                Value::Num(value) => *value,
                other => panic!("expected scalar x, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![x - 2.0, x - 2.0],
                    vec![1, 2],
                ))
            })
        },
    )));
    let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
    let result = block_on(runtime.scope(lsqnonlin_builtin(
        objective,
        vec![
            Value::Num(0.0),
            Value::Num(-1.0),
            Value::Num(1.0),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty,
        ],
    )))
    .expect("lsqnonlin");
    match result {
        Value::Num(value) => assert!((value - 1.0).abs() < 1.0e-8),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqnonlin_accepts_empty_options_placeholder() {
    let (runtime, objective, _bound) = selected_residual(22, "empty_options");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 22);
            let x = match &args[0] {
                Value::Num(value) => *value,
                other => panic!("expected scalar x, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    x - 3.0,
                ))
            })
        },
    )));
    let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
    let result = block_on(runtime.scope(lsqnonlin_builtin(
        objective,
        vec![Value::Num(0.0), empty.clone(), empty.clone(), empty],
    )))
    .expect("lsqnonlin accepts [] options");
    match result {
        Value::Num(value) => assert!((value - 3.0).abs() < 1.0e-5),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqnonlin_partial_bounds_apply_to_leading_variables() {
    let (runtime, objective, _bound) = selected_residual(23, "partial_bounds");
    let _entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 23);
            let x = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected x tensor, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![x[0] - 2.0, x[1] - 5.0],
                    vec![2, 1],
                ))
            })
        },
    )));
    let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
    let result = block_on(runtime.scope(lsqnonlin_builtin(
        objective,
        vec![tensor(vec![0.0, 0.0], vec![2, 1]), empty, Value::Num(1.0)],
    )))
    .expect("lsqnonlin partial bounds");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert!((t.materialize_f64()[0] - 1.0).abs() < 1.0e-5);
            assert!((t.materialize_f64()[1] - 5.0).abs() < 1.0e-5);
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqnonlin_multi_output_includes_diagnostics_lambda_and_jacobian() {
    let (runtime, objective, _bound) = selected_residual(3, "multi");
    let _entry = runtime.enter();
    let _guard = crate::output_count::push_output_count(Some(7));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 3);
            let x = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected x tensor, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![x[0] - 1.0, 2.0 * x[1] - 4.0],
                    vec![1, 2],
                ))
            })
        },
    )));
    let mut options = StructValue::new();
    options.insert("Algorithm", Value::from("levenberg-marquardt"));
    let result = block_on(runtime.scope(lsqnonlin_builtin(
        objective,
        vec![
            tensor(vec![0.0, 0.0], vec![1, 2]),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Struct(options),
        ],
    )))
    .expect("lsqnonlin");
    match result {
        Value::OutputList(outputs) => {
            assert_eq!(outputs.len(), 7);
            assert!(matches!(&outputs[1], Value::Num(resnorm) if *resnorm < 1.0e-10));
            assert!(matches!(&outputs[2], Value::Tensor(t) if t.shape() == [1, 2]));
            assert!(matches!(&outputs[3], Value::Num(flag) if *flag > 0.0));
            assert!(
                matches!(&outputs[4], Value::Struct(output) if output.fields.contains_key("funcCount") && output.fields.contains_key("stepsize") && output.fields.contains_key("cgiterations") && output.fields.contains_key("lssteplength"))
            );
            assert!(
                matches!(&outputs[5], Value::Struct(lambda) if lambda.fields.contains_key("lower") && lambda.fields.contains_key("upper"))
            );
            match &outputs[6] {
                Value::Tensor(j) => {
                    assert_eq!(j.shape(), &[2, 2]);
                    let expected = [1.0, 0.0, 0.0, 2.0];
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
fn lsqnonlin_accepts_problem_struct() {
    let (runtime, objective, _bound) = selected_residual(4, "problem");
    let _entry = runtime.enter();
    let _guard = crate::output_count::push_output_count(Some(2));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, _requested_outputs| {
            assert_eq!(function, 4);
            let x = match &args[0] {
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected x tensor, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(tensor(
                    vec![x[0] - 4.0, x[1] - 5.0],
                    vec![2, 1],
                ))
            })
        },
    )));
    let mut problem = StructValue::new();
    problem.insert("solver", Value::from("lsqnonlin"));
    problem.insert("objective", objective);
    problem.insert("x0", tensor(vec![0.0, 0.0], vec![2, 1]));
    problem.insert("lb", tensor(vec![1.0, 2.0], vec![2, 1]));
    problem.insert("ub", tensor(vec![10.0, 10.0], vec![2, 1]));
    let result = block_on(runtime.scope(lsqnonlin_builtin(Value::Struct(problem), Vec::new())))
        .expect("lsqnonlin problem");
    match result {
        Value::OutputList(outputs) => {
            assert_eq!(outputs.len(), 2);
            assert!(
                matches!(&outputs[0], Value::Tensor(t) if (t.materialize_f64()[0] - 4.0).abs() < 1.0e-5 && (t.materialize_f64()[1] - 5.0).abs() < 1.0e-5)
            );
            assert!(matches!(&outputs[1], Value::Num(resnorm) if *resnorm < 1.0e-10));
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn lsqnonlin_problem_struct_rejects_aineq_and_bineq() {
    let mut problem = StructValue::new();
    problem.insert("solver", Value::from("lsqnonlin"));
    problem.insert("objective", Value::FunctionHandle("sin".into()));
    problem.insert("x0", Value::Num(0.0));
    problem.insert("Aineq", Value::Num(1.0));
    problem.insert("bineq", Value::Num(0.0));
    let err = block_on(lsqnonlin_builtin(Value::Struct(problem), Vec::new())).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lsqnonlin:InvalidArgument"));
    assert!(err.message().contains("constraints"));
}

#[test]
fn lsqnonlin_default_max_fun_evals_matches_optimoptions() {
    let options = LsqOptions::from_struct(None, 7).expect("default options");
    assert_eq!(
        options.solver.max_fun_evals,
        DEFAULT_MAX_FUN_EVALS_FACTOR * DEFAULT_MAX_ITER
    );
}

#[test]
fn lsqnonlin_rejects_real_constraints_until_interior_point_exists() {
    let err = block_on(lsqnonlin_builtin(
        Value::FunctionHandle("sin".into()),
        vec![
            Value::Num(0.0),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Num(1.0),
            Value::Num(0.0),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Value::Tensor(Tensor::zeros(vec![0, 0])),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lsqnonlin:InvalidArgument"));
}

#[test]
fn lsqnonlin_rejects_more_than_seven_outputs() {
    let _guard = crate::output_count::push_output_count(Some(8));
    let err = block_on(lsqnonlin_builtin(
        Value::FunctionHandle("sin".into()),
        vec![Value::Num(0.0)],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lsqnonlin:TooManyOutputs"));
}
