use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage};
use std::sync::Arc;

fn bound(id: usize) -> Value {
    crate::user_functions::test_support::exact_local_value(
        &format!("objective_{id}"),
        crate::user_functions::test_support::exact_portable_binding(id, "fminunc fixture"),
    )
}

fn bound_forwarder(id: usize) -> crate::user_functions::BoundFunctionInvokerGuard {
    crate::user_functions::test_support::install_bound_forwarder(
        crate::user_functions::test_support::exact_portable_binding(id, "fminunc fixture"),
    )
}

fn selected_bound(
    id: usize,
) -> (
    crate::context::RuntimeContext,
    Value,
    crate::user_functions::BoundFunctionInvokerGuard,
) {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, id);
    let entry = runtime.enter();
    let forwarder = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    drop(entry);
    (
        runtime,
        crate::user_functions::test_support::exact_local_value(&format!("objective_{id}"), binding),
        forwarder,
    )
}

fn tensor(data: Vec<f64>) -> Value {
    Value::Tensor(Tensor::new(data, vec![3, 1]).unwrap())
}

fn vector_from_value(value: &Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64().clone(),
        Value::Num(n) => vec![*n],
        other => panic!("expected numeric value, got {other:?}"),
    }
}

fn assert_close_vec(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() <= tol, "at {idx}: expected {e}, got {a}");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_quadratic_reaches_vector_minimum() {
    let (runtime, objective, _bound) = selected_bound(101);
    let _entry = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 101);
            assert_eq!(requested_outputs, 1);
            let x = vector_from_value(&args[0]);
            Box::pin(async move {
                let target = [1.0, 2.0, 3.0];
                let f = x
                    .iter()
                    .zip(target.iter())
                    .map(|(xi, ti)| (xi - ti).powi(2))
                    .sum::<f64>();
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(f))
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(runtime.scope(fminunc_builtin(
        objective,
        tensor(vec![0.0, 0.0, 0.0]),
        Vec::new(),
    )))
    .expect("fminunc");
    assert_close_vec(&vector_from_value(&result), &[1.0, 2.0, 3.0], 1.0e-4);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_rosenbrock_converges_from_standard_start() {
    let (runtime, objective, _bound) = selected_bound(102);
    let _entry = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 102);
            assert_eq!(requested_outputs, 1);
            let x = vector_from_value(&args[0]);
            Box::pin(async move {
                let a = x[0];
                let b = x[1];
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2),
                ))
            })
        },
    )))
    .expect("call-port scope");
    let x0 = Value::Tensor(Tensor::new(vec![-1.2, 1.0], vec![2, 1]).unwrap());
    let result =
        block_on(runtime.scope(fminunc_builtin(objective, x0, Vec::new()))).expect("fminunc");
    assert_close_vec(&vector_from_value(&result), &[1.0, 1.0], 2.0e-3);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_handles_high_dimensional_smooth_objective() {
    let (runtime, objective, _bound) = selected_bound(103);
    let _entry = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 103);
            assert_eq!(requested_outputs, 1);
            let x = vector_from_value(&args[0]);
            Box::pin(async move {
                let f = x
                    .iter()
                    .enumerate()
                    .map(|(idx, xi)| {
                        let target = (idx + 1) as f64;
                        (xi - target).powi(2)
                    })
                    .sum::<f64>();
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(f))
            })
        },
    )))
    .expect("call-port scope");
    let x0 = Value::Tensor(Tensor::new(vec![0.0; 8], vec![8, 1]).unwrap());
    let result =
        block_on(runtime.scope(fminunc_builtin(objective, x0, Vec::new()))).expect("fminunc");
    assert_close_vec(
        &vector_from_value(&result),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        1.0e-4,
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_uses_objective_gradient_and_returns_six_outputs() {
    let (runtime, objective, _bound) = selected_bound(104);
    let _entry = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 104);
            assert_eq!(requested_outputs, 2);
            let x = vector_from_value(&args[0]);
            Box::pin(async move {
                let target = [1.0, 2.0, 3.0];
                let grad = x
                    .iter()
                    .zip(target.iter())
                    .map(|(xi, ti)| 2.0 * (xi - ti))
                    .collect::<Vec<_>>();
                let f = x
                    .iter()
                    .zip(target.iter())
                    .map(|(xi, ti)| (xi - ti).powi(2))
                    .sum::<f64>();
                let sequence = crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Num(f),
                    Value::Tensor(Tensor::new(grad, vec![3, 1]).unwrap()),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)?;
                crate::execution::RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())
            })
        },
    )))
    .expect("call-port scope");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("SpecifyObjectiveGradient", Value::Bool(true));
    opts.insert("TolX", Value::Num(1.0e-10));
    opts.insert("TolFun", Value::Num(1.0e-10));
    let _outputs = crate::output_count::push_output_count(Some(6)).expect("output-count scope");
    let result = block_on(runtime.scope(fminunc_builtin(
        objective,
        tensor(vec![0.0, 0.0, 0.0]),
        vec![Value::Struct(opts.finish())],
    )))
    .expect("fminunc");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 6);
    assert_close_vec(&vector_from_value(&outputs[0]), &[1.0, 2.0, 3.0], 1.0e-7);
    assert!(matches!(outputs[1], Value::Num(f) if f.abs() < 1.0e-14));
    assert!(matches!(outputs[2], Value::Num(flag) if flag > 0.0));
    match &outputs[3] {
        Value::Struct(output) => {
            assert!(matches!(
                output.fields.get("iterations"),
                Some(Value::Num(_))
            ));
            assert!(matches!(
                output.fields.get("funcCount"),
                Some(Value::Num(_))
            ));
            assert!(
                matches!(output.fields.get("firstorderopt"), Some(Value::Num(v)) if *v < 1.0e-8)
            );
            assert!(
                matches!(output.fields.get("algorithm"), Some(Value::String(text)) if text.contains("bfgs"))
            );
        }
        other => panic!("unexpected output struct {other:?}"),
    }
    assert_close_vec(&vector_from_value(&outputs[4]), &[0.0, 0.0, 0.0], 1.0e-7);
    match &outputs[5] {
        Value::Tensor(hessian) => {
            assert_eq!(hessian.shape(), vec![3, 3]);
            assert!(hessian
                .materialize_f64()
                .iter()
                .all(|value| value.is_finite()));
            assert!(hessian.materialize_f64()[0] > 0.0);
            assert!(hessian.materialize_f64()[4] > 0.0);
            assert!(hessian.materialize_f64()[8] > 0.0);
        }
        other => panic!("unexpected hessian {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_rejects_invalid_gradient_length() {
    let (runtime, objective, _bound) = selected_bound(105);
    let _entry = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, _args, requested_outputs| {
            assert_eq!(function, 105);
            assert_eq!(requested_outputs, 2);
            Box::pin(async {
                let sequence = crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Num(1.0),
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)?;
                crate::execution::RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())
            })
        },
    )))
    .expect("call-port scope");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("SpecifyObjectiveGradient", Value::Bool(true));
    let err = block_on(runtime.scope(fminunc_builtin(
        objective,
        tensor(vec![0.0, 0.0, 0.0]),
        vec![Value::Struct(opts.finish())],
    )))
    .unwrap_err();
    assert!(err.message().contains("gradient length"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_scalar_input_returns_scalar() {
    let (runtime, objective, _bound) = selected_bound(106);
    let _entry = runtime.enter();
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 106);
            assert_eq!(requested_outputs, 1);
            let x = match &args[0] {
                Value::Num(value) => *value,
                other => panic!("expected scalar, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    (x - 4.0).powi(2),
                ))
            })
        },
    )))
    .expect("call-port scope");
    let result = block_on(runtime.scope(fminunc_builtin(objective, Value::Num(0.0), Vec::new())))
        .expect("fminunc");
    assert!(matches!(result, Value::Num(x) if (x - 4.0).abs() < 1.0e-4));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_rejects_non_struct_options() {
    let err = block_on(fminunc_builtin(
        bound(107),
        Value::Num(0.0),
        vec![Value::Int(IntValue::I32(1))],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:fminunc:InvalidArgument"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_rejects_budget_too_small_for_finite_difference_gradient() {
    let _bound = bound_forwarder(108);
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, _args, _requested_outputs| {
            Box::pin(async { panic!("budget check should happen before invoking objective") })
        },
    )))
    .expect("call-port scope");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxFunEvals", Value::Num(3.0));
    let err = block_on(fminunc_builtin(
        bound(108),
        tensor(vec![0.0, 0.0, 0.0]),
        vec![Value::Struct(opts.finish())],
    ))
    .unwrap_err();
    assert!(err.message().contains("MaxFunEvals"));
}

#[test]
fn options_read_typed_integer_tensor_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let max_iter = Tensor::new_integer(IntegerStorage::U16(vec![5]), vec![1, 1]).expect("MaxIter");
    let gradient = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1])
        .expect("SpecifyObjectiveGradient");

    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Tensor(max_iter));
    opts.insert("SpecifyObjectiveGradient", Value::Tensor(gradient));
    let opts = opts.finish();
    let parsed = FminuncOptions::from_struct(Some(&opts), 1).expect("options");
    assert_eq!(parsed.max_iter, 5);
    assert!(parsed.specify_objective_gradient);
}

#[test]
fn documented_logical_option_flag_is_not_extension_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("SpecifyObjectiveGradient", Value::Bool(true));
    let opts = opts.finish();
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Fminunc,
        None,
    );
    let admitted = block_on(
        crate::builtins::math::optim::common::prepare_solver_options(
            runmat_builtins::OptimSolver::Fminunc,
            Some(opts.clone()),
            Some(&FMINUNC_OPTION_EXTENSION),
            Some(&FMINUNC_RESIDENT_EXTENSION),
            None,
            &mut cooperation,
        ),
    )
    .expect("documented logical option is admitted without extensions")
    .expect("present options");
    let parsed =
        FminuncOptions::from_struct(Some(admitted.structure()), 1).expect("logical option");
    assert!(parsed.specify_objective_gradient);

    let mut opts = runmat_value::UnboundedStructValueBuilder::from_structure(opts);
    opts.insert(
        "SpecifyObjectiveGradient",
        Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap()),
    );
    let opts = opts.finish();
    // from_struct consumes already-admitted options. Exercise the actual
    // preparation/compatibility owner rather than bypassing its gate.
    let error = block_on(fminunc_builtin(
        bound(109),
        Value::Num(0.0),
        vec![Value::Struct(opts)],
    ))
    .expect_err("typed-integer spelling is independently gated before the objective");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FminuncOptionExtension")
    );
}

#[test]
fn options_reject_typed_integer_limits_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let negative = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("MaxIter");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Tensor(negative));
    let opts = opts.finish();
    assert!(FminuncOptions::from_struct(Some(&opts), 1).is_err());

    let too_large = Tensor::new_integer(
        IntegerStorage::U64(vec![(MAX_FUN_EVAL_LIMIT as u64) + 1]),
        vec![1, 1],
    )
    .expect("MaxFunEvals");
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxFunEvals", Value::Tensor(too_large));
    let opts = opts.finish();
    assert!(FminuncOptions::from_struct(Some(&opts), 1).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_rejects_problem_too_large_for_dense_bfgs() {
    let variables = MAX_DENSE_BFGS_VARIABLES + 1;
    let x0 = Value::Tensor(Tensor::new(vec![0.0; variables], vec![variables, 1]).unwrap());
    let err = block_on(fminunc_builtin(bound(109), x0, Vec::new())).unwrap_err();
    assert!(err.message().contains("dense BFGS"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fminunc_rejects_huge_direct_iteration_limits() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxFunEvals", Value::Num((MAX_FUN_EVAL_LIMIT as f64) + 1.0));
    let err = block_on(fminunc_builtin(
        bound(110),
        Value::Num(0.0),
        vec![Value::Struct(opts.finish())],
    ))
    .unwrap_err();
    assert!(err.message().contains("MaxFunEvals"));
}
