use super::*;

#[test]
fn finite_difference_zero_iteration_and_hard_budget_outputs_never_fabricate_jacobians() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("finite_difference_system", 989)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(id, 989);
            assert_eq!(requested, 1);
            observed.fetch_add(1, Ordering::SeqCst);
            let Value::Tensor(ref x) = args[0] else {
                panic!("state")
            };
            let x = x.materialize_f64();
            let residual = vec![x[0] * x[0] + x[1] - 3.0, 3.0 * x[1] - 3.0];
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    runmat_value::Tensor::new(residual, vec![2, 1]).unwrap(),
                ))
            })
        },
    )))
    .unwrap();
    // One fixture spans omitted J, zero-step J, unavailable/partial J and
    // accepted-state renewal, with exact original callback counts.
    for (requested, iterations, budget, expected_calls) in [
        (1, 0, 10, 1),
        (5, 0, 10, 3),
        (5, 0, 1, 1),
        (5, 1, 2, 2),
        (5, 1, 4, 4),
        (5, 1, 10, 6),
    ] {
        calls.store(0, Ordering::SeqCst);
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("MaxIterations", Value::Num(iterations as f64));
        options.insert("MaxFunEvals", Value::Num(budget as f64));
        let _outputs = crate::output_count::push_output_count(Some(requested)).unwrap();
        let result = block_on(runtime.scope(fsolve_builtin(
            Value::FunctionHandle("finite_difference_system".into()),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
            vec![Value::Struct(options.finish())],
        )))
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
        assert!(expected_calls <= budget);
        let Value::OutputList(outputs) = result else {
            panic!("outputs")
        };
        let Value::Tensor(ref x) = outputs[0] else {
            panic!("x")
        };
        let x = x.materialize_f64();
        if iterations == 0 || budget == 2 {
            assert_eq!(x.as_slice(), &[1.0, 0.0]);
        }
        if requested == 1 {
            continue;
        }
        let Value::Tensor(ref jacobian) = outputs[4] else {
            panic!("J")
        };
        assert_eq!(jacobian.shape(), [2, 2]);
        let jacobian = jacobian.materialize_f64();
        let Value::Struct(ref info) = outputs[3] else {
            panic!("info")
        };
        if budget == 1 || budget == 4 {
            assert!(jacobian.iter().all(|value| value.is_nan()));
            assert!(matches!(info.fields["firstorderopt"],Value::Num(value) if value.is_nan()));
        } else if budget == 2 {
            assert!((jacobian[0] - 2.0).abs() < 1e-6);
            assert_eq!(jacobian[1], 0.0);
            assert!(jacobian[2].is_nan() && jacobian[3].is_nan());
            assert!(matches!(info.fields["firstorderopt"],Value::Num(value) if value.is_nan()));
        } else {
            let expected = [2.0 * x[0], 0.0, 1.0, 3.0];
            assert!(jacobian
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 1e-6));
        }
    }
}

#[test]
fn nonlinear_terminal_outputs_keep_the_jacobian_paired_to_returned_x_without_extra_calls() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("nonlinear_pair", 987), ("nonlinear_sparse_pair", 988)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(requested, 2);
            observed.fetch_add(1, Ordering::SeqCst);
            let Value::Tensor(ref x) = args[0] else {
                panic!("state")
            };
            let x = x.materialize_f64();
            let residual = vec![x[0] * x[0] + x[1] - 3.0, 3.0 * x[1] - 3.0];
            let jacobian = if id == 988 {
                Value::SparseTensor(
                    runmat_value::SparseTensor::new(
                        2,
                        2,
                        vec![0, 1, 3],
                        vec![0, 0, 1],
                        vec![2.0 * x[0], 1.0, 3.0],
                    )
                    .unwrap(),
                )
            } else {
                assert_eq!(id, 987);
                Value::Tensor(
                    runmat_value::Tensor::new(vec![2.0 * x[0], 0.0, 1.0, 3.0], vec![2, 2]).unwrap(),
                )
            };
            Box::pin(async move {
                let sequence = crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Tensor(runmat_value::Tensor::new(residual, vec![2, 1]).unwrap()),
                    jacobian,
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)?;
                crate::execution::RootedValueSequence::new(sequence)
                    .map_err(|source| source.into_runtime_error())
            })
        },
    )))
    .unwrap();
    // Zero iterations, accepted final iteration, evaluation exhaustion, sparse
    // paired Jacobian. This is one nonlinear boundary fixture, not an accuracy slew.
    for (iterations, evaluations, sparse, initial) in [
        (0, 10, false, [1.0, 0.0]),
        (1, 10, false, [1.0, 0.0]),
        (5, 2, false, [1.0, 0.0]),
        (1, 2, true, [1.0, 0.0]),
        (5, 2, false, [0.1, 1.0]),
    ] {
        calls.store(0, Ordering::SeqCst);
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("SpecifyObjectiveGradient", Value::Bool(true));
        options.insert("MaxIterations", Value::Num(iterations as f64));
        options.insert("MaxFunEvals", Value::Num(evaluations as f64));
        let _outputs = crate::output_count::push_output_count(Some(5)).unwrap();
        let function = if sparse {
            "nonlinear_sparse_pair"
        } else {
            "nonlinear_pair"
        };
        let output = block_on(runtime.scope(fsolve_builtin(
            Value::FunctionHandle(function.into()),
            Value::Tensor(runmat_value::Tensor::new(initial.to_vec(), vec![2, 1]).unwrap()),
            vec![Value::Struct(options.finish())],
        )))
        .unwrap();
        let Value::OutputList(outputs) = output else {
            panic!("outputs")
        };
        let Value::Tensor(ref x) = outputs[0] else {
            panic!("x")
        };
        let x = x.materialize_f64();
        if initial[0] == 0.1 {
            assert_eq!(x.as_slice(), &initial);
        }
        let Value::Tensor(ref jacobian) = outputs[4] else {
            panic!("J")
        };
        let jacobian = jacobian.materialize_f64();
        assert_eq!(jacobian.as_slice(), &[2.0 * x[0], 0.0, 1.0, 3.0]);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            if iterations == 0 { 1 } else { 2 }
        );
        assert_eq!(outputs[2], Value::Num(0.0));
        let Value::Struct(ref info) = outputs[3] else {
            panic!("info")
        };
        assert!(
            matches!(info.fields["funcCount"],Value::Num(count) if count as usize == calls.load(Ordering::SeqCst))
        );
    }
}
use futures::executor::block_on;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn supplied_nonsymmetric_jacobian_uses_two_output_callbacks_and_preserves_column_major_result() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("analytic_system", 983)],
    )
    .unwrap();
    let _entry = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |id, args, requested| {
            assert_eq!(id, 983);
            assert_eq!(requested, 2);
            observed.fetch_add(1, Ordering::SeqCst);
            let Value::Tensor(ref x) = args[0] else {
                panic!("vector state")
            };
            let x = x.materialize_f64();
            let residual = vec![x[0] + 2.0 * x[1] - 5.0, 3.0 * x[0] + 4.0 * x[1] - 11.0];
            Box::pin(async move {
                let sequence = crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Tensor(runmat_value::Tensor::new(residual, vec![2, 1]).unwrap()),
                    Value::Tensor(
                        runmat_value::Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap(),
                    ),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)?;
                crate::execution::RootedValueSequence::new(sequence)
                    .map_err(|source| source.into_runtime_error())
            })
        },
    )))
    .unwrap();
    let mut options = runmat_value::UnboundedStructValueBuilder::new();
    options.insert("SpecifyObjectiveGradient", Value::Bool(true));
    options.insert("MaxIterations", Value::Num(40.0));
    let _outputs = crate::output_count::push_output_count(Some(5)).unwrap();
    let output = block_on(runtime.scope(fsolve_builtin(
        Value::FunctionHandle("analytic_system".into()),
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap()),
        vec![Value::Struct(options.finish())],
    )))
    .unwrap();
    let Value::OutputList(outputs) = output else {
        panic!("outputs")
    };
    let Value::Tensor(ref x) = outputs[0] else {
        panic!("x")
    };
    let x = x.materialize_f64();
    assert!((x[0] - 1.0).abs() < 1e-5 && (x[1] - 2.0).abs() < 1e-5);
    let Value::Struct(ref info) = outputs[3] else {
        panic!("info")
    };
    assert!(
        matches!(info.fields["funcCount"],Value::Num(count) if count as usize == calls.load(Ordering::SeqCst))
    );
    let Value::Tensor(ref jacobian) = outputs[4] else {
        panic!("Jacobian")
    };
    assert_eq!(jacobian.materialize_f64().as_slice(), &[1.0, 3.0, 2.0, 4.0]);
}
