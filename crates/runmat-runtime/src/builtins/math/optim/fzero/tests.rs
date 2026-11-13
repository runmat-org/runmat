use super::*;
use crate::builtins::math::optim::brent::interpolation_step_accepted;
use futures::executor::block_on;
use runmat_builtins::FZERO_DESCRIPTOR;
use runmat_value::{IntegerStorage, Tensor};
use std::sync::Arc;

#[test]
fn registered_fzero_preserves_invocation_cancellation_before_solver_effects() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(fzero_builtin_registered(
        Some(runtime),
        Value::FunctionHandle("sin".into()),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cancelled invocation must not call callback");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn dropping_a_pending_callback_abandons_the_solver_without_publication() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7_711);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    struct DropCount(Arc<AtomicUsize>);
    impl Drop for DropCount {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let observed_calls = calls.clone();
    let observed_drop = dropped.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, _args, _requested_outputs| {
            observed_calls.fetch_add(1, Ordering::SeqCst);
            let guard = DropCount(observed_drop.clone());
            Box::pin(async move {
                let _guard = guard;
                futures::future::pending::<()>().await;
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(0.0))
            })
        },
    )))
    .expect("call-port scope");
    let mut call = Box::pin(fzero_builtin(
        crate::user_functions::test_support::exact_local_value("pending_callback", binding),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(call);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn fzero_bracketed_builtin_handle() {
    let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let root = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-6),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_accepts_admitted_logical_interval_without_widening_its_control_class() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let interval = runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
    let root = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::LogicalArray(interval),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(matches!(root, Value::Num(value) if value == 0.0));
}

#[test]
fn fzero_single_interval_returns_single_numerical_outputs() {
    let bracket =
        Tensor::new_with_dtype(vec![3.0, 4.0], vec![1, 2], runmat_value::NumericDType::F32)
            .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::OutputList(outputs) = result else {
        panic!("expected three outputs");
    };
    assert_eq!(outputs.len(), 3);
    for output in outputs.try_into_owned().expect("unique output list") {
        assert!(
            matches!(output, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
        );
    }
}

#[test]
fn fzero_single_precision_is_explicitly_threaded_through_all_callback_calls() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7_709);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed_calls = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, _requested_outputs| {
            observed_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Value::Tensor(argument) = &args[0] else {
                panic!("expected a single-precision scalar tensor")
            };
            assert_eq!(argument.numeric_dtype(), runmat_value::NumericDType::F32);
            let residual = argument.materialize_f64()[0] - std::f64::consts::PI;
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    Tensor::new_with_dtype(
                        vec![residual],
                        vec![1, 1],
                        runmat_value::NumericDType::F32,
                    )
                    .unwrap(),
                ))
            })
        },
    )))
    .expect("call-port scope");
    let bracket =
        Tensor::new_with_dtype(vec![3.0, 4.0], vec![1, 2], runmat_value::NumericDType::F32)
            .unwrap();
    let root = block_on(fzero_builtin(
        crate::user_functions::test_support::exact_local_value("single_root", binding),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(
        matches!(root, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
    );
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 2);
}

#[test]
fn fzero_single_callback_result_selects_single_default_tolerance() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7_710);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_function, args, _requested_outputs| {
            assert!(matches!(args[0], Value::Num(_)));
            let x = match &args[0] {
                Value::Num(value) => *value,
                _ => unreachable!(),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    Tensor::new_with_dtype(
                        vec![x - 1.0],
                        vec![1, 1],
                        runmat_value::NumericDType::F32,
                    )
                    .unwrap(),
                ))
            })
        },
    )))
    .expect("call-port scope");
    let options = FzeroOptions::from_struct(None, None).unwrap();
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let bracket = block_on(bracket_from_endpoints(
        &crate::user_functions::test_support::exact_local_value("single_result", binding),
        0.0,
        2.0,
        false,
        &mut cooperation,
    ))
    .unwrap();
    assert!(bracket.function_single);
    assert_eq!(default_tol_x(bracket.function_single), f32::EPSILON as f64);
    assert_eq!(default_tol_x(false), f64::EPSILON);
    assert!(!options.tol_x_explicit);
}

#[test]
fn fzero_bracket_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let bracket =
        Tensor::new_integer(IntegerStorage::U16(vec![3, 4]), vec![1, 2]).expect("bracket");

    let root = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - std::f64::consts::PI).abs() < 1.0e-6),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_scalar_initial_guess_expands_bracket() {
    let root = block_on(fzero_builtin(
        Value::FunctionHandle("cos".into()),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - std::f64::consts::FRAC_PI_2).abs() < 1.0e-6),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_initial_guess_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let guess = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("guess");

    let root = block_on(fzero_builtin(
        Value::FunctionHandle("cos".into()),
        Value::Tensor(guess),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - std::f64::consts::FRAC_PI_2).abs() < 1.0e-6),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_scalar_initial_guess_uses_center_sign_for_bracket() {
    let root = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(std::f64::consts::FRAC_PI_2),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!(n.abs() < 1.0e-6),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn scalar_expansion_near_both_finite_extremes_never_calls_nonfinite_abscissa() {
    for start in [-f64::MAX, f64::MAX, f64::from_bits(1)] {
        let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
        let options = FzeroOptions::from_struct(None, None).unwrap();
        let bracket = block_on(expand_bracket(
            &Value::FunctionHandle("sign".into()),
            start,
            &options,
            false,
            &mut cooperation,
        ))
        .unwrap();
        assert!(bracket.a.is_finite() && bracket.b.is_finite());
        assert!(
            bracket.fa == 0.0 || bracket.fb == 0.0 || bracket.fa.signum() != bracket.fb.signum()
        );
    }
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let zero = block_on(expand_bracket(
        &Value::FunctionHandle("sign".into()),
        0.0,
        &FzeroOptions::from_struct(None, None).unwrap(),
        false,
        &mut cooperation,
    ))
    .unwrap();
    assert_eq!(zero.evals, 1);
    assert_eq!(zero.a, 0.0);
}

#[test]
fn scalar_expansion_honors_eval_budget_and_stops_at_finite_domain_exhaustion() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 42);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    use std::sync::atomic::{AtomicUsize, Ordering};

    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, requested_outputs| {
            assert_eq!(requested_outputs, 1);
            assert!(matches!(&args[0], Value::Num(value) if value.is_finite()));
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(1.0))
            })
        },
    )))
    .expect("call-port scope");
    let function =
        crate::user_functions::test_support::exact_local_value("constant_positive", binding);
    let mut options = FzeroOptions::from_struct(None, None).unwrap();
    options.max_fun_evals = 3;
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let limited = block_on(expand_bracket(
        &function,
        f64::MAX,
        &options,
        false,
        &mut cooperation,
    ))
    .unwrap_err();
    assert!(limited
        .message()
        .contains("could not find a sign-changing bracket"));
    assert_eq!(calls.load(Ordering::SeqCst), 3);

    options.max_fun_evals = 500;
    let mut cooperation = ScalarCooperation::new(OptimSolver::Fzero, None);
    let exhausted = block_on(expand_bracket(
        &function,
        f64::MAX,
        &options,
        false,
        &mut cooperation,
    ))
    .unwrap_err();
    assert!(exhausted
        .message()
        .contains("could not find a sign-changing bracket"));
    let total = calls.load(Ordering::SeqCst);
    assert!(total > 3 && total < 503, "{total}");
}
#[test]
fn fzero_accepts_semantic_function_handle_callback() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 42);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, args, requested_outputs| {
            assert_eq!(function, 42);
            assert_eq!(requested_outputs, 1);
            let x = match &args[0] {
                Value::Num(value) => *value,
                other => panic!("expected scalar numeric argument, got {other:?}"),
            };
            Box::pin(async move {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                    x - 2.0,
                ))
            })
        },
    )))
    .expect("call-port scope");

    let root = block_on(fzero_builtin(
        crate::user_functions::test_support::exact_local_value("root_function", binding),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    match root {
        Value::Num(n) => assert!((n - 2.0).abs() < 1.0e-6),
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_multi_output_two_returns_root_and_fval() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let result = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fzero");
    match result {
        Value::OutputList(outputs) => {
            assert_eq!(outputs.len(), 2);
            match (&outputs[0], &outputs[1]) {
                (Value::Num(x), Value::Num(fval)) => {
                    assert!((x - std::f64::consts::PI).abs() < 1.0e-6);
                    assert!(fval.abs() < 1.0e-6);
                }
                other => panic!("unexpected outputs {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_multi_output_four_includes_output_struct() {
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let result = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("fzero");
    match result {
        Value::OutputList(outputs) => {
            assert_eq!(outputs.len(), 4);
            assert!(matches!(&outputs[2], Value::Num(flag) if *flag == 1.0));
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
                    match output.fields.get("algorithm") {
                        Some(Value::String(runmat_value::StringElement::Text(text))) => {
                            assert!(text.contains("bisection"))
                        }
                        other => panic!("unexpected algorithm field {other:?}"),
                    }
                    match output.fields.get("message") {
                        Some(Value::String(runmat_value::StringElement::Text(text))) => {
                            assert!(text.contains("Zero found"))
                        }
                        other => panic!("unexpected message field {other:?}"),
                    }
                }
                other => panic!("unexpected output struct {other:?}"),
            }
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_reports_zero_exitflag_when_iteration_budget_exhausted() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Num(1.0));
    opts.insert("MaxFunEvals", Value::Num(2.0));
    opts.insert("Display", Value::from("off"));
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let result = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .expect("fzero");
    match result {
        Value::OutputList(outputs) => match &outputs[2] {
            Value::Num(flag) => assert_eq!(*flag, 0.0),
            other => panic!("unexpected exitflag {other:?}"),
        },
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_reports_convergence_when_final_step_hits_root() {
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("MaxIter", Value::Num(1.0));
    opts.insert("Display", Value::from("off"));
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let bracket = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
    let result = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .expect("fzero");
    match result {
        Value::OutputList(outputs) => {
            assert!(matches!(&outputs[0], Value::Num(x) if x.abs() < 1.0e-12));
            assert!(matches!(&outputs[1], Value::Num(fval) if fval.abs() < 1.0e-12));
            assert!(matches!(&outputs[2], Value::Num(flag) if *flag == 1.0));
        }
        other => panic!("unexpected value {other:?}"),
    }
}

#[test]
fn fzero_rejects_more_than_four_outputs() {
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let err = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("too many outputs should fail");
    assert_eq!(err.identifier(), Some("RunMat:fzero:TooManyOutputs"));
    assert!(err.message().contains("maximum is 4"));
}

#[test]
fn fzero_iter_display_records_iteration_rows() {
    crate::console::reset_thread_buffer();
    let mut opts = runmat_value::UnboundedStructValueBuilder::new();
    opts.insert("Display", Value::from("iter"));
    let bracket = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
    let result = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Tensor(bracket),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
            opts.finish(),
        )]),
    ))
    .expect("fzero");
    assert!(matches!(result, Value::Num(_)));

    let joined =
        crate::console::take_thread_buffer()
            .into_iter()
            .fold(String::new(), |mut text, entry| {
                text.push_str(entry.text.as_str());
                text
            });
    assert!(joined.contains("Func-count"), "{joined}");
    assert!(joined.contains("initial"), "{joined}");
    assert!(
        joined.contains("interpolation") || joined.contains("bisection"),
        "{joined}"
    );
    assert!(joined.contains("exitflag = 1"), "{joined}");
}

#[test]
fn brent_interpolation_acceptance_uses_signed_q() {
    assert!(!interpolation_step_accepted(1.0, -2.0, 1.0, 0.1, 10.0));
    assert!(interpolation_step_accepted(1.0, -2.0, -1.0, 0.1, 10.0));
}

#[test]
fn fzero_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FZERO_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "fzero(fun, x0)",
            "fzero(fun, x0, options)",
            "x = fzero(fun, x0)",
            "x = fzero(fun, x0, options)",
            "[x, fval] = fzero(fun, x0)",
            "[x, fval] = fzero(fun, x0, options)",
            "[x, fval, exitflag] = fzero(fun, x0)",
            "[x, fval, exitflag] = fzero(fun, x0, options)",
            "[x, fval, exitflag, output] = fzero(fun, x0)",
            "[x, fval, exitflag, output] = fzero(fun, x0, options)",
        ]
    );

    let codes: Vec<&str> = FZERO_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert_eq!(
        codes,
        vec![
            "RM.FZERO.INVALID_ARGUMENT",
            "RM.FZERO.INVALID_INPUT",
            "RM.FZERO.TOO_MANY_OUTPUTS",
        ]
    );
}

#[test]
fn fzero_too_many_args_uses_stable_identifier() {
    let err = block_on(fzero_builtin(
        Value::FunctionHandle("sin".into()),
        Value::Num(0.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Struct(StructValue::new()),
            Value::Struct(StructValue::new()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:fzero:InvalidArgument"));
}
