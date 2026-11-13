use super::*;
use std::sync::Arc;

#[test]
fn discarded_outputs_still_invoke_each_callback_once() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 92);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = calls.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_, _, outputs| {
            assert_eq!(outputs, 0);
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async {
                crate::sequence::ValueSequence::comma_separated(Vec::new())
                    .map_err(crate::sequence::sequence_error_to_runtime)
                    .and_then(rooted_sequence)
            })
        },
    )))
    .expect("call-port scope");
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let result = call(
        crate::user_functions::test_support::exact_local_value("effect_only", binding),
        numbers(),
        Vec::new(),
    )
    .unwrap();
    assert!(matches!(result.value(), Value::OutputList(values) if values.is_empty()));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn passes_requested_output_count_to_bound_callbacks() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 91);
    let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_, arguments, outputs| {
            assert_eq!(outputs, 2);
            let Value::Num(value) = arguments[0] else {
                panic!("expected numeric field")
            };
            Box::pin(async move {
                crate::sequence::ValueSequence::comma_separated(vec![
                    Value::Num(value),
                    Value::Num(value + 10.0),
                ])
                .map_err(crate::sequence::sequence_error_to_runtime)
                .and_then(rooted_sequence)
            })
        },
    )))
    .expect("call-port scope");
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = call(
        crate::user_functions::test_support::exact_local_value("two_outputs", binding),
        numbers(),
        Vec::new(),
    )
    .unwrap();
    let Value::OutputList(outputs) = result.value() else {
        panic!("expected output list")
    };
    assert_eq!(outputs.len(), 2);
    let Value::Tensor(first) = &outputs[0] else {
        panic!("expected tensor")
    };
    let Value::Tensor(second) = &outputs[1] else {
        panic!("expected tensor")
    };
    assert_eq!(first.materialize_f64(), vec![1.0, 2.0]);
    assert_eq!(second.materialize_f64(), vec![11.0, 12.0]);
}

#[test]
fn callback_errors_are_wrapped_without_text_rewriting() {
    let error = call(
        Value::FunctionHandle("missing_structfun_callback".into()),
        numbers(),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::STRUCTFUN_ERROR_FUNCTION_ERROR.identifier
    );
    assert!(error.message().contains("Undefined function"));
    assert!(!error.message().contains("cellfun:"));
}

#[test]
fn error_handler_receives_the_original_callback_identity() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _entry = runtime.enter();
    let failing = crate::user_functions::test_support::exact_local_binding(&runtime, 101);
    let recovery = crate::user_functions::test_support::exact_local_binding(&runtime, 102);
    let _bound = crate::user_functions::test_support::install_bound_forwarders(vec![
        failing.clone(),
        recovery.clone(),
    ]);
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            let context = arguments.first().cloned();
            Box::pin(async move {
                match function {
                    101 => Err(crate::build_runtime_error("field callback failed")
                        .with_identifier("RunMat:Test:OriginalCallback")
                        .build()),
                    102 => {
                        let Some(Value::Struct(context)) = context else {
                            panic!("expected callback error context")
                        };
                        assert_eq!(
                            context.fields.get("identifier"),
                            Some(&Value::String("RunMat:Test:OriginalCallback".into()))
                        );
                        assert_eq!(
                            context.fields.get("field"),
                            Some(&Value::String("a".into()))
                        );
                        Ok(Value::Num(7.0))
                    }
                    _ => panic!("unexpected semantic callback"),
                }
                .and_then(crate::sequence::single_value_sequence)
                .and_then(rooted_sequence)
            })
        },
    )))
    .expect("call-port scope");
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("a", Value::Num(1.0));
    let result = call(
        crate::user_functions::test_support::exact_local_value("fails", failing),
        structure.finish(),
        vec![
            Value::String("ErrorHandler".into()),
            crate::user_functions::test_support::exact_local_value("recovers", recovery),
        ],
    )
    .unwrap();
    let Value::Tensor(result) = result.value() else {
        panic!("expected uniform output")
    };
    assert_eq!(result.materialize_f64(), vec![7.0]);
}
