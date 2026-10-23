use super::*;
use runmat_value::Closure;
use std::sync::Arc;

mod errors;

#[test]
fn bound_callback_uses_semantic_identity() {
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 417);
    let expected = binding.clone();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_local_bound_function_invoker(std::rc::Rc::new(
        move |_runtime, actual, arguments, outputs, _lexical_environment| {
            assert_eq!(actual, expected);
            assert_eq!(outputs, 1);
            assert_eq!(arguments.capture_count(), 1);
            assert_eq!(arguments.supplied_count(), 1);
            let (Some(Value::Num(seed)), Some(Value::Num(value))) =
                (arguments.get(0), arguments.get(1))
            else {
                panic!("unexpected callback arguments: {arguments:?}");
            };
            let result = seed + value;
            crate::execution::admit_executable_sequence_with(
                &_runtime.clone(),
                move || async move {
                    crate::sequence::single_value_sequence(Value::Num(result))
                        .and_then(rooted_sequence)
                },
            )
        },
    ))
    .expect("call-port scope");
    drop(entry);
    let callback = Value::Closure(Closure::from_data(runmat_value::ClosureData {
        lexical_environment: None,
        function_name: "cellfun_bound_fixture".into(),
        binding: Some(binding),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
        captures: vec![Value::Num(10.0)].into(),
    }));
    let result = futures::executor::block_on(
        runtime.scope(super::cellfun_builtin(
            None,
            callback,
            crate::call::BuiltinArguments::from_borrowed(
                &[cell(vec![Value::Num(1.0), Value::Num(2.0)], &[1, 2])],
                Some(&runtime),
            )
            .unwrap(),
        )),
    )
    .unwrap();
    assert_eq!(tensor_values(result.value()), vec![11.0, 12.0]);
}

#[test]
fn name_callback_resolves_to_semantic_function() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let _fixtures = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[("project_callback", 418)],
    )
    .unwrap();
    let entry = runtime.enter();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |function, arguments, _| {
            assert_eq!(function, 418);
            let [Value::Num(value)] = arguments else {
                panic!("unexpected arguments")
            };
            let result = value * 2.0;
            Box::pin(async move {
                crate::sequence::single_value_sequence(Value::Num(result)).and_then(rooted_sequence)
            })
        },
    )))
    .expect("call-port scope");
    drop(entry);
    let result = futures::executor::block_on(
        runtime.scope(super::cellfun_builtin(
            None,
            Value::String("project_callback".into()),
            crate::call::BuiltinArguments::from_borrowed(
                &[cell(vec![Value::Num(3.0)], &[1, 1])],
                Some(&runtime),
            )
            .unwrap(),
        )),
    )
    .unwrap();
    assert_eq!(tensor_values(result.value()), vec![6.0]);
}

#[test]
fn requested_output_count_is_forwarded_and_collected_independently() {
    use std::sync::Arc;
    for requested in [0, 2] {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 191);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_, arguments, outputs| {
                assert_eq!(outputs, requested);
                observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Value::Num(value) = arguments[0] else {
                    panic!("numeric");
                };
                Box::pin(async move {
                    crate::sequence::ValueSequence::comma_separated(if requested == 0 {
                        vec![]
                    } else {
                        vec![Value::Num(value), Value::Num(value + 10.)]
                    })
                    .map_err(crate::sequence::sequence_error_to_runtime)
                    .and_then(rooted_sequence)
                })
            },
        )))
        .expect("call-port scope");
        let _outputs =
            crate::output_count::push_output_count(Some(requested)).expect("output-count scope");
        let result = call(
            crate::user_functions::test_support::exact_local_value("multiple", binding),
            vec![cell(vec![Value::Num(1.), Value::Num(2.)], &[1, 2])],
        )
        .unwrap();
        let Value::OutputList(values) = result.value() else {
            panic!("output list");
        };
        assert_eq!(values.len(), requested);
        if requested == 2 {
            assert_eq!(tensor_values(&values[0]), vec![1., 2.]);
            assert_eq!(tensor_values(&values[1]), vec![11., 12.]);
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
