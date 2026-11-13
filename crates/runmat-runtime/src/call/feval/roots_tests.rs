use std::cell::Cell;
use std::rc::Rc;
use std::task::{Context, Poll};

use runmat_value::{CellArray, Closure, HandleRef, LexicalEnvironment, Value};

#[test]
fn selected_callback_rebinds_old_segments_but_keeps_real_captures_and_activation() {
    runmat_gc::gc_test_context(|| {
        struct Resolver;
        impl crate::call::descriptor::FunctionNameResolver for Resolver {
            fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
                None
            }
        }
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 33);
        let activation = runmat_gc::gc_allocate_rooted(Value::Num(40.)).unwrap();
        let captured = runmat_gc::gc_allocate_rooted(Value::Num(41.)).unwrap();
        let activation_handle = activation.handle();
        let captured_handle = captured.handle();
        let target = crate::execution::RootedValue::in_context(
            Value::Closure(Closure::from_data(runmat_value::ClosureData {
                function_name: "nested_rebind".into(),
                binding: Some(binding.clone()),
                source_role: runmat_types::FunctionSourceRole::Nested,
                source_file: None,
                captures: vec![Value::HandleObject(HandleRef {
                    class_name: "CallbackCapture".into(),
                    target: captured_handle,
                    valid: true,
                })]
                .into(),
                lexical_environment: Some(LexicalEnvironment::new(activation_handle, None)),
            })),
            &runtime,
        )
        .unwrap();
        let selected = crate::call::descriptor::RepeatableCallable::select_rooted(
            Some(&runtime),
            target,
            &Resolver,
        )
        .unwrap();
        match selected.selected_target() {
            crate::call::descriptor::CallableTarget::BoundLexical {
                binding: actual,
                closure,
            } => {
                assert_eq!(actual, &binding);
                assert!(closure.as_closure().lexical_environment.is_some());
            }
            _ => panic!("selected callback must retain its exact lexical binding"),
        }
        let old = crate::execution::RetainedValueReadLease::prepare_owned(
            Value::Closure(Closure::new(
                "old_caller".into(),
                None,
                runmat_types::FunctionSourceRole::Anonymous,
                None,
                vec![Value::Num(1.)],
                None,
            )),
            &runtime,
        )
        .unwrap();
        let arguments = crate::call::RootedFunctionArguments::captured(
            old,
            crate::execution::RootedValues::new(vec![Value::Num(2.)]).unwrap(),
        )
        .unwrap()
        .prepend_routing(Value::Num(0.), Some(&runtime))
        .unwrap();
        let prepared = selected
            .prepare_arguments(Some(&runtime), arguments.into())
            .unwrap();
        assert_eq!(prepared.capture_count(), 1);
        assert_eq!(prepared.supplied_count(), 3);
        assert_eq!(prepared.get(1), Some(&Value::Num(0.)));
        assert_eq!(prepared.get(2), Some(&Value::Num(1.)));
        assert_eq!(prepared.get(3), Some(&Value::Num(2.)));
        activation.unroot().unwrap();
        captured.unroot().unwrap();
        drop(selected);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&activation_handle, |_| ()).is_ok());
        assert!(runmat_gc::gc_with_value(&captured_handle, |_| ()).is_ok());
        drop(prepared);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&activation_handle, |_| ()).is_err());
        assert!(runmat_gc::gc_with_value(&captured_handle, |_| ()).is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn public_feval_future_roots_unpolled_and_pending_nested_inputs_until_drop() {
    runmat_gc::gc_test_context(|| {
        for retained in [false, true] {
            let runtime = crate::context::RuntimeContext::new(Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ));
            let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 31);
            let expected_binding = binding.clone();
            let reached = Rc::new(Cell::new(false));
            let reached_invoker = Rc::clone(&reached);
            let entry = runtime.enter();
            let _invoker = crate::user_functions::install_local_bound_function_invoker(Rc::new(
                move |bound_runtime, actual, arguments, _, environment| {
                    assert_eq!(actual, expected_binding);
                    assert!(
                        environment.is_some(),
                        "nested activation must reach the invoker"
                    );
                    reached_invoker.set(true);
                    crate::execution::admit_executable_sequence_with(
                        &bound_runtime.clone(),
                        move || async move {
                            let _arguments = arguments;
                            let _environment = environment;
                            std::future::pending::<()>().await;
                            unreachable!("pending test invocation cannot complete")
                        },
                    )
                },
            ))
            .expect("call-port scope");
            drop(entry);

            let activation = runmat_gc::gc_allocate_rooted(Value::Cell(
                CellArray::new(Vec::new(), 1, 0).expect("empty activation"),
            ))
            .expect("root activation");
            let argument = runmat_gc::gc_allocate_rooted(Value::String("input".into()))
                .expect("root argument");
            let activation_handle = activation.handle();
            let argument_handle = argument.handle();
            let closure = Value::Closure(Closure::from_data(runmat_value::ClosureData {
                function_name: "nested".into(),
                binding: Some(binding),
                source_role: runmat_types::FunctionSourceRole::Nested,
                source_file: None,
                captures: Vec::new().into(),
                lexical_environment: Some(LexicalEnvironment::new(activation_handle, None)),
            }));
            let arguments = [Value::HandleObject(HandleRef {
                class_name: "PendingArgument".into(),
                target: argument_handle,
                valid: true,
            })];
            // Only this test boundary erases the two concrete future types. Both
            // wrappers must preserve the same original nested activation and input.
            let mut future: futures::future::LocalBoxFuture<
                '_,
                crate::BuiltinResult<crate::execution::RootedValueSequence>,
            > = if retained {
                let owned = crate::execution::RootedValues::try_initialize_in_context(
                    arguments.len(),
                    &runtime,
                    |index| Ok::<_, crate::RuntimeError>(arguments[index].clone()),
                )
                .expect("original retained argument owner");
                Box::pin(
                    crate::call_feval_retained_sequence_async_with_outputs_in_context(
                        &runtime,
                        closure,
                        owned.into(),
                        1,
                    ),
                )
            } else {
                Box::pin(crate::call_feval_sequence_async_with_outputs_in_context(
                    &runtime, closure, &arguments, 1,
                ))
            };
            activation.unroot().expect("transfer activation root");
            argument.unroot().expect("transfer argument root");
            runmat_gc::gc_collect_major().expect("collect unpolled future");
            assert!(!reached.get(), "admission must not invoke user code");
            assert!(runmat_gc::gc_with_value(&activation_handle, |_| ()).is_ok());
            assert!(runmat_gc::gc_with_value(&argument_handle, |_| ()).is_ok());

            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            assert!(reached.get(), "the real selected invoker must be pending");
            runmat_gc::gc_collect_major().expect("collect pending future");
            assert!(runmat_gc::gc_with_value(&activation_handle, |_| ()).is_ok());
            assert!(runmat_gc::gc_with_value(&argument_handle, |_| ()).is_ok());

            drop(future);
            runmat_gc::gc_collect_major().expect("collect dropped future");
            assert!(runmat_gc::gc_with_value(&activation_handle, |_| ()).is_err());
            assert!(runmat_gc::gc_with_value(&argument_handle, |_| ()).is_err());
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn mapped_nested_closure_uses_explicit_runtime_without_ambient_or_foreign_ambient() {
    runmat_gc::gc_test_context(|| {
        let selected = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let foreign = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let binding = crate::user_functions::test_support::exact_local_binding(&selected, 32);
        let selected_identity = selected.local_identity();
        let selected_calls = Rc::new(Cell::new(0));
        let selected_observed = Rc::clone(&selected_calls);
        let selected_entry = selected.enter();
        let _selected_invoker = crate::user_functions::install_local_bound_function_invoker(
            Rc::new(move |runtime, _, _, _, environment| {
                assert_eq!(runtime.local_identity(), selected_identity);
                assert!(environment.is_some());
                selected_observed.set(selected_observed.get() + 1);
                crate::execution::admit_executable_sequence_with(
                    &runtime.clone(),
                    move || async move {
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(32.0),
                        )
                    },
                )
            }),
        )
        .expect("call-port scope");
        drop(selected_entry);

        let foreign_calls = Rc::new(Cell::new(0));
        let foreign_observed = Rc::clone(&foreign_calls);
        let foreign_entry = foreign.enter();
        let _foreign_invoker = crate::user_functions::install_local_bound_function_invoker(
            Rc::new(move |bound_runtime, _, _, _, _| {
                foreign_observed.set(foreign_observed.get() + 1);
                crate::execution::admit_executable_sequence_with(
                    &bound_runtime.clone(),
                    move || async move {
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(-1.0),
                        )
                    },
                )
            }),
        )
        .expect("call-port scope");
        drop(foreign_entry);

        let activation = runmat_gc::gc_allocate_rooted(Value::Cell(
            CellArray::new(Vec::new(), 1, 0).expect("empty activation"),
        ))
        .expect("root activation");
        let mapped = crate::builtins::common::mapped_callable::MappedCallable::Closure(
            crate::call::descriptor::RetainedClosureValue::new(Closure::from_data(
                runmat_value::ClosureData {
                    function_name: "nested".into(),
                    binding: Some(binding),
                    source_role: runmat_types::FunctionSourceRole::Nested,
                    source_file: None,
                    captures: Vec::new().into(),
                    lexical_environment: Some(LexicalEnvironment::new(activation.handle(), None)),
                },
            )),
        );
        let no_ambient = futures::executor::block_on(mapped.invoke_with_outputs_in_context(
            Some(&selected),
            &[],
            1,
        ))
        .unwrap_or_else(|_| panic!("explicit selected runtime without ambient"));
        assert_eq!(no_ambient, Value::Num(32.0));
        let foreign_entry = foreign.enter();
        let foreign_ambient = futures::executor::block_on(mapped.invoke_with_outputs_in_context(
            Some(&selected),
            &[],
            1,
        ))
        .unwrap_or_else(|_| panic!("explicit selected runtime overrides foreign ambient"));
        drop(foreign_entry);
        assert_eq!(foreign_ambient, Value::Num(32.0));
        assert_eq!(selected_calls.get(), 2);
        assert_eq!(foreign_calls.get(), 0);
    });
}
