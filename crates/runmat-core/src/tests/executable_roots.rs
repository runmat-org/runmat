use futures::executor::block_on;
use runmat_value::{HandleRef, Value};

use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunMatSession,
};

#[test]
fn repeated_local_selection_keeps_original_definition_and_nested_handle_authority() {
    use runmat_runtime::{call::descriptor::CallableTarget, execution::RootedValue};
    runmat_gc::gc_test_context(|| {
        block_on(async {
            let mut session = RunMatSession::with_options(false, false).unwrap();
            let unit = session.compile_executable_unit(ExecutableSource::new(
                "local-selection@1", "outerCapture.m",
                "function h = outerCapture()\nstate = plain(0);\nh = @child;\nfunction y = child(x)\ny = x + state;\nend\nend\nfunction y = plain(x)\ny = x + 7;\nend\n",
            ), None).await.unwrap();
            let control = InvocationControl::default();
            let returned = session
                .invoke_executable(
                    &unit,
                    ProcedureInvocation {
                        target: ProcedureTarget::Function("outerCapture".into()),
                        arguments: crate::ProcedureArguments::Source(vec![]),
                        requested_outputs: 1,
                    },
                    &control,
                )
                .await
                .unwrap();
            let Value::Closure(closure) = &returned.as_slice()[0] else {
                panic!("the parsed nested callback must be an actual closure handle");
            };
            assert!(closure.binding.is_some());
            assert!(closure.lexical_environment.is_some());
            // Compiled units carry portable source ordinals, not live local
            // selection authority. Use the same original source-binding
            // projection that an actual executable invocation installs.
            let selected_runtime = unit
                .bind_runtime_sources(session.runtime_context().clone())
                .unwrap();
            let runtime = &selected_runtime;
            let functions = unit.runtime_bytecode(runtime).unwrap().function_registry();
            let plain = crate::callable_selection::select(
                runtime,
                &functions,
                RootedValue::in_context(Value::FunctionHandle("plain".into()), runtime).unwrap(),
            )
            .unwrap();
            let CallableTarget::Bound(original) = plain.selected_target() else {
                panic!(
                    "ordinary named function must retain its exact definition: target={:?}, published={:?}, bound={:?}",
                    plain.selected_target(),
                    functions.resolve_name("plain"),
                    functions.resolve_name("plain").and_then(|function| functions.exact_bound_binding(function))
                );
            };
            let function = functions.resolve_name("plain").unwrap();
            assert_eq!(*original, functions.exact_bound_binding(function).unwrap());
            let text = crate::callable_selection::select(
                runtime,
                &functions,
                RootedValue::in_context(Value::String("plain".into()), runtime).unwrap(),
            )
            .unwrap();
            assert_eq!(plain.selected_target(), text.selected_target());
            let builtin = crate::callable_selection::select(
                runtime,
                &functions,
                RootedValue::in_context(Value::String("abs".into()), runtime).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                builtin.selected_target(),
                CallableTarget::RetainedBuiltin(_)
            ));
            let actual = crate::callable_selection::select(
                runtime,
                &functions,
                RootedValue::in_context(returned.as_slice()[0].clone(), runtime).unwrap(),
            )
            .unwrap();
            let CallableTarget::BoundLexical {
                binding,
                closure: retained,
            } = actual.selected_target()
            else {
                panic!("nested callback selection must retain its binding and lexical activation");
            };
            assert_eq!(Some(binding), closure.binding.as_ref());
            assert!(retained.as_closure().lexical_environment.is_some());
            let name = closure.function_name.as_str();
            assert!(functions.resolve_name(name).is_none());
            let textual = crate::callable_selection::select(
                runtime,
                &functions,
                RootedValue::in_context(Value::String(name.into()), runtime).unwrap(),
            )
            .unwrap();
            assert!(
                !matches!(
                    textual.selected_target(),
                    CallableTarget::Bound(_) | CallableTarget::BoundLexical { .. }
                ),
                "text cannot manufacture nested binding authority"
            );
            let replacement = session
                .compile_executable_unit(
                    ExecutableSource::new(
                        "local-selection@2",
                        "plain.m",
                        "function y = plain(x)\ny = x + 100;\nend\nfunction y = abs(x)\ny = x + 200;\nend\n",
                    ),
                    None,
                )
                .await
                .unwrap();
            let replacement_runtime = replacement
                .bind_runtime_sources(session.runtime_context().clone())
                .unwrap();
            let replacement_functions = replacement
                .runtime_bytecode(&replacement_runtime)
                .unwrap()
                .function_registry();
            let changed = crate::callable_selection::select(
                &replacement_runtime,
                &replacement_functions,
                RootedValue::in_context(
                    Value::FunctionHandle("plain".into()),
                    &replacement_runtime,
                )
                .unwrap(),
            )
            .unwrap();
            assert_ne!(plain.selected_target(), changed.selected_target());
            let changed_builtin = crate::callable_selection::select(
                &replacement_runtime,
                &replacement_functions,
                RootedValue::in_context(Value::String("abs".into()), &replacement_runtime).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                changed_builtin.selected_target(),
                CallableTarget::Bound(_)
            ));
            assert!(matches!(
                builtin.selected_target(),
                CallableTarget::RetainedBuiltin(_)
            ));
            assert_eq!(plain.selected_target(), text.selected_target());
            for requested in [0, 1] {
                let arguments = runmat_runtime::execution::RootedValues::in_context(
                    vec![Value::Num(-3.)],
                    session.runtime_context(),
                )
                .unwrap();
                let result = builtin
                    .invoke_retained(Some(session.runtime_context()), arguments.into(), requested)
                    .await
                    .unwrap();
                if requested == 0 {
                    assert!(result.as_slice().is_empty());
                } else {
                    assert_eq!(result.as_slice(), &[Value::Num(3.)]);
                }
            }

            assert_eq!(*original, functions.exact_bound_binding(function).unwrap());
        });
    });
}

#[test]
fn public_invocation_roots_arguments_before_poll_and_retires_them_on_drop() {
    runmat_gc::gc_test_context(|| {
        for coverage in [false, true] {
            for execute in [false, true] {
                let mut session = RunMatSession::with_options(false, false).unwrap();
                let unit = block_on(session.compile_executable_unit(
                    ExecutableSource::new(
                        "invocation-roots@1",
                        "keepManaged.m",
                        "function out = keepManaged(value)\nout = value;\nend\n",
                    ),
                    None,
                ))
                .unwrap();
                let caller_root =
                    runmat_gc::gc_allocate_rooted(Value::String("payload".into())).unwrap();
                let handle = caller_root.handle();
                let invocation = ProcedureInvocation {
                    target: ProcedureTarget::Function("keepManaged".into()),
                    arguments: crate::ProcedureArguments::Source(vec![Value::HandleObject(
                        HandleRef {
                            class_name: "InvocationHandle".into(),
                            target: handle,
                            valid: true,
                        },
                    )]),
                    requested_outputs: 1,
                };
                let control = InvocationControl::default();
                let output = if coverage {
                    let pending =
                        session.invoke_executable_with_coverage(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    if execute {
                        Some(block_on(pending).unwrap().0)
                    } else {
                        drop(pending);
                        None
                    }
                } else {
                    let pending = session.invoke_executable(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    if execute {
                        Some(block_on(pending).unwrap())
                    } else {
                        drop(pending);
                        None
                    }
                };
                if let Some(output) = output.as_ref() {
                    assert!(matches!(
                        output.as_slice(),
                        [Value::HandleObject(result)] if result.target == handle
                    ));
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                }
                drop(output);
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_err());
            }
        }
    });
}

#[test]
fn rejected_invocations_check_control_before_registering_argument_roots() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "rejected-invocation@1",
                "keepManaged.m",
                "function out = keepManaged(value)\nout = value;\nend\n",
            ),
            None,
        ))
        .unwrap();
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        cancellation.cancel(runmat_execution::CancellationReason::User);
        for (control, identifier) in [
            (
                InvocationControl::default().with_cancellation(cancellation.token()),
                "RunMat:ExecutionCancelled",
            ),
            (
                InvocationControl::default().with_deadline_unix_ms(0),
                "RunMat:ExecutionDeadline",
            ),
        ] {
            for coverage in [false, true] {
                let caller_root =
                    runmat_gc::gc_allocate_rooted(Value::String("payload".into())).unwrap();
                let handle = caller_root.handle();
                let invocation = ProcedureInvocation {
                    target: ProcedureTarget::Function("keepManaged".into()),
                    arguments: crate::ProcedureArguments::Source(vec![Value::HandleObject(
                        HandleRef {
                            class_name: "InvocationHandle".into(),
                            target: handle,
                            valid: true,
                        },
                    )]),
                    requested_outputs: 1,
                };
                let error = if coverage {
                    let pending =
                        session.invoke_executable_with_coverage(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_err());
                    block_on(pending).unwrap_err()
                } else {
                    let pending = session.invoke_executable(&unit, invocation, &control);
                    caller_root.unroot().unwrap();
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_err());
                    block_on(pending).unwrap_err()
                };
                let crate::RunError::Runtime(error) = error else {
                    panic!("control rejection must retain the runtime error domain");
                };
                assert_eq!(error.identifier(), Some(identifier));
            }
        }
    });
}
