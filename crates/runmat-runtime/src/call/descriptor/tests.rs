use super::*;
use crate::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_types::BuiltinId;
use std::{cell::Cell, rc::Rc};

#[test]
fn owned_name_classification_preserves_qualified_and_builtin_identity_with_deferred_resolver() {
    struct DeferredResolver;
    impl FunctionNameResolver for DeferredResolver {
        fn resolve_function(&self, _: &str) -> Option<FunctionId> {
            None
        }
    }
    for target in [
        Value::ExternalFunctionHandle("pkg.fn".into()),
        Value::function_handle("logical".into()),
    ] {
        let descriptor =
            CallableDescriptor::from_feval_value(target, Vec::new(), 0, &DeferredResolver);
        let CallableTarget::Resolved {
            identity,
            fallback_policy,
        } = descriptor.target
        else {
            panic!("owned selector retains its established resolved identity contract")
        };
        match identity.as_identity() {
            CallableIdentity::Builtin(name) => {
                assert_eq!(name.0, "logical");
                assert_eq!(fallback_policy, CallableFallbackPolicy::None);
            }
            _ => {
                assert_eq!(
                    identity.as_identity(),
                    &crate::external_callable_identity_for_name("pkg.fn")
                );
                assert_eq!(fallback_policy, CallableFallbackPolicy::ExternalBoundary);
            }
        }
    }
}

#[test]
fn retained_descriptor_requires_target_only_selection() {
    let selected = CallableDescriptor::resolved(
        CallableIdentity::Builtin(BuiltinId("numel".into())),
        vec![Value::Num(1.)],
        1,
        CallableFallbackPolicy::None,
        CallableCallKind::Direct,
    );
    let arguments = crate::execution::RootedValues::new(Vec::new()).unwrap();
    assert!(selected.with_retained_arguments(arguments.into()).is_err());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn retained_arguments_survive_required_and_optional_cold_target_futures() {
    fn check(future: impl std::future::Future, handle: runmat_gc_api::GcHandle) {
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());
        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
    }
    for optional in [false, true] {
        for identity in [
            CallableIdentity::Builtin(BuiltinId("numel".into())),
            CallableIdentity::BoundFunction(FunctionId(9)),
            CallableIdentity::ExternalFunction {
                function: FunctionId(9),
                display_name: "external".into(),
            },
        ] {
            runmat_gc::gc_test_context(|| {
                let root = runmat_gc::gc_allocate_rooted(Value::Num(17.)).unwrap();
                let handle = root.handle();
                let arguments = crate::execution::RootedValues::new(vec![Value::HandleObject(
                    runmat_value::HandleRef {
                        class_name: "ColdArguments".into(),
                        target: handle,
                        valid: true,
                    },
                )])
                .unwrap();
                let pointer = arguments.values().as_ptr();
                let descriptor = CallableDescriptor::resolved(
                    identity,
                    Vec::new(),
                    1,
                    CallableFallbackPolicy::None,
                    CallableCallKind::Direct,
                )
                .with_retained_arguments(arguments.into())
                .unwrap();
                assert_eq!(descriptor.args.first().unwrap() as *const Value, pointer);
                root.unroot().unwrap();
                if optional {
                    check(
                        try_execute_retained_callable_descriptor(None, descriptor),
                        handle,
                    )
                } else {
                    check(
                        execute_retained_callable_descriptor(None, descriptor),
                        handle,
                    )
                }
            });
        }
    }
}

#[test]
fn retained_identity_survives_required_and_optional_cold_futures_without_a_new_owner() {
    for identity in [
        crate::call::identity::external_qualified_identity("pkg.Point", "origin"),
        CallableIdentity::BoundFunction(FunctionId(9)),
        CallableIdentity::Builtin(BuiltinId("numel".into())),
        CallableIdentity::ExternalFunction {
            function: FunctionId(9),
            display_name: "external".into(),
        },
    ] {
        let source = std::sync::Arc::new(identity);
        let descriptor = CallableDescriptor::resolved_retained(
            std::sync::Arc::clone(&source),
            Vec::new(),
            1,
            CallableFallbackPolicy::ExternalBoundary,
            CallableCallKind::Direct,
        );
        let CallableTarget::Resolved {
            identity: CallableIdentityOwner::Retained(owner),
            ..
        } = &descriptor.target
        else {
            panic!("descriptor replaced the retained identity")
        };
        assert!(std::sync::Arc::ptr_eq(&source, owner));
        let future = execute_callable_descriptor(descriptor);
        assert_eq!(std::sync::Arc::strong_count(&source), 2);
        drop(future);
        assert_eq!(std::sync::Arc::strong_count(&source), 1);

        let descriptor = CallableDescriptor::resolved_retained(
            std::sync::Arc::clone(&source),
            Vec::new(),
            1,
            CallableFallbackPolicy::ExternalBoundary,
            CallableCallKind::Direct,
        );
        let future = try_execute_callable_descriptor(descriptor);
        assert_eq!(std::sync::Arc::strong_count(&source), 2);
        drop(future);
        assert_eq!(std::sync::Arc::strong_count(&source), 1);
    }
}

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static DIAGNOSTIC: Cell<&'static str> = const { Cell::new("RunMat:UndefinedFunction") };
}

struct FailingBuiltin;

thread_local! {
    static CAPTURE_POINTERS: Cell<[usize; 4]> = const { Cell::new([0; 4]) };
    static CAPTURE_READY: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_DENIED: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_EFFECTS: Cell<usize> = const { Cell::new(0) };
}

struct CapturedBuiltin;
impl RuntimeBuiltinService for CapturedBuiltin {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        if name != "horzcat" {
            return Vec::new();
        }
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        vec![crate::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            captured_builtin,
        )
        .with_continuation_layout(|| crate::builtin::continuation_layout_for(captured_body))]
    }
}
fn captured_builtin(
    invocation: crate::builtin::RuntimeBuiltinInvocation<'_>,
) -> crate::builtin::RuntimeBuiltinFuture {
    let input = match invocation.into_continuation_input() {
        Ok(input) => input,
        Err(error) => {
            return crate::builtin::admitted_builtin_future(
                None,
                "fixture",
                std::future::ready(Err(error)),
            )
        }
    };
    let runtime = input.context.clone();
    crate::builtin::admitted_builtin_future_with(runtime, "horzcat", move |_| captured_body(input))
}
async fn captured_body(
    input: crate::builtin::BuiltinContinuationInput,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    let original = input.args.fork().into_function_arguments();
    assert_eq!(original.capture_count(), 1);
    assert_eq!(original.supplied_count(), 1);
    assert_eq!(original.supplied(), &[Value::Num(73.)]);
    let Value::Object(object) = &original.captures()[0] else {
        panic!("original object capture")
    };
    let Value::Struct(record) = object.property("record").unwrap() else {
        panic!("nested structure")
    };
    let Value::String(text) = record.fields.get("text").unwrap() else {
        panic!("nested text")
    };
    let Value::Tensor(tensor) = record.fields.get("tensor").unwrap() else {
        panic!("nested tensor")
    };
    CAPTURE_POINTERS.with(|pointers| {
        assert_eq!(
            pointers.get(),
            [
                object.properties().entries().as_ptr() as usize,
                record.fields.entries().as_ptr() as usize,
                text.as_str().expect("nonmissing text").as_ptr() as usize,
                tensor.as_f64_slice().unwrap().as_ptr() as usize,
            ]
        )
    });
    // Both success and denial exercise the real final-slot admission factory.
    // Denial exceeds this test host's finite capacity before a callback effect.
    let count = if CAPTURE_DENIED.with(Cell::get) {
        1 << 20
    } else {
        input.args.len()
    };
    let headers = input.args.fork().try_map_preserving_roots(
        count,
        input.context.as_ref(),
        |index, source| Ok(source[index % source.len()].clone()),
    )?;
    CAPTURE_EFFECTS.with(|effects| effects.set(effects.get() + 1));
    std::future::poll_fn(|_| {
        if CAPTURE_READY.with(Cell::get) {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    })
    .await;
    assert_eq!(headers.len(), 2);
    Ok(crate::execution::RootedValueSequence::empty())
}

#[test]
fn bounded_captured_builtin_preserves_original_generations_and_pending_roots() {
    use crate::context::host_allocation::HostAllocationError;
    use crate::context::{
        host_effect::{HostEffectAllocationError, ManagedHostEffectLedger},
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService,
    };
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                cpu_millicores_available: 1000,
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    struct Resolver;
    impl FunctionNameResolver for Resolver {
        fn name_resolution_completeness(&self) -> NameResolutionCompleteness {
            NameResolutionCompleteness::Complete
        }

        fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
            None
        }
    }
    for dynamic in [false, true] {
        for outcome in ["complete", "drop", "deny"] {
            runmat_gc::gc_test_context(|| {
                CAPTURE_READY.with(|ready| ready.set(false));
                CAPTURE_DENIED.with(|denied| denied.set(outcome == "deny"));
                CAPTURE_EFFECTS.with(|effects| effects.set(0));
                let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 1);
                let runtime =
                    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                        .with_service_ports(
                            RuntimeServicePorts::default()
                                .with_parallel(Rc::new(Parallel(ledger.clone())))
                                .with_builtin(Rc::new(CapturedBuiltin)),
                        );
                let root = runmat_gc::gc_allocate_rooted(Value::Num(19.)).unwrap();
                let handle = root.handle();
                let tensor = runmat_value::Tensor::new(vec![2., 3., 5.], vec![1, 3]).unwrap();
                let mut record = runmat_value::UnboundedStructValueBuilder::new();
                record.insert("text", Value::string("original captured text"));
                record.insert("tensor", Value::Tensor(tensor));
                record.insert(
                    "handle",
                    Value::HandleObject(runmat_value::HandleRef {
                        class_name: "CapturedBuiltinFixture".into(),
                        target: handle,
                        valid: true,
                    }),
                );
                let record = record.finish();
                let mut object =
                    runmat_value::UnboundedObjectBuilder::new("CapturedBuiltinFixture");
                object.insert("record".into(), Value::Struct(record));
                let object = object.finish();
                let Value::Struct(record) = object.property("record").unwrap() else {
                    unreachable!()
                };
                let Value::String(text) = record.fields.get("text").unwrap() else {
                    unreachable!()
                };
                let Value::Tensor(tensor) = record.fields.get("tensor").unwrap() else {
                    unreachable!()
                };
                CAPTURE_POINTERS.with(|pointers| {
                    pointers.set([
                        object.properties().entries().as_ptr() as usize,
                        record.fields.entries().as_ptr() as usize,
                        text.as_str().expect("nonmissing text").as_ptr() as usize,
                        tensor.as_f64_slice().unwrap().as_ptr() as usize,
                    ])
                });
                let closure = runmat_value::Closure::new(
                    "horzcat".into(),
                    None,
                    runmat_types::FunctionSourceRole::Anonymous,
                    None,
                    vec![Value::Object(object)],
                    None,
                );
                let mut descriptor = CallableDescriptor::from_feval_value(
                    Value::Closure(closure),
                    Vec::new(),
                    0,
                    &Resolver,
                );
                let CallableTarget::CapturedResolved {
                    identity,
                    fallback_policy,
                    ..
                } = &mut descriptor.target
                else {
                    panic!("captured descriptor")
                };
                assert!(matches!(
                    identity.as_identity(),
                    CallableIdentity::Builtin(_)
                ));
                if dynamic {
                    // Exercise semantic Continue on the same selected capture
                    // generation, not the new-callee normalization adapter.
                    *identity =
                        CallableIdentity::DynamicName(runmat_types::SymbolName("horzcat".into()))
                            .into();
                    *fallback_policy = CallableFallbackPolicy::RuntimeNameResolution;
                }
                let supplied =
                    crate::execution::RootedValues::try_initialize_in_context(1, &runtime, |_| {
                        Ok::<_, RuntimeError>(Value::Num(73.))
                    })
                    .expect("admit the original supplied argument owner");
                let descriptor = descriptor.with_retained_arguments(supplied.into()).unwrap();
                let mut future = Box::pin(execute_retained_callable_descriptor(
                    Some(&runtime),
                    descriptor,
                ));
                root.unroot().unwrap();
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                let mut context = Context::from_waker(futures::task::noop_waker_ref());
                let polled = future.as_mut().poll(&mut context);
                if outcome == "deny" {
                    let Poll::Ready(Err(error)) = polled else {
                        panic!("final headers must be denied before effect")
                    };
                    assert_eq!(
                        error
                            .source
                            .as_deref()
                            .and_then(|source| source.downcast_ref::<HostAllocationError>()),
                        Some(&HostAllocationError::Resource(
                            HostEffectAllocationError::ExceedsCapacity
                        ))
                    );
                    CAPTURE_EFFECTS.with(|effects| assert_eq!(effects.get(), 0));
                } else {
                    assert!(polled.is_pending());
                    CAPTURE_EFFECTS.with(|effects| assert_eq!(effects.get(), 1));
                    runmat_gc::gc_collect_major().unwrap();
                    assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    assert!(ledger.snapshot().live_reserved_bytes > 0);
                    if outcome == "complete" {
                        CAPTURE_READY.with(|ready| ready.set(true));
                        let Poll::Ready(Ok(outputs)) = future.as_mut().poll(&mut context) else {
                            panic!("complete selected body")
                        };
                        assert!(outputs.value().is_empty());
                    }
                }
                drop(future);
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_err());
                assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
            });
        }
    }
}

impl RuntimeBuiltinService for FailingBuiltin {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        let Some(entry) = runmat_builtins::builtin_catalog_primary_entry_by_name(name) else {
            return Vec::new();
        };
        vec![crate::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            |_| {
                CALLS.with(|calls| calls.set(calls.get() + 1));
                let diagnostic = DIAGNOSTIC.with(Cell::get);
                crate::builtin::rejected_builtin_future(
                    crate::build_runtime_error("the selected callee failed")
                        .with_identifier(diagnostic)
                        .build(),
                )
            },
        )]
    }
}

#[test]
fn selected_named_fallback_error_is_not_absence_or_replayed() {
    struct NoSemanticFunctions;
    impl FunctionNameResolver for NoSemanticFunctions {
        fn name_resolution_completeness(&self) -> NameResolutionCompleteness {
            NameResolutionCompleteness::Complete
        }

        fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
            None
        }
    }

    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(FailingBuiltin)));
    let source = crate::source_context::lease_source_context_in_context(
        &runtime,
        "selected-numeric-expression",
        None,
        "logical(1)",
    )
    .expect("issued selected source");
    let _entry = runtime.enter();
    for identifier in ["RunMat:UndefinedFunction", "RunMat:ProbeReceiverInvalid"] {
        CALLS.with(|calls| calls.set(0));
        DIAGNOSTIC.with(|diagnostic| diagnostic.set(identifier));
        let mut descriptor = CallableDescriptor::resolved(
            CallableIdentity::DynamicName(runmat_types::SymbolName("logical".into())),
            vec![Value::Num(1.0)],
            1,
            CallableFallbackPolicy::RuntimeNameResolution,
            CallableCallKind::Direct,
        );
        descriptor.metadata.source_id = Some(source.source_id());
        descriptor.metadata.source_ref = Some(source.source_ref());
        descriptor.metadata.source_invocation =
            Some(crate::context::RuntimeInvocationArity::Script);
        let err = futures::executor::block_on(try_execute_callable_descriptor_in_context(
            &runtime, descriptor,
        ))
        .expect_err("the selected builtin error must propagate");
        assert_eq!(err.identifier(), Some(identifier));
        assert!(err
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<MissingCallableTarget>())
            .is_none());
        assert_eq!(
            err.failure_origin(),
            crate::runtime_error::RuntimeFailureOrigin::Unclassified
        );
        CALLS.with(|calls| assert_eq!(calls.get(), 1));

        for target in [
            Value::function_handle("logical".into()),
            Value::ExternalFunctionHandle("logical".into()),
            Value::String("logical".into()),
            Value::CharArray(runmat_value::CharArray::new_row("logical")),
        ] {
            let original_name = match &target {
                Value::String(text) => text.as_str().map(str::as_ptr),
                Value::FunctionHandle(name) | Value::ExternalFunctionHandle(name) => {
                    Some(name.name().as_ptr())
                }
                _ => None, // Char refinement owns its admitted text conversion.
            };
            let callback = RepeatableCallable::select(&runtime, target, &NoSemanticFunctions)
                .expect("select the named builtin callback once");
            let CallableTarget::RetainedBuiltin(retained) = callback.selected_target() else {
                panic!("known builtin must freeze its borrowed-name dispatch");
            };
            let selected_pointer = super::repeatable::retained_target_name(retained.value())
                .unwrap()
                .as_ptr();
            if let Some(original_name) = original_name {
                assert_eq!(selected_pointer, original_name);
            }
            let cloned = callback.selected_descriptor().clone();
            let CallableTarget::RetainedBuiltin(alias) = &cloned.target else {
                panic!("same selected target")
            };
            assert_eq!(
                super::repeatable::retained_target_name(alias.value())
                    .unwrap()
                    .as_ptr(),
                selected_pointer
            );
            // A subsequent real name publication must not retarget this selected
            // builtin to a user function. The old forwarded path would enter it.
            let _replacement = crate::user_functions::test_support::install_exact_named_forwarders(
                &runtime,
                &[("logical", 29091)],
            )
            .unwrap();
            let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
                std::sync::Arc::new(|_, _, _| {
                    panic!("a retained builtin repeated semantic name resolution")
                }),
            ))
            .unwrap();
            for requested in [0, 1] {
                let expected_calls = CALLS.with(Cell::get) + 1;
                let arguments =
                    crate::execution::RootedValues::in_context(vec![Value::Num(1.)], &runtime)
                        .unwrap();
                let error = futures::executor::block_on(
                    callback.invoke_rooted(&runtime, arguments, requested),
                )
                .expect_err("original builtin failure, not a redefined user callback");
                assert_eq!(error.identifier(), Some(identifier));
                CALLS.with(|calls| assert_eq!(calls.get(), expected_calls));
            }
        }
    }
}

#[test]
fn absent_named_callable_is_a_producer_classified_language_failure() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let _scope = runtime.enter();
    let source = crate::source_context::lease_source_context_in_context(
        &runtime,
        "numeric-expression",
        None,
        "missing_numeric_function(2)",
    )
    .expect("issued source");
    let error = futures::executor::block_on(forward_named_fallback(
        Some(&runtime),
        "__runmat_missing_input_callable__".into(),
        crate::execution::RootedValues::new(Vec::new()).expect("empty rooted arguments"),
        1,
        None,
        Some(source.source_ref()),
        Some(crate::context::RuntimeInvocationArity::Script),
    ))
    .expect_err("missing callable");
    assert_eq!(
        error.failure_origin(),
        crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation
    );
    assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
    let missing = error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<MissingCallableTarget>())
        .expect("only absent selection carries the typed source cause");
    assert!(missing.in_source_script(source.source_ref()));
    let other_runtime =
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let other_source = crate::source_context::lease_source_context_in_context(
        &other_runtime,
        "other-numeric-expression",
        None,
        "missing_numeric_function(2)",
    )
    .expect("separate issuer");
    assert_eq!(source.source_id(), other_source.source_id());
    assert!(!missing.in_source_script(other_source.source_ref()));
    let nested = MissingCallableTarget {
        source: source.source_ref(),
        invocation_arity: crate::context::RuntimeInvocationArity::function(1, 1),
    };
    assert!(!nested.in_source_script(source.source_ref()));
}

#[test]
fn unselected_named_fallback_is_absent_without_invocation() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(FailingBuiltin)));
    CALLS.with(|calls| calls.set(0));
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::DynamicName(runmat_types::SymbolName(
            "AbsentMethodResolutionFixture".into(),
        )),
        vec![Value::Num(1.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let result = futures::executor::block_on(try_execute_callable_descriptor_in_context(
        &runtime, descriptor,
    ))
    .expect("a missing candidate is not a callee error");
    assert!(result.is_none());
    CALLS.with(|calls| assert_eq!(calls.get(), 0));
}

#[test]
fn resolved_family_selection_is_cold_until_the_selected_future_is_polled() {
    let calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&calls);
    let _invoker = crate::user_functions::install_local_semantic_function_invoker(Rc::new(
        move |_runtime, function, _, _| {
            assert_eq!(function, 41);
            observed.set(observed.get() + 1);
            crate::user_functions::admit_user_function_with(_runtime.as_ref(), move || async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(41.0))
            })
        },
    ))
    .expect("call-port scope");
    let descriptor = || {
        CallableDescriptor::resolved(
            CallableIdentity::BoundFunction(runmat_types::FunctionId(41)),
            Vec::new(),
            1,
            CallableFallbackPolicy::None,
            CallableCallKind::Direct,
        )
    };

    let unpolled = super::execution::execute(None, descriptor());
    assert_eq!(calls.get(), 0);
    drop(unpolled);
    assert_eq!(calls.get(), 0);

    let result = futures::executor::block_on(super::execution::execute(None, descriptor()))
        .expect("selected semantic function");
    assert_eq!(result.as_slice(), &[Value::Num(41.0)]);
    assert_eq!(calls.get(), 1);

    struct ColdRepeatable;
    impl FunctionNameResolver for ColdRepeatable {
        fn resolve_function(&self, _: &str) -> Option<FunctionId> {
            Some(FunctionId(41))
        }
    }
    let selected = RepeatableCallable::select_rooted(
        None,
        crate::execution::RootedValue::in_optional_context(
            Value::function_handle("ColdRepeatable".into()),
            None,
        )
        .unwrap(),
        &ColdRepeatable,
    )
    .unwrap();
    let cold = crate::builtins::math::optim::common::callback_source::Callable::Selected(&selected)
        .invoke(None, Vec::<Value>::new(), 1);
    assert_eq!(calls.get(), 1);
    drop(cold);
    assert_eq!(calls.get(), 1);
    for outputs in [0, 1] {
        let expected = calls.get() + 1;
        let result = futures::executor::block_on(
            crate::builtins::math::optim::common::callback_source::Callable::Selected(&selected)
                .invoke(None, Vec::<Value>::new(), outputs),
        )
        .expect("selected callback after lazy preparation");
        if outputs == 0 {
            assert!(result.as_slice().is_empty());
        } else {
            assert_eq!(result.as_slice(), &[Value::Num(41.)]);
        }
        assert_eq!(calls.get(), expected);
    }
}
