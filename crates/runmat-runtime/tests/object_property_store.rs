//! One runtime's canonical object service, without the public setfield extension.
use futures::executor::block_on;
use runmat_runtime::class_registry::{
    register_class, RuntimeClass, RuntimeMethod, RuntimeProperty,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::object::resolve::store_member;
use runmat_types::MemberAccess;
use runmat_value::{HandleRef, ObjectInstance, Value, ValueSequence};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn class(name: &str) -> RuntimeClass {
    let mut class = RuntimeClass {
        events: Default::default(),
        name: name.into(),
        parent: None,
        properties: Default::default(),
        methods: Default::default(),
    };
    for name in [
        "Value",
        "Other",
        "Dependent",
        "Private",
        "Protected",
        "Constant",
        "Static",
    ] {
        class.properties.insert(
            name.into(),
            RuntimeProperty {
                name: name.into(),
                is_static: name == "Static",
                is_constant: name == "Constant",
                is_dependent: name == "Dependent",
                get_access: MemberAccess::Public,
                set_access: match name {
                    "Private" => MemberAccess::Private,
                    "Protected" => MemberAccess::Protected,
                    _ => MemberAccess::Public,
                },
                default_value: None,
            },
        );
    }
    class
}

fn setter(class: &mut RuntimeClass, function: &str) {
    class.methods.insert(
        "set.Dependent".into(),
        RuntimeMethod {
            name: "set.Dependent".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: function.into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
}

fn new_handle(name: &str) -> HandleRef {
    new_handle_configured(name, |_| {})
}
fn new_handle_configured(
    name: &str,
    configure: impl FnOnce(&mut runmat_value::UnboundedObjectBuilder),
) -> HandleRef {
    let mut object = runmat_value::UnboundedObjectBuilder::new(name);
    object.insert("Value".into(), Value::Num(1.0));
    object.insert("Other".into(), Value::Num(f64::NAN));
    configure(&mut object);
    HandleRef {
        class_name: name.into(),
        target: runmat_gc::gc_allocate(Value::Object(object.finish())).unwrap(),
        valid: true,
    }
}

fn property(handle: &HandleRef, name: &str) -> Value {
    runmat_gc::gc_with_value(&handle.target, |value| {
        let Value::Object(object) = value else {
            panic!("handle target")
        };
        object.properties()[name].clone()
    })
    .unwrap()
}

#[test]
fn strict_handle_stores_preserve_identity_nan_and_access_controls() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(runtime.scope(async {
            let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
            register_class(class("PropertyHandle"));
            let handle = new_handle("PropertyHandle");
            let result = store_member(
                Value::HandleObject(handle.clone()),
                "Value".into(),
                Value::Num(9.0),
                None,
                |_, _| {},
            )
            .await
            .unwrap();
            let Value::HandleObject(result) = result.value() else {
                panic!("same handle")
            };
            assert_eq!(result.target, handle.target);
            assert_eq!(property(&handle, "Value"), Value::Num(9.0));
            assert!(matches!(property(&handle, "Other"), Value::Num(value) if value.is_nan()));
            for (name, identifier) in [
                ("Private", "RunMat:PropertyPrivateAccess"),
                ("Protected", "RunMat:PropertyPrivateAccess"),
                ("Constant", "RunMat:PropertyReadOnly"),
                ("Static", "RunMat:PropertyStaticAccess"),
            ] {
                let error = store_member(
                    Value::HandleObject(handle.clone()),
                    name.into(),
                    Value::Num(7.0),
                    None,
                    |_, _| {},
                )
                .await
                .unwrap_err();
                assert_eq!(error.identifier(), Some(identifier), "{name}: {error:?}");
                assert_eq!(property(&handle, "Value"), Value::Num(9.0));
            }
            let error = runmat_runtime::call_builtin_async(
                "setfield",
                &[
                    Value::HandleObject(handle.clone()),
                    Value::String("Value".into()),
                    Value::Num(3.0),
                ],
            )
            .await
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SetfieldObjectFamilyExtension")
            );
            assert_eq!(property(&handle, "Value"), Value::Num(9.0));
            let mut dead = handle;
            dead.valid = false;
            assert_eq!(
                store_member(
                    Value::HandleObject(dead),
                    "Value".into(),
                    Value::Num(0.0),
                    None,
                    |_, _| {}
                )
                .await
                .unwrap_err()
                .identifier(),
                Some("RunMat:setfield:InvalidHandle")
            );
        }));
    });
}

#[test]
fn suspended_handle_setter_roots_arguments_until_cancellation() {
    use std::future::Future;
    use std::task::{Context, Poll};
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let (receiver, rhs) = block_on(runtime.scope(async {
            let _resolver =
                runmat_runtime::user_functions::install_semantic_function_resolver(Some(Arc::new(
                    |name| (name == "Pending.set.Dependent").then_some(43),
                )));
            let mut definition = class("Pending");
            setter(&mut definition, "Pending.set.Dependent");
            register_class(definition);
            (new_handle("Pending"), new_handle("Payload"))
        }));
        let receiver_target = receiver.target;
        let rhs_target = rhs.target;
        let _scope = runtime.enter();
        let _invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|_, args, requested| {
                assert!(matches!(args[0], Value::HandleObject(_)));
                assert_eq!(requested, 0);
                Box::pin(futures::future::pending())
            }),
        ));
        let mut future = Box::pin(store_member(
            Value::HandleObject(receiver),
            "Dependent".into(),
            Value::HandleObject(rhs),
            None,
            |_, _| {},
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&receiver_target).is_ok());
        assert!(runmat_gc::gc_clone_value(&rhs_target).is_ok());
        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&receiver_target).is_err());
        assert!(runmat_gc::gc_clone_value(&rhs_target).is_err());
    });
}

#[test]
fn dependent_handle_setter_receives_original_receiver_and_keeps_reentrant_changes() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(runtime.scope(async {
            let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
            let _resolver =
                runmat_runtime::user_functions::install_semantic_function_resolver(Some(Arc::new(
                    |name| (name == "PropertyHandle.set.Dependent").then_some(41),
                )));
            let mut definition = class("PropertyHandle");
            setter(&mut definition, "PropertyHandle.set.Dependent");
            register_class(definition);
            let handle = new_handle("PropertyHandle");
            let expected = handle.target;
            let payload = new_handle("Payload");
            let payload_target = payload.target;
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = calls.clone();
            let _invoker = runmat_runtime::user_functions::install_local_semantic_function_invoker(
                Rc::new(move |_runtime, function, args, requested| {
                    assert_eq!(function, 41);
                    assert_eq!(requested, 0, "handle setter has no replacement result");
                    observed.fetch_add(1, Ordering::SeqCst);
                    let receiver = args[0].clone();
                    let rhs = args[1].clone();
                    runmat_runtime::user_functions::admit_user_function_with(
                        _runtime.as_ref(),
                        move || async move {
                            let Value::HandleObject(handle) = &receiver else {
                                panic!("original handle receiver, not Object snapshot")
                            };
                            assert_eq!(handle.target, expected);
                            runmat_gc::gc_collect_major().unwrap();
                            assert!(
                                runmat_gc::gc_clone_value(&payload_target).is_ok(),
                                "setter RHS rooted during callback"
                            );
                            store_member(
                                receiver.clone(),
                                "Other".into(),
                                Value::Num(88.0),
                                None,
                                |_, _| {},
                            )
                            .await?;
                            store_member(receiver, "Value".into(), rhs, None, |_, _| {}).await?;
                            runmat_runtime::execution::RootedValueSequence::new(
                                ValueSequence::empty(),
                            )
                            .map_err(|error| error.into_runtime_error())
                        },
                    )
                }),
            );
            let result = store_member(
                Value::HandleObject(handle.clone()),
                "Dependent".into(),
                Value::HandleObject(payload),
                None,
                |_, _| {},
            )
            .await
            .unwrap();
            let Value::HandleObject(result) = result.value() else {
                panic!("handle result")
            };
            assert_eq!(result.target, expected);
            assert_eq!(property(&handle, "Other"), Value::Num(88.0));
            let Value::HandleObject(stored) = property(&handle, "Value") else {
                panic!("stored RHS")
            };
            assert_eq!(stored.target, payload_target);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let rooted = runmat_gc::gc_root(handle.target).unwrap();
            runmat_gc::gc_collect_major().unwrap();
            assert!(
                runmat_gc::gc_clone_value(&payload_target).is_ok(),
                "published child remains traced"
            );
            drop(rooted);
        }));
    });
}

#[test]
fn dependent_value_setter_still_returns_one_replacement() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    block_on(runtime.scope(async {
        let _resolver = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
            Arc::new(|name| (name == "PropertyValue.set.Dependent").then_some(42)),
        ));
        let mut definition = class("PropertyValue");
        setter(&mut definition, "PropertyValue.set.Dependent");
        register_class(definition);
        let _invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
            Arc::new(|function, args, requested| {
                assert_eq!((function, requested), (42, 1));
                let Value::Object(mut object) = args[0].clone() else {
                    panic!("value receiver")
                };
                object.replace_property_unbounded("Value".into(), args[1].clone());
                Box::pin(async move {
                    let sequence =
                        runmat_runtime::sequence::single_value_sequence(Value::Object(object))?;
                    runmat_runtime::execution::RootedValueSequence::new(sequence)
                        .map_err(|error| error.into_runtime_error())
                })
            }),
        ));
        let result = store_member(
            Value::Object(ObjectInstance::new("PropertyValue")),
            "Dependent".into(),
            Value::Num(8.0),
            None,
            |_, _| {},
        )
        .await
        .unwrap();
        let Value::Object(object) = result.value() else {
            panic!("replacement value")
        };
        assert_eq!(object.properties()["Value"], Value::Num(8.0));
    }));
}

#[test]
fn dynamic_stores_and_private_setter_admission_use_registered_metadata() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(runtime.scope(async {
            let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
            let _resolver =
                runmat_runtime::user_functions::install_semantic_function_resolver(Some(Arc::new(
                    |name| (name == "Restricted.set.Dependent").then_some(45),
                )));
            let mut definition = class("Restricted");
            setter(&mut definition, "Restricted.set.Dependent");
            definition
                .methods
                .get_mut(&runmat_types::MethodName::from("set.Dependent"))
                .unwrap()
                .access = MemberAccess::Private;
            register_class(definition);
            let _invoker = runmat_runtime::user_functions::install_semantic_function_invoker(Some(
                Arc::new(|_, _, _| panic!("private setter must not execute")),
            ));
            let handle = new_handle_configured("Restricted", |object| {
                for name in ["Dynamic", "PrivateDynamic", "ProtectedDynamic"] {
                    let mut property =
                        runmat_value::DynamicPropertyDef::new(name.into(), "Restricted");
                    property.set_access = match name {
                        "PrivateDynamic" => MemberAccess::Private,
                        "ProtectedDynamic" => MemberAccess::Protected,
                        _ => MemberAccess::Public,
                    };
                    object.insert_dynamic(name.into(), property);
                    object.insert(name.into(), Value::Num(0.0));
                }
            });
            store_member(
                Value::HandleObject(handle.clone()),
                "Dynamic".into(),
                Value::Num(12.0),
                None,
                |_, _| {},
            )
            .await
            .unwrap();
            assert_eq!(property(&handle, "Dynamic"), Value::Num(12.0));
            for name in ["PrivateDynamic", "ProtectedDynamic"] {
                let error = store_member(
                    Value::HandleObject(handle.clone()),
                    name.into(),
                    Value::Num(2.0),
                    None,
                    |_, _| {},
                )
                .await
                .unwrap_err();
                assert_eq!(error.identifier(), Some("RunMat:PropertyPrivateAccess"));
                assert_eq!(property(&handle, name), Value::Num(0.0));
                store_member(
                    Value::HandleObject(handle.clone()),
                    name.into(),
                    Value::Num(6.0),
                    Some("Restricted.set.Dependent"),
                    |_, _| {},
                )
                .await
                .unwrap();
                assert_eq!(
                    property(&handle, name),
                    Value::Num(6.0),
                    "defining-class caller is authorized"
                );
            }
            let mut child = class("RestrictedChild");
            child.parent = Some("Restricted".into());
            setter(&mut child, "RestrictedChild.set.Dependent");
            register_class(child);
            store_member(
                Value::HandleObject(handle.clone()),
                "ProtectedDynamic".into(),
                Value::Num(10.0),
                Some("RestrictedChild.set.Dependent"),
                |_, _| {},
            )
            .await
            .unwrap();
            assert_eq!(property(&handle, "ProtectedDynamic"), Value::Num(10.0));
            assert_eq!(
                store_member(
                    Value::HandleObject(handle.clone()),
                    "PrivateDynamic".into(),
                    Value::Num(10.0),
                    Some("RestrictedChild.set.Dependent"),
                    |_, _| {}
                )
                .await
                .unwrap_err()
                .identifier(),
                Some("RunMat:PropertyPrivateAccess")
            );
            let error = store_member(
                Value::HandleObject(handle.clone()),
                "Dependent".into(),
                Value::Num(3.0),
                None,
                |_, _| {},
            )
            .await
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:MethodPrivate"));
            assert!(store_member(
                Value::HandleObject(handle),
                "Unknown".into(),
                Value::Num(3.0),
                None,
                |_, _| {}
            )
            .await
            .is_err());
        }));
    });
}

#[test]
fn dependent_property_without_setter_never_creates_backing_storage() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(runtime.scope(async {
            register_class(class("ReadOnlyDependent"));
            let handle = new_handle("ReadOnlyDependent");
            for value in [
                Value::Object(ObjectInstance::new("ReadOnlyDependent")),
                Value::HandleObject(handle.clone()),
            ] {
                let error =
                    store_member(value, "Dependent".into(), Value::Num(9.0), None, |_, _| {
                        panic!("no write before no-setter rejection")
                    })
                    .await
                    .unwrap_err();
                assert_eq!(error.identifier(), Some("RunMat:PropertyReadOnly"));
            }
            runmat_gc::gc_with_value(&handle.target, |value| {
                let Value::Object(object) = value else {
                    panic!("target")
                };
                assert!(!object.properties().contains_key("Dependent"));
                assert_eq!(object.properties()["Value"], Value::Num(1.0));
            })
            .unwrap();
        }));
    });
}

#[test]
fn overloaded_assignment_runs_once_and_listener_writes_remain_unsupported() {
    runmat_gc::gc_test_context(|| {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(runtime.scope(async {
            let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
            let _resolver =
                runmat_runtime::user_functions::install_semantic_function_resolver(Some(Arc::new(
                    |name| (name == "Overloaded.subsasgn").then_some(44),
                )));
            let mut definition = class("Overloaded");
            definition.methods.insert(
                "subsasgn".into(),
                RuntimeMethod {
                    name: "subsasgn".into(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: "Overloaded.subsasgn".into(),
                    implicit_class_argument: None,
                    role: runmat_types::ClassMethodRole::Ordinary,
                },
            );
            register_class(definition);
            let handle = new_handle("Overloaded");
            let expected = handle.target;
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = calls.clone();
            let _invoker = runmat_runtime::user_functions::install_local_semantic_function_invoker(
                Rc::new(move |_runtime, function, args, requested| {
                    assert_eq!((function, requested), (44, 1));
                    observed.fetch_add(1, Ordering::SeqCst);
                    let Value::HandleObject(receiver) = &args[0] else {
                        panic!("original overloaded receiver")
                    };
                    assert_eq!(receiver.target, expected);
                    assert_eq!(args.iter().next_back(), Some(&Value::Num(7.0)));
                    let result = args[0].clone();
                    runmat_runtime::user_functions::admit_user_function_with(
                        _runtime.as_ref(),
                        move || async move {
                            let sequence = runmat_runtime::sequence::single_value_sequence(result)?;
                            runmat_runtime::execution::RootedValueSequence::new(sequence)
                                .map_err(|error| error.into_runtime_error())
                        },
                    )
                }),
            );
            let result = store_member(
                Value::HandleObject(handle.clone()),
                "NotAProperty".into(),
                Value::Num(7.0),
                None,
                |_, _| {},
            )
            .await
            .unwrap();
            let Value::HandleObject(result) = result.value() else {
                panic!("overload result")
            };
            assert_eq!(result.target, expected);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                property(&handle, "Value"),
                Value::Num(1.0),
                "default store did not run after overload"
            );
            let listener = runmat_value::Listener {
                id: 1,
                target: handle.target,
                target_class_name: handle.class_name,
                event_name: "Changed".into(),
                callback: runmat_gc::gc_allocate(Value::Num(0.0)).unwrap(),
                enabled: true,
                valid: true,
            };
            let error = store_member(
                Value::Listener(listener),
                "Enabled".into(),
                Value::Bool(false),
                None,
                |_, _| {},
            )
            .await
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:setfield:NonStructAssignment")
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }));
    });
}
