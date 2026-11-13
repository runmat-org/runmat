use super::*;
use futures::executor::block_on;
use runmat_types::{standard, EventAccess, EventAttributes, EventDeclaration, MemberName};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

async fn notify_builtin(
    target: Value,
    event_name: String,
    arguments: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    // Explicit test ingress; production forwards its injected invocation context.
    let runtime = crate::context::legacy::active();
    super::notify_builtin(
        runtime.as_ref(),
        target,
        event_name,
        crate::call::BuiltinArguments::from_borrowed(&arguments, runtime.as_ref())?,
    )
    .await
}

static EVENT_ADMISSION_TEST_LOCK: Mutex<()> = Mutex::new(());

fn in_event_test(test: impl FnOnce()) {
    let _guard = EVENT_ADMISSION_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    reset_event_registry_for_test();
    test();
    reset_event_registry_for_test();
}

fn register_source(name: &str, access: EventAccess) -> runmat_types::ClassIdentity {
    let identity = runmat_types::ClassIdentity::from(name);
    let event = EventDeclaration {
        name: MemberName::from("Changed"),
        attributes: EventAttributes {
            listen_access: access,
            ..EventAttributes::default()
        },
        declared_attributes: Vec::new(),
        span: Default::default(),
    };
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: identity.clone(),
        parent: Some(standard::HANDLE.owned()),
        properties: HashMap::new(),
        methods: HashMap::new(),
        events: HashMap::from([(event.name.clone(), event)]),
    });
    identity
}

#[test]
fn unknown_and_inaccessible_events_reject_before_listener_publication() {
    in_event_test(|| {
        let owner = register_source("EventAdmissionNoEffectSource", EventAccess::Private);
        let target = block_on(crate::new_handle_object_builtin(owner.to_string())).unwrap();
        let callback = Value::FunctionHandle("sin".into());
        let unknown = block_on(addlistener_builtin(
            target.clone(),
            "Missing".into(),
            callback.clone(),
        ))
        .unwrap_err();
        assert_eq!(unknown.identifier(), Some("RunMat:AddListenerEventUnknown"));
        let inaccessible = block_on(addlistener_builtin(
            target.clone(),
            "Changed".into(),
            callback.clone(),
        ))
        .unwrap_err();
        assert_eq!(
            inaccessible.identifier(),
            Some("RunMat:AddListenerEventAccess")
        );
        assert!(with_event_registry(|registry| registry
            .listeners
            .is_empty()));
        assert!(with_event_registry(|registry| registry
            .listener_roots
            .is_empty()));
        let unknown_notify =
            block_on(notify_builtin(target.clone(), "Missing".into(), vec![])).unwrap_err();
        assert_eq!(
            unknown_notify.identifier(),
            Some("RunMat:NotifyEventUnknown")
        );
        let _scope = crate::dispatcher::push_class_access_context(Some(owner)).unwrap();
        block_on(addlistener_builtin(target, "Changed".into(), callback)).unwrap();
        assert_eq!(
            with_event_registry(|registry| registry.listener_roots.len()),
            1
        );
    });
}

#[test]
fn default_and_subclass_event_data_reach_callbacks_as_second_argument() {
    in_event_test(|| {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 1461);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, outputs| {
                assert_eq!(function, 1461);
                assert_eq!(outputs, 0);
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Value::HandleObject(_)));
                assert!(crate::class_registry::is_event_data_value(&args[1]));
                let Value::HandleObject(data) = &args[1] else {
                    unreachable!()
                };
                runmat_gc::gc_with_value(&data.target, |value| {
                    let Value::Object(object) = value else {
                        unreachable!()
                    };
                    assert!(matches!(
                        object.properties().get("Source"),
                        Some(Value::HandleObject(_))
                    ));
                    assert_eq!(
                        object.properties().get("EventName"),
                        Some(&Value::CharArray(runmat_value::CharArray::new_row(
                            "Changed"
                        )))
                    );
                })
                .expect("event data target");
                observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        0.0,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        let source = register_source("EventAdmissionPayloadSource", EventAccess::Public);
        let target = block_on(crate::new_handle_object_builtin(source.to_string())).unwrap();
        block_on(addlistener_builtin(
            target.clone(),
            "Changed".into(),
            crate::user_functions::test_support::exact_local_value("event_callback", binding),
        ))
        .unwrap();
        block_on(notify_builtin(target.clone(), "Changed".into(), vec![])).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let invalid = block_on(notify_builtin(
            target.clone(),
            "Changed".into(),
            vec![Value::Num(1.0)],
        ))
        .unwrap_err();
        assert_eq!(invalid.identifier(), Some("RunMat:NotifyDataInvalid"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let derived = runmat_types::ClassIdentity::from("EventAdmissionPayloadDerived");
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: derived.clone(),
            parent: Some(standard::EVENT_DATA.owned()),
            properties: HashMap::new(),
            methods: HashMap::new(),
            events: HashMap::new(),
        });
        let custom = block_on(crate::new_handle_object_builtin(derived.to_string())).unwrap();
        block_on(notify_builtin(target, "Changed".into(), vec![custom])).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn callback_republication_stops_later_delivery_without_replay() {
    in_event_test(|| {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 1462);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let source = register_source("EventAdmissionReentrySource", EventAccess::Public);
        let changed_source = source.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, outputs| {
                assert_eq!(function, 1462);
                assert_eq!(outputs, 0);
                assert_eq!(args.len(), 2);
                observed.fetch_add(1, Ordering::SeqCst);
                crate::class_registry::register_class(crate::class_registry::RuntimeClass {
                    name: changed_source.clone(),
                    parent: Some(standard::HANDLE.owned()),
                    properties: HashMap::new(),
                    methods: HashMap::new(),
                    events: HashMap::new(),
                });
                Box::pin(async {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        0.0,
                    ))
                })
            },
        )))
        .expect("call-port scope");
        let target = block_on(crate::new_handle_object_builtin(source.to_string())).unwrap();
        let callback =
            crate::user_functions::test_support::exact_local_value("event_callback", binding);
        for _ in 0..2 {
            block_on(addlistener_builtin(
                target.clone(),
                "Changed".into(),
                callback.clone(),
            ))
            .unwrap();
        }
        let error = block_on(notify_builtin(target, "Changed".into(), vec![])).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NotifyEventUnknown"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    });
}
