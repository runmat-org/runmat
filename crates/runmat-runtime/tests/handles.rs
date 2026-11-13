#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use runmat_value::Value;

async fn handle_identity_and_delete_impl() {
    let h1 = runmat_runtime::call_builtin_async(
        "new_handle_object",
        &[Value::String("Point".to_string().into())],
    )
    .await
    .unwrap();
    let h2 = runmat_runtime::call_builtin_async(
        "new_handle_object",
        &[Value::String("Point".to_string().into())],
    )
    .await
    .unwrap();
    let e = runmat_runtime::call_builtin_async("eq", &[h1.value().clone(), h2.value().clone()])
        .await
        .unwrap();
    assert_eq!(e, Value::Bool(false));
    let e2 = runmat_runtime::call_builtin_async("eq", &[h1.value().clone(), h1.value().clone()])
        .await
        .unwrap();
    assert_eq!(e2, Value::Bool(true));
    let v = runmat_runtime::call_builtin_async("isvalid", std::slice::from_ref(h1.value()))
        .await
        .unwrap();
    if let Value::Bool(b) = v.value() {
        assert!(*b);
    } else {
        panic!();
    }
    let d = runmat_runtime::call_builtin_async("delete", std::slice::from_ref(h1.value()))
        .await
        .unwrap();
    let v2 = runmat_runtime::call_builtin_async("isvalid", std::slice::from_ref(d.value()))
        .await
        .unwrap();
    if let Value::Bool(b) = v2.value() {
        assert!(!*b);
    } else {
        panic!();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn handle_identity_and_delete() {
    futures::executor::block_on(handle_identity_and_delete_impl());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn handle_identity_and_delete() {
    handle_identity_and_delete_impl().await;
}

async fn events_addlistener_notify_impl() {
    use runmat_runtime::context::RuntimeContext;
    use runmat_types::{EventAttributes, EventDeclaration, MemberName};
    use std::cell::Cell;
    use std::rc::Rc;

    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let class = runmat_types::ClassIdentity::from("HandleEventDeliveryFixture");
    let event = EventDeclaration {
        name: MemberName::from("moved"),
        attributes: EventAttributes::default(),
        declared_attributes: Vec::new(),
        span: Default::default(),
    };
    {
        let _entry = runtime.enter();
        runmat_runtime::class_registry::register_checked_class(
            runmat_runtime::class_registry::RuntimeClass {
                name: class.clone(),
                parent: Some(runmat_types::standard::HANDLE.owned()),
                properties: Default::default(),
                methods: Default::default(),
                events: [(event.name.clone(), event)].into(),
            },
            false,
            false,
            Vec::new(),
        )
        .expect("authored public moved event on a handle subclass");
    }
    let obj = runmat_runtime::call_builtin_async_in_context(
        &runtime,
        "new_handle_object",
        &[Value::String(class.to_string().into())],
    )
    .await
    .unwrap();
    let Value::HandleObject(source) = obj.value() else {
        panic!("handle source");
    };
    let source_target = source.target;
    let binding = runmat_value::BoundFunctionBinding::portable(
        73,
        runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("handle event delivery callback"),
            function: runmat_types::ProgramFunctionId(73),
        },
    );
    let expected = binding.clone();
    let delivered = Rc::new(Cell::new(0));
    let observed = delivered.clone();
    let _invoker = {
        let _entry = runtime.enter();
        runmat_runtime::user_functions::install_local_bound_function_invoker(Rc::new(
            move |runtime, actual, arguments, requested_outputs, _lexical_environment| {
                assert_eq!(actual, expected, "exact callback definition owner");
                assert_eq!(requested_outputs, 0);
                assert_eq!(arguments.len(), 2, "source plus EventData");
                assert!(matches!(arguments.get(0), Some(Value::HandleObject(handle))
                    if handle.target == source_target));
                let Some(Value::HandleObject(data)) = arguments.get(1) else {
                    panic!("default notification EventData");
                };
                assert_eq!(data.class_name, runmat_types::standard::EVENT_DATA);
                runmat_gc::gc_with_value(&data.target, |value| {
                    let Value::Object(data) = value else {
                        panic!("EventData object");
                    };
                    assert!(
                        matches!(data.property("Source"), Some(Value::HandleObject(handle))
                        if handle.target == source_target)
                    );
                    assert_eq!(
                        data.property("EventName"),
                        Some(&Value::CharArray(runmat_value::CharArray::new_row("moved")))
                    );
                })
                .unwrap();
                let observed = observed.clone();
                runmat_runtime::execution::admit_executable_sequence_with(
                    &runtime.clone(),
                    move || async move {
                        let _arguments = arguments;
                        observed.set(observed.get() + 1);
                        runmat_runtime::execution::RootedValueSequence::new(
                            runmat_value::ValueSequence::empty(),
                        )
                        .map_err(|error| error.into_runtime_error())
                    },
                )
            },
        ))
    };
    let callback = Value::bound_function_handle(
        "handle_event_callback".into(),
        binding.function,
        binding.authority,
        runmat_types::FunctionSourceRole::Unknown,
        None,
    );
    let unknown = runmat_runtime::call_builtin_async_in_context(
        &runtime,
        "addlistener",
        &[
            obj.value().clone(),
            Value::String("missing".into()),
            callback.clone(),
        ],
    )
    .await
    .expect_err("undeclared event still rejects");
    assert_eq!(unknown.identifier(), Some("RunMat:AddListenerEventUnknown"));
    assert_eq!(delivered.get(), 0);
    let _listener = runmat_runtime::call_builtin_async_in_context(
        &runtime,
        "addlistener",
        &[obj.value().clone(), Value::String("moved".into()), callback],
    )
    .await
    .unwrap();
    let status = runmat_runtime::call_builtin_async_in_context(
        &runtime,
        "notify",
        &[obj.value().clone(), Value::String("moved".into())],
    )
    .await
    .expect("actual event callback delivery");
    assert_eq!(status.value(), &Value::Num(0.0));
    assert_eq!(delivered.get(), 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn events_addlistener_notify() {
    futures::executor::block_on(events_addlistener_notify_impl());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn events_addlistener_notify() {
    events_addlistener_notify_impl().await;
}
