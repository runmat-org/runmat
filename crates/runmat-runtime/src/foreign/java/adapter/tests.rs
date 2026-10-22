use super::*;

#[test]
fn java_control_decoder_accepts_only_nonempty_text_scalar_shapes() {
    assert_eq!(
        string_argument(Some(Value::String("java.lang.String".into())), "class").unwrap(),
        "java.lang.String"
    );
    let text = "java.lang.String";
    let trailing = CharArray::from_unicode_scalars_with_shape(
        text.chars().collect(),
        vec![1, text.chars().count(), 1],
    )
    .unwrap();
    assert_eq!(
        string_argument(Some(Value::CharArray(trailing)), "class").unwrap(),
        text
    );
    let pages = CharArray::from_unicode_scalars_with_shape(
        text.chars().collect(),
        vec![1, 1, text.chars().count()],
    )
    .unwrap();
    for invalid in [
        Value::Num(7.0),
        Value::Bool(true),
        Value::String(" ".into()),
        Value::CharArray(pages),
    ] {
        assert!(string_argument(Some(invalid), "class").is_err());
    }
}

#[test]
fn java_feature_identity_is_shared_with_catalog() {
    assert_eq!(
        runmat_builtins::JavaFeature::parse("JVM"),
        Some(runmat_builtins::JavaFeature::Jvm)
    );
    assert_eq!(
        runmat_builtins::JavaFeature::parse("aWt"),
        Some(runmat_builtins::JavaFeature::Awt)
    );
    assert_eq!(runmat_builtins::JavaFeature::parse("other"), None);
}

#[test]
fn callback_proxy_reuses_only_the_same_callable_instance() {
    runmat_gc::gc_test_context(|| {
        let foreign = Rc::new(crate::foreign::ForeignRuntime::new(
            crate::foreign::ForeignPlatform::Native,
        ));
        let adapter = JavaAdapter::new(foreign.handles().clone()).expect("Java adapter");
        foreign
            .register_adapter(adapter.clone())
            .expect("registered Java adapter");
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    crate::context::RuntimeServicePorts::default().with_foreign(foreign),
                );
        let first = Value::function_handle("listener".into());
        let cloned = first.clone();
        let newly_created_equal = Value::function_handle("listener".into());
        assert_eq!(first, newly_created_equal, "language equality is unchanged");
        assert_ne!(
            first.callable_instance_key(),
            newly_created_equal.callable_instance_key()
        );
        let JavaValue::Callback(first_token) = adapter
            .argument_to_java(&context, first.into())
            .expect("first callback")
        else {
            panic!("callback token");
        };
        let JavaValue::Callback(cloned_token) = adapter
            .argument_to_java(&context, cloned.into())
            .expect("cloned callback")
        else {
            panic!("callback token");
        };
        let JavaValue::Callback(new_token) = adapter
            .argument_to_java(&context, newly_created_equal.into())
            .expect("new callback")
        else {
            panic!("callback token");
        };
        assert_eq!(first_token, cloned_token);
        assert_ne!(first_token, new_token);
        let named = runmat_value::NamedFunctionHandle::new("listener".into());
        let workspace = Value::FunctionHandle(named.clone());
        let external = Value::ExternalFunctionHandle(named);
        assert_ne!(
            workspace.callable_instance_key(),
            external.callable_instance_key()
        );
        let JavaValue::Callback(workspace_token) = adapter
            .argument_to_java(&context, workspace.into())
            .expect("workspace form")
        else {
            panic!("callback token");
        };
        let JavaValue::Callback(external_token) = adapter
            .argument_to_java(&context, external.into())
            .expect("external form")
        else {
            panic!("callback token");
        };
        assert_ne!(workspace_token, external_token);
        assert!(
            !adapter.is_running(),
            "identity admission must not start the JVM"
        );
    });
}

#[test]
fn missing_qualified_target_leaves_callback_unregistered() {
    let Some(home) = std::env::var_os("RUNMAT_TEST_JAVA_HOME") else {
        return;
    };
    runmat_gc::gc_test_context(|| {
        let foreign = Rc::new(crate::foreign::ForeignRuntime::new(
            crate::foreign::ForeignPlatform::Native,
        ));
        let adapter = JavaAdapter::with_configuration(
            foreign.handles().clone(),
            JavaDiscoveryRequest {
                explicit_home: Some(home.into()),
                search_system: false,
                ..JavaDiscoveryRequest::default()
            },
            JvmConfig::default(),
        )
        .expect("Java adapter");
        foreign
            .register_adapter(adapter.clone())
            .expect("registered Java adapter");
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    crate::context::RuntimeServicePorts::default().with_foreign(foreign),
                );
        let callback = Value::function_handle("callback".into());
        let arguments = vec![
            Value::String("no.such.RunMatMissingClass.callback".into()),
            callback,
        ];
        let original_backing = arguments.as_ptr();
        let completion = futures::executor::block_on(adapter.clone().invoke_in_process(
            context,
            ForeignCall {
                adapter: JAVA_ADAPTER_ID.into(),
                symbol: "invoke_qualified".into(),
                arguments: crate::call::RootedFunctionArguments::ordinary(
                    crate::execution::RootedValues::new(arguments).unwrap(),
                ),
                requested_outputs: 1,
            },
        ));
        let call = match completion.into_owner_return() {
            Err(call) => call,
            Ok(_) => panic!("missing class must decline before callback marshalling"),
        };
        assert_eq!(call.arguments.supplied().as_ptr(), original_backing);
        assert_eq!(call.arguments.len(), 2);
        assert!(adapter.callbacks.borrow().is_empty());
    });
}
