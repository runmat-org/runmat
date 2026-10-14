use super::*;

#[test]
fn java_control_decoder_accepts_only_nonempty_text_scalar_shapes() {
    assert_eq!(
        string_argument(Some(Value::String("java.lang.String".into())), "class").unwrap(),
        "java.lang.String"
    );
    let text = "java.lang.String";
    let trailing =
        CharArray::new_with_shape(text.chars().collect(), vec![1, text.chars().count(), 1])
            .unwrap();
    assert_eq!(
        string_argument(Some(Value::CharArray(trailing)), "class").unwrap(),
        text
    );
    let pages = CharArray::new_with_shape(text.chars().collect(), vec![1, 1, text.chars().count()])
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
            .argument_to_java(&context, first)
            .expect("first callback")
        else {
            panic!("callback token");
        };
        let JavaValue::Callback(cloned_token) = adapter
            .argument_to_java(&context, cloned)
            .expect("cloned callback")
        else {
            panic!("callback token");
        };
        let JavaValue::Callback(new_token) = adapter
            .argument_to_java(&context, newly_created_equal)
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
            .argument_to_java(&context, workspace)
            .expect("workspace form")
        else {
            panic!("callback token");
        };
        let JavaValue::Callback(external_token) = adapter
            .argument_to_java(&context, external)
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
