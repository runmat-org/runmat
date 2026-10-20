use super::super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_getfield_output_admission_uses_selected_session_not_enclosing_session() {
    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _wrong_scope = wrong.enter();
    let _wrong_outputs =
        crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let selected_outputs = {
        let _selected_scope = selected.enter();
        crate::output_count::push_output_count(Some(1)).expect("selected output-count scope")
    };
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    value.fields.insert("p".to_string(), Value::Num(7.0));
    let value = value.finish();
    let result = futures::executor::block_on(getfield_builtin(
        Some(selected.clone()),
        Value::Struct(value.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("p")]),
    ))
    .expect("enclosing two-output request must not reject selected one-output call");
    assert_eq!(result.value(), &Value::Num(7.0));
    assert_eq!(crate::output_count::current_output_count(), Some(2));

    drop(selected_outputs);
    let _wrong_one_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let _selected_two_outputs = {
        let _selected_scope = selected.enter();
        crate::output_count::push_output_count(Some(2)).expect("selected output-count scope")
    };
    let error = futures::executor::block_on(getfield_builtin(
        Some(selected),
        Value::Struct(value),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("p")]),
    ))
    .expect_err("selected two-output request must be rejected before field access");
    assert_eq!(
        error.identifier(),
        runmat_builtins::GETFIELD_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    assert_eq!(crate::output_count::current_output_count(), Some(1));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_getfield_dependent_getter_uses_selected_session() {
    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    wrong.set_runmat_extensions_enabled(false);
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    selected.set_runmat_extensions_enabled(true);
    let expected_session = selected.session_lease();
    let selected_scope = selected.enter();
    let class = runmat_types::ClassIdentity::from("SelectedGetfieldSession");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: class.clone(),
        parent: None,
        properties: std::collections::HashMap::from([(
            runmat_types::MemberName::from("p"),
            crate::class_registry::RuntimeProperty {
                name: "p".into(),
                is_static: false,
                is_constant: false,
                is_dependent: true,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        )]),
        methods: std::collections::HashMap::from([(
            runmat_types::MethodName::from("get.p"),
            crate::class_registry::RuntimeMethod {
                name: "get.p".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: "SelectedGetfieldSession.get.p".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]),
    });
    let _resolver = crate::user_functions::install_semantic_function_resolver(Some(
        std::sync::Arc::new(|name| (name == "SelectedGetfieldSession.get.p").then_some(942)),
    ))
    .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
        std::sync::Arc::new(move |function, args, requested_outputs| {
            assert_eq!(function, 942);
            assert_eq!(requested_outputs, 1);
            assert!(matches!(args, [Value::Object(_)]));
            let active = crate::context::legacy::active().expect("selected getter scope");
            assert!(std::sync::Weak::ptr_eq(
                &active.session_lease(),
                &expected_session
            ));
            Box::pin(async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(29.0))
            })
        }),
    ))
    .expect("call-port scope");
    drop(selected_scope);
    let _wrong_scope = wrong.enter();
    let result = futures::executor::block_on(getfield_builtin(
        Some(selected),
        Value::Object(ObjectInstance::new(class)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("p")]),
    ))
    .expect("selected public getfield getter");
    assert_eq!(result.value(), &Value::Num(29.0));
    assert!(!wrong.runmat_extensions_enabled());
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("wrong enclosing session restored")
            .session_lease(),
        &wrong.session_lease()
    ));
}
