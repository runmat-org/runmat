use super::*;

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
    let _wrong_outputs = crate::output_count::push_output_count(Some(2));
    let selected_outputs = {
        let _selected_scope = selected.enter();
        crate::output_count::push_output_count(Some(1))
    };
    let mut value = StructValue::new();
    value.fields.insert("p".to_string(), Value::Num(7.0));
    let result = futures::executor::block_on(getfield_builtin(
        Some(selected.clone()),
        Value::Struct(value.clone()),
        vec![Value::from("p")],
    ))
    .expect("enclosing two-output request must not reject selected one-output call");
    assert_eq!(result, Value::Num(7.0));
    assert_eq!(crate::output_count::current_output_count(), Some(2));

    drop(selected_outputs);
    let _wrong_one_output = crate::output_count::push_output_count(Some(1));
    let _selected_two_outputs = {
        let _selected_scope = selected.enter();
        crate::output_count::push_output_count(Some(2))
    };
    let error = futures::executor::block_on(getfield_builtin(
        Some(selected),
        Value::Struct(value),
        vec![Value::from("p")],
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
fn pending_public_getfield_restores_enclosing_session_and_drops_getter() {
    use std::future::Future as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct DropProbe(Arc<AtomicUsize>);
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());
    selected.set_runmat_extensions_enabled(true);
    let expected_session = selected.session_lease();
    let _wrong_scope = wrong.enter();
    let _wrong_outputs = crate::output_count::push_output_count(Some(2));
    let _selected_outputs = {
        let _selected_scope = selected.enter();
        crate::output_count::push_output_count(Some(1))
    };
    let selected_setup_scope = selected.enter();
    let class = runmat_types::ClassIdentity::from("PendingSelectedGetfieldSession");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
                function_name: "PendingSelectedGetfieldSession.get.p".into(),
                implicit_class_argument: None,
            },
        )]),
    });
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "PendingSelectedGetfieldSession.get.p").then_some(943)
        })));
    let dropped = Arc::new(AtomicUsize::new(0));
    let getter_dropped = dropped.clone();
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, args, requested_outputs| {
            assert_eq!(function, 943);
            assert_eq!(requested_outputs, 1);
            assert!(matches!(args, [Value::Object(_)]));
            let active = crate::context::legacy::active().expect("selected getter poll");
            assert!(std::sync::Weak::ptr_eq(
                &active.session_lease(),
                &expected_session
            ));
            let budget = active.cpu_cooperation();
            let dropped = getter_dropped.clone();
            Box::pin(async move {
                let _probe = DropProbe(dropped);
                let error = budget
                    .await_cancellable(std::future::pending::<()>())
                    .await
                    .expect_err("pending getter ends only on cancellation");
                Err(error.into_runtime_error())
            })
        },
    )));
    drop(selected_setup_scope);
    let mut work = Box::pin(getfield_builtin(
        Some(selected.clone()),
        Value::Object(ObjectInstance::new(class.clone())),
        vec![Value::from("p")],
    ));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(work.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(crate::output_count::current_output_count(), Some(2));
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("enclosing scope restored after pending poll")
            .session_lease(),
        &wrong.session_lease()
    ));
    drop(work);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(crate::output_count::current_output_count(), Some(2));

    let mut cancelled_work = Box::pin(getfield_builtin(
        Some(selected),
        Value::Object(ObjectInstance::new(class)),
        vec![Value::from("p")],
    ));
    assert!(matches!(
        cancelled_work.as_mut().poll(&mut context),
        Poll::Pending
    ));
    assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
    let Poll::Ready(Err(error)) = cancelled_work.as_mut().poll(&mut context) else {
        panic!("selected cancelled getter must finish with a typed error")
    };
    assert_eq!(error.identifier(), "RunMat:Cancelled");
    drop(cancelled_work);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    assert_eq!(crate::output_count::current_output_count(), Some(2));
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("enclosing scope restored after cancellation")
            .session_lease(),
        &wrong.session_lease()
    ));
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
            },
        )]),
    });
    let _resolver = crate::user_functions::install_semantic_function_resolver(Some(
        std::sync::Arc::new(|name| (name == "SelectedGetfieldSession.get.p").then_some(942)),
    ));
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
                crate::sequence::ValueSequence::single(Value::Num(29.0))
                    .map_err(crate::sequence::sequence_error_to_runtime)
            })
        }),
    ));
    drop(selected_scope);
    let _wrong_scope = wrong.enter();
    let result = futures::executor::block_on(getfield_builtin(
        Some(selected),
        Value::Object(ObjectInstance::new(class)),
        vec![Value::from("p")],
    ))
    .expect("selected public getfield getter");
    assert_eq!(result, Value::Num(29.0));
    assert!(!wrong.runmat_extensions_enabled());
    assert!(std::sync::Weak::ptr_eq(
        &crate::context::legacy::active()
            .expect("wrong enclosing session restored")
            .session_lease(),
        &wrong.session_lease()
    ));
}

#[test]
fn getfield_textual_index_and_object_family_extensions_are_mode_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let mut st = StructValue::new();
    st.fields.insert(
        "values".to_string(),
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor")),
    );
    let index = CellArray::new_with_shape(
        vec![Value::CharArray(CharArray::new_row("end"))],
        vec![1, 1],
    )
    .expect("index cell");
    let err = run_getfield(
        Value::Struct(st),
        vec![Value::from("values"), Value::Cell(index)],
    )
    .expect_err("textual index extension");
    assert_eq!(
        err.identifier(),
        GETFIELD_TEXTUAL_INDEX_EXTENSION.error_identifier
    );

    let mut object = ObjectInstance::new("TestClass".to_string());
    object
        .properties
        .insert("value".to_string(), Value::Num(1.0));
    let err = run_getfield(Value::Object(object), vec![Value::from("value")])
        .expect_err("object extension");
    assert_eq!(
        err.identifier(),
        GETFIELD_OBJECT_FAMILY_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_object_property() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut obj = ObjectInstance::new("TestClass".to_string());
    obj.properties.insert("value".to_string(), Value::Num(7.0));
    let result = run_getfield(Value::Object(obj), vec![Value::from("value")]).expect("object");
    assert_eq!(result, Value::Num(7.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_dependent_property_invokes_getter() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let class_name = "runmat.unittest.GetfieldDependent";
    let mut def = crate::class_registry::RuntimeClass {
        name: class_name.into(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    };
    def.properties.insert(
        "p".into(),
        crate::class_registry::RuntimeProperty {
            name: "p".into(),
            is_static: false,
            is_constant: false,
            is_dependent: true,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: None,
        },
    );
    crate::class_registry::register_class(def);

    let mut obj = ObjectInstance::new(class_name.to_string());
    obj.properties
        .insert("p_backing".to_string(), Value::Num(42.0));

    let result = run_getfield(Value::Object(obj), vec![Value::from("p")]).expect("dependent");
    assert_eq!(result, Value::Num(42.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_inherited_dependent_property_uses_parent_metadata() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let parent_name = "runmat.unittest.GetfieldDependentParent";
    let child_name = "runmat.unittest.GetfieldDependentChild";

    let mut parent = crate::class_registry::RuntimeClass {
        name: parent_name.into(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    };
    parent.properties.insert(
        "p".into(),
        crate::class_registry::RuntimeProperty {
            name: "p".into(),
            is_static: false,
            is_constant: false,
            is_dependent: true,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: None,
        },
    );
    crate::class_registry::register_class(parent);

    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: child_name.into(),
        parent: Some(parent_name.into()),
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });

    let mut obj = ObjectInstance::new(child_name.to_string());
    obj.properties
        .insert("p_backing".to_string(), Value::Num(17.0));

    let result =
        run_getfield(Value::Object(obj), vec![Value::from("p")]).expect("inherited dependent");
    assert_eq!(result, Value::Num(17.0));
}
