use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_getfield_dependent_getter_uses_selected_session() {
    let wrong = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    wrong.set_runmat_extensions_enabled(false);
    let _wrong_scope = wrong.enter();
    let selected = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    selected.set_runmat_extensions_enabled(true);
    let expected_session = selected.session_lease();
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
