use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_assigns_object_property() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut class_def = crate::class_registry::RuntimeClass {
        name: "Simple".into(),
        parent: None,
        properties: Default::default(),
        methods: Default::default(),
    };
    class_def.properties.insert(
        "x".into(),
        crate::class_registry::RuntimeProperty {
            name: "x".into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: None,
        },
    );
    crate::class_registry::register_class(class_def);

    let mut obj = ObjectInstance::new("Simple".to_string());
    obj.properties.insert("x".to_string(), Value::Num(0.0));

    let updated = run_setfield(Value::Object(obj), vec![Value::from("x"), Value::Num(5.0)])
        .expect("setfield");

    match updated {
        Value::Object(o) => {
            assert_eq!(o.properties.get("x"), Some(&Value::Num(5.0)));
        }
        other => panic!("expected object result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_errors_on_static_property_assignment() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut class_def = crate::class_registry::RuntimeClass {
        name: "StaticSetfield".into(),
        parent: None,
        properties: Default::default(),
        methods: Default::default(),
    };
    class_def.properties.insert(
        "version".into(),
        crate::class_registry::RuntimeProperty {
            name: "version".into(),
            is_static: true,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: None,
        },
    );
    crate::class_registry::register_class(class_def);

    let obj = ObjectInstance::new("StaticSetfield".to_string());
    let err = error_message(
        run_setfield(
            Value::Object(obj),
            vec![Value::from("version"), Value::Num(2.0)],
        )
        .expect_err("setfield should reject static property writes"),
    );
    assert!(
        err.contains("Property 'version' is static"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_rejects_inherited_static_property_assignment() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let parent_name = "runmat.unittest.StaticSetfieldParent";
    let child_name = "runmat.unittest.StaticSetfieldChild";

    let mut parent = crate::class_registry::RuntimeClass {
        name: parent_name.into(),
        parent: None,
        properties: Default::default(),
        methods: Default::default(),
    };
    parent.properties.insert(
        "version".into(),
        crate::class_registry::RuntimeProperty {
            name: "version".into(),
            is_static: true,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: None,
        },
    );
    crate::class_registry::register_class(parent);
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: child_name.into(),
        parent: Some(parent_name.into()),
        properties: Default::default(),
        methods: Default::default(),
    });

    let obj = ObjectInstance::new(child_name.to_string());
    let err = error_message(
        run_setfield(
            Value::Object(obj),
            vec![Value::from("version"), Value::Num(2.0)],
        )
        .expect_err("setfield should reject inherited static property writes"),
    );
    assert!(
        err.contains("Property 'version' is static"),
        "unexpected error message: {err}"
    );
}
