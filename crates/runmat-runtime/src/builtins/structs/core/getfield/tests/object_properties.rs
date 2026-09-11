use super::*;

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
