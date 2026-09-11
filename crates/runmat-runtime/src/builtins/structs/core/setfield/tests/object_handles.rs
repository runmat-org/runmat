use super::*;

#[test]
fn setfield_object_extension_rejects_before_handle_mutation() {
    let mut object = ObjectInstance::new("StrictHandle");
    object.properties.insert("value".into(), Value::Num(1.0));
    let target = gc_allocate(Value::Object(object)).expect("handle target");
    let handle = HandleRef {
        class_name: "StrictHandle".into(),
        target,
        valid: true,
    };
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let error = run_setfield(
        Value::HandleObject(handle),
        vec![Value::from("value"), Value::Num(2.0)],
    )
    .expect_err("object extension must be rejected");
    assert_eq!(
        error.identifier(),
        SETFIELD_OBJECT_FAMILY_EXTENSION.error_identifier
    );
    let Value::Object(object) = runmat_gc::gc_clone_value(&target).expect("unchanged target")
    else {
        panic!("expected object target");
    };
    assert_eq!(object.properties.get("value"), Some(&Value::Num(1.0)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn setfield_updates_handle_target() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut inner = ObjectInstance::new("PointHandle".to_string());
    inner.properties.insert("x".to_string(), Value::Num(0.0));
    let gc_ptr = gc_allocate(Value::Object(inner)).expect("gc allocation");
    let handle_ptr = gc_ptr;
    let handle = HandleRef {
        class_name: "PointHandle".into(),
        target: handle_ptr,
        valid: true,
    };

    let updated = run_setfield(
        Value::HandleObject(handle.clone()),
        vec![Value::from("x"), Value::Num(7.0)],
    )
    .expect("setfield handle update");

    match updated {
        Value::HandleObject(h) => assert!(crate::is_handle_valid(&h)),
        other => panic!("expected handle, got {other:?}"),
    }

    let pointee = runmat_gc::gc_clone_value(&gc_ptr).expect("valid handle target");
    match pointee {
        Value::Object(obj) => {
            assert_eq!(obj.properties.get("x"), Some(&Value::Num(7.0)));
        }
        other => panic!("expected object pointee, got {other:?}"),
    }
}
