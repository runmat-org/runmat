use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_exception_fields() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let ex = MException::new("RunMat:Test".to_string(), "failure".to_string());
    let msg =
        run_getfield(Value::MException(ex.clone()), vec![Value::from("message")]).expect("message");
    assert_eq!(msg, Value::String("failure".to_string()));
    let ident =
        run_getfield(Value::MException(ex), vec![Value::from("identifier")]).expect("identifier");
    assert_eq!(ident, Value::String("RunMat:Test".to_string()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_exception_stack_cell() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut ex = MException::new("RunMat:Test".to_string(), "failure".to_string());
    ex.stack.push("demo.m:5".to_string());
    ex.stack.push("main.m:1".to_string());
    let stack = run_getfield(Value::MException(ex), vec![Value::from("stack")]).expect("stack");
    let Value::Cell(cell) = stack else {
        panic!("expected cell array");
    };
    assert_eq!(cell.rows, 2);
    assert_eq!(cell.cols, 1);
    let first = cell.data[0].clone();
    assert_eq!(first, Value::String("demo.m:5".to_string()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_invalid_handle_errors() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let target = runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc allocate target");
    let handle = HandleRef {
        class_name: "Demo".into(),
        target,
        valid: false,
    };
    let err = error_message(
        run_getfield(Value::HandleObject(handle), vec![Value::from("x")]).unwrap_err(),
    );
    assert!(err.contains("Invalid or deleted handle object 'Demo'"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_listener_fields_resolved() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let target = runmat_gc::gc_allocate(Value::Num(7.0)).expect("gc allocate target");
    let callback = runmat_gc::gc_allocate(Value::FunctionHandle("cb".to_string()))
        .expect("gc allocate callback");
    let listener = Listener {
        id: 9,
        target,
        target_class_name: "EventTarget".into(),
        event_name: "tick".into(),
        callback,
        enabled: true,
        valid: true,
    };
    let enabled = run_getfield(
        Value::Listener(listener.clone()),
        vec![Value::from("Enabled")],
    )
    .expect("enabled");
    assert_eq!(enabled, Value::Bool(true));
    let event_name = run_getfield(
        Value::Listener(listener.clone()),
        vec![Value::from("EventName")],
    )
    .expect("event name");
    assert_eq!(event_name, Value::String("tick".to_string()));
    let callback =
        run_getfield(Value::Listener(listener), vec![Value::from("Callback")]).expect("callback");
    assert!(matches!(callback, Value::FunctionHandle(_)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn getfield_invalid_listener_rejects_rooted_fields() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let target = runmat_gc::gc_allocate(Value::Num(7.0)).expect("gc allocate target");
    let callback = runmat_gc::gc_allocate(Value::FunctionHandle("cb".to_string()))
        .expect("gc allocate callback");
    let listener = Listener {
        id: 10,
        target,
        target_class_name: "EventTarget".into(),
        event_name: "tick".into(),
        callback,
        enabled: false,
        valid: false,
    };

    let err = error_message(
        run_getfield(
            Value::Listener(listener.clone()),
            vec![Value::from("Callback")],
        )
        .unwrap_err(),
    );
    assert!(err.contains("listener is invalid or deleted"));

    let err = error_message(
        run_getfield(Value::Listener(listener), vec![Value::from("Target")]).unwrap_err(),
    );
    assert!(err.contains("listener is invalid or deleted"));
}
