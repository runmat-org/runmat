use runmat_builtins::Value;

#[test]
fn handle_identity_and_delete() {
    let h1 =
        runmat_runtime::call_builtin("new_handle_object", &[Value::String("Point".to_string())])
            .unwrap();
    let h2 =
        runmat_runtime::call_builtin("new_handle_object", &[Value::String("Point".to_string())])
            .unwrap();
    let e = runmat_runtime::call_builtin("eq", &[h1.clone(), h2.clone()]).unwrap();
    if let Value::Num(n) = e {
        assert_eq!(n, 0.0);
    } else {
        panic!();
    }
    let e2 = runmat_runtime::call_builtin("eq", &[h1.clone(), h1.clone()]).unwrap();
    if let Value::Num(n) = e2 {
        assert_eq!(n, 1.0);
    } else {
        panic!();
    }
    let v = runmat_runtime::call_builtin("isvalid", std::slice::from_ref(&h1)).unwrap();
    if let Value::Bool(b) = v {
        assert!(b);
    } else {
        panic!();
    }
    let d = runmat_runtime::call_builtin("delete", std::slice::from_ref(&h1)).unwrap();
    let v2 = runmat_runtime::call_builtin("isvalid", &[d]).unwrap();
    if let Value::Bool(b) = v2 {
        assert!(!b);
    } else {
        panic!();
    }
}

#[test]
fn events_addlistener_notify() {
    // Add a simple listener and ensure notify calls do not error
    let obj =
        runmat_runtime::call_builtin("new_handle_object", &[Value::String("Point".to_string())])
            .unwrap();
    let cb = Value::String("@isvalid".to_string()); // simple callable that accepts the handle
    let _l = runmat_runtime::call_builtin(
        "addlistener",
        &[obj.clone(), Value::String("moved".to_string()), cb],
    )
    .unwrap();
    let _ =
        runmat_runtime::call_builtin("notify", &[obj.clone(), Value::String("moved".to_string())])
            .unwrap();
}
