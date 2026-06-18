#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use runmat_builtins::Value;

async fn handle_identity_and_delete_impl() {
    let h1 = runmat_runtime::call_builtin_async(
        "new_handle_object",
        &[Value::String("Point".to_string())],
    )
    .await
    .unwrap();
    let h2 = runmat_runtime::call_builtin_async(
        "new_handle_object",
        &[Value::String("Point".to_string())],
    )
    .await
    .unwrap();
    let e = runmat_runtime::call_builtin_async("eq", &[h1.clone(), h2.clone()])
        .await
        .unwrap();
    if let Value::Num(n) = e {
        assert_eq!(n, 0.0);
    } else {
        panic!();
    }
    let e2 = runmat_runtime::call_builtin_async("eq", &[h1.clone(), h1.clone()])
        .await
        .unwrap();
    if let Value::Num(n) = e2 {
        assert_eq!(n, 1.0);
    } else {
        panic!();
    }
    let v = runmat_runtime::call_builtin_async("isvalid", std::slice::from_ref(&h1))
        .await
        .unwrap();
    if let Value::Bool(b) = v {
        assert!(b);
    } else {
        panic!();
    }
    let d = runmat_runtime::call_builtin_async("delete", std::slice::from_ref(&h1))
        .await
        .unwrap();
    let v2 = runmat_runtime::call_builtin_async("isvalid", &[d])
        .await
        .unwrap();
    if let Value::Bool(b) = v2 {
        assert!(!b);
    } else {
        panic!();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn handle_identity_and_delete() {
    futures::executor::block_on(handle_identity_and_delete_impl());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn handle_identity_and_delete() {
    handle_identity_and_delete_impl().await;
}

async fn events_addlistener_notify_impl() {
    // Add a simple listener and ensure notify calls do not error
    let obj = runmat_runtime::call_builtin_async(
        "new_handle_object",
        &[Value::String("Point".to_string())],
    )
    .await
    .unwrap();
    let cb = Value::String("@isvalid".to_string()); // simple callable that accepts the handle
    let _l = runmat_runtime::call_builtin_async(
        "addlistener",
        &[obj.clone(), Value::String("moved".to_string()), cb],
    )
    .await
    .unwrap();
    let _ = runmat_runtime::call_builtin_async(
        "notify",
        &[obj.clone(), Value::String("moved".to_string())],
    )
    .await
    .unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn events_addlistener_notify() {
    futures::executor::block_on(events_addlistener_notify_impl());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn events_addlistener_notify() {
    events_addlistener_notify_impl().await;
}
