use super::*;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn file_read_capability_requires_an_installed_read_hook() {
    let bindings = js_sys::Object::new();
    assert!(!JsFsFuncs::new(&bindings.clone().into())
        .unwrap()
        .supports_read());
    let read = Function::new_no_args("return new Uint8Array([1])");
    Reflect::set(&bindings, &JsValue::from_str("readFile"), &read).unwrap();
    assert!(JsFsFuncs::new(&bindings.into()).unwrap().supports_read());
}

#[wasm_bindgen_test]
async fn missing_replace_hook_never_calls_legacy_rename_or_remove() {
    let bindings = js_sys::Object::new();
    let forbidden = Function::new_no_args("throw new Error('legacy mutation must not run')");
    Reflect::set(&bindings, &JsValue::from_str("rename"), &forbidden).unwrap();
    Reflect::set(&bindings, &JsValue::from_str("removeFile"), &forbidden).unwrap();
    let funcs = JsFsFuncs::new(&bindings.into()).unwrap();
    let error = funcs
        .rename_replace_async(Path::new("/source"), Path::new("/target"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[wasm_bindgen_test]
async fn explicit_replace_hook_receives_both_paths_and_propagates_rejection() {
    let bindings = js_sys::Object::new();
    let hook = Function::new_with_args("from,to", "this.source=from; this.target=to; return Promise.reject(new Error('replacement rejected'));");
    Reflect::set(&bindings, &JsValue::from_str("renameReplace"), &hook).unwrap();
    let funcs = JsFsFuncs::new(&bindings.clone().into()).unwrap();
    assert!(funcs
        .rename_replace_async(Path::new("/source"), Path::new("/target"))
        .await
        .is_err());
    assert_eq!(
        Reflect::get(&bindings, &JsValue::from_str("source"))
            .unwrap()
            .as_string()
            .as_deref(),
        Some("/source")
    );
    assert_eq!(
        Reflect::get(&bindings, &JsValue::from_str("target"))
            .unwrap()
            .as_string()
            .as_deref(),
        Some("/target")
    );
}
