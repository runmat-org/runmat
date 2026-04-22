#![cfg(target_arch = "wasm32")]

use runmat_wasm::init_runmat;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn workspace_replay_import_rejects_invalid_payload() {
    let runtime = init_runmat(JsValue::NULL)
        .await
        .expect("initialize wasm runtime");
    let imported = runtime
        .import_workspace_state(&[1, 2, 3, 4])
        .expect("workspace import result");
    assert!(!imported);
}

#[wasm_bindgen_test(async)]
async fn figure_scene_import_rejects_invalid_payload() {
    let runtime = init_runmat(JsValue::NULL)
        .await
        .expect("initialize wasm runtime");
    let error = runtime
        .import_figure_scene(br#"{"schemaVersion":99,"kind":"figure-scene"}"#)
        .await
        .expect_err("figure scene import should reject invalid payload");
    let message = error.as_string().unwrap_or_default();
    assert!(message.contains("unsupported figure replay schema version"));
}
