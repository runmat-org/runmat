#![cfg(target_arch = "wasm32")]

use std::time::Duration;

use js_sys::Array;
use runmat_runtime::context::RuntimeHttpService;
use runmat_runtime::http::{BrowserHttpService, HttpMethod, HttpRequest};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_worker);

#[wasm_bindgen_test(async)]
async fn worker_global_scope_fetches_bounded_response_stream() {
    assert!(
        web_sys::window().is_none(),
        "worker route must not use Window"
    );
    let parts = Array::new();
    parts.push(&JsValue::from_str("RunMat worker stream"));
    let blob = web_sys::Blob::new_with_str_sequence(&parts).expect("worker blob");
    let url = web_sys::Url::create_object_url_with_blob(&blob).expect("worker blob URL");
    let request = HttpRequest {
        url: url::Url::parse(&url).expect("parse blob URL"),
        method: HttpMethod::Get,
        headers: Vec::new(),
        body: None,
        timeout: Duration::from_secs(5),
        user_agent: "RunMat worker test".into(),
    };
    let mut stream = BrowserHttpService::new()
        .open_bounded(request, 1024)
        .await
        .expect("WorkerGlobalScope response stream");
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next_chunk().await.expect("worker body chunk") {
        bytes.extend_from_slice(&chunk);
    }
    web_sys::Url::revoke_object_url(&url).expect("revoke worker blob URL");
    assert_eq!(bytes, b"RunMat worker stream");
}
