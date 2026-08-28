#![cfg(target_arch = "wasm32")]

use js_sys::{Object, Reflect};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

#[path = "support/browser_parallel.rs"]
mod browser_parallel;
use browser_parallel::{execution_host, source_request};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn browser_scheduler_preserves_parallel_failure_identity_and_source() {
    let (host, _closures) = execution_host();
    let options = Object::new();
    Reflect::set(&options, &JsValue::from_str("executionHost"), &host).unwrap();
    Reflect::set(&options, &JsValue::from_str("enableGpu"), &JsValue::FALSE).unwrap();
    let runtime = runmat_wasm::init_runmat(options.into()).await.unwrap();
    let source = r#"input = ones(1, 4);
values = zeros(1, 4);
parfor (index = 1:4, 2)
  values(index) = input(5);
end
"#;
    let result = runtime
        .execute_request_js(source_request("browser-parallel-failure.m", source, 0))
        .await
        .expect("execution returns a structured run result");
    let failure =
        Reflect::get(&result, &JsValue::from_str("error")).expect("run result has an error field");
    let identifier = Reflect::get(&failure, &JsValue::from_str("identifier"))
        .unwrap()
        .as_string();
    let message = Reflect::get(&failure, &JsValue::from_str("message"))
        .unwrap()
        .as_string()
        .unwrap_or_default();
    let span = Reflect::get(&failure, &JsValue::from_str("span")).unwrap();
    let line = Reflect::get(&span, &JsValue::from_str("line"))
        .unwrap()
        .as_f64();
    assert_eq!(
        identifier.as_deref(),
        Some("RunMat:IndexOutOfBounds"),
        "browser parallel failure message: {message}"
    );
    assert_eq!(line, Some(4.0));
}
