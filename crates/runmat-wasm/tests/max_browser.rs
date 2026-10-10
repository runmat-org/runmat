#![cfg(target_arch = "wasm32")]

use runmat_builtins::{
    BuiltinExampleFixture, BuiltinExampleHarness, BuiltinExampleVerification, MAX_CATALOG_ENTRY,
    MAX_ERROR_INVALID_ARGUMENT,
};
use serde::Deserialize;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecPayload {
    error: Option<ExecError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecError {
    message: String,
    identifier: Option<String>,
}

fn init_options() -> JsValue {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &JsValue::from_str("enableGpu"), &JsValue::FALSE)
        .expect("disable browser GPU for portable examples");
    js_sys::Reflect::set(
        &options,
        &JsValue::from_str("telemetryConsent"),
        &JsValue::FALSE,
    )
    .expect("disable telemetry for browser qualification");
    options.into()
}

fn text_request(name: &str, text: &str) -> JsValue {
    let source = js_sys::Object::new();
    js_sys::Reflect::set(
        &source,
        &JsValue::from_str("kind"),
        &JsValue::from_str("text"),
    )
    .expect("set source kind");
    js_sys::Reflect::set(
        &source,
        &JsValue::from_str("name"),
        &JsValue::from_str(name),
    )
    .expect("set source name");
    js_sys::Reflect::set(
        &source,
        &JsValue::from_str("text"),
        &JsValue::from_str(text),
    )
    .expect("set source text");
    let request = js_sys::Object::new();
    js_sys::Reflect::set(&request, &JsValue::from_str("source"), source.as_ref())
        .expect("set source request");
    request.into()
}

#[wasm_bindgen_test(async)]
async fn portable_max_catalog_examples_execute_in_browser() {
    let runtime = runmat_wasm::init_runmat(init_options())
        .await
        .expect("initialize browser runtime without GPU");
    let examples = MAX_CATALOG_ENTRY.documentation.examples;
    assert_eq!(
        examples.len(),
        6,
        "the six max examples are the browser oracle"
    );
    for example in examples {
        assert_eq!(example.harness, BuiltinExampleHarness::Portable);
        assert_eq!(example.fixture, BuiltinExampleFixture::None);
        let BuiltinExampleVerification::Assertions { source: assertions } = example.verification
        else {
            panic!("max example {} lacks executable assertions", example.id);
        };
        let program = format!("{}\n{}\n", example.program, assertions);
        let name = format!("max_{}.m", example.id);
        let payload: ExecPayload = serde_wasm_bindgen::from_value(
            runtime
                .execute_request_js(text_request(&name, &program))
                .await
                .expect("execute max browser example"),
        )
        .expect("decode browser execution payload");
        assert!(
            payload.error.is_none(),
            "max example {} failed: {:?}",
            example.id,
            payload.error
        );
    }
}

#[wasm_bindgen_test(async)]
async fn invalid_max_dimension_keeps_owning_error_identifier_in_browser() {
    let runtime = runmat_wasm::init_runmat(init_options())
        .await
        .expect("initialize browser runtime without GPU");
    let payload: ExecPayload = serde_wasm_bindgen::from_value(
        runtime
            .execute_request_js(text_request("max_invalid_dimension.m", "max([1 2], [], 0)"))
            .await
            .expect("execute invalid max dimension"),
    )
    .expect("decode browser execution payload");
    let error = payload.error.expect("invalid dimension must fail");
    assert_eq!(
        error.identifier.as_deref(),
        MAX_ERROR_INVALID_ARGUMENT.identifier,
        "unexpected browser max error: {}",
        error.message
    );
}
