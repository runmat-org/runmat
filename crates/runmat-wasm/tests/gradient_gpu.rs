#![cfg(target_arch = "wasm32")]

use runmat_wasm::init_runmat;
use serde::Deserialize;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecPayload {
    value_text: Option<String>,
    error: Option<ExecError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuStatusPayload {
    requested: bool,
    active: bool,
    error: Option<String>,
}

fn init_options(enable_gpu: bool) -> JsValue {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(
        &options,
        &JsValue::from_str("enableGpu"),
        &JsValue::from_bool(enable_gpu),
    )
    .expect("set enableGpu");
    options.into()
}

#[wasm_bindgen_test(async)]
async fn gradient_gpu_row_vector_matches_expected_output() {
    let runtime = init_runmat(init_options(true))
        .await
        .expect("initialize wasm runtime");

    let gpu_status: GpuStatusPayload =
        serde_wasm_bindgen::from_value(runtime.gpu_status().expect("gpu status"))
            .expect("deserialize gpu status");

    let payload: ExecPayload = serde_wasm_bindgen::from_value(
        runtime
            .execute(
                "G = gpuArray(single([1 4 9]));\nD = gradient(G, 2);\nout = gather(D)".to_string(),
            )
            .await
            .expect("execute gradient script"),
    )
    .expect("deserialize execution payload");

    if let Some(err) = payload.error {
        panic!(
            "gradient wasm execution failed: {} (gpu requested={}, active={}, gpu error={:?})",
            err.message, gpu_status.requested, gpu_status.active, gpu_status.error
        );
    }

    let value_text = payload.value_text.unwrap_or_default();
    assert!(
        value_text.contains("1.5") && value_text.contains("2") && value_text.contains("2.5"),
        "unexpected gradient output: {:?} (gpu requested={}, active={}, gpu error={:?})",
        value_text,
        gpu_status.requested,
        gpu_status.active,
        gpu_status.error
    );
}

#[wasm_bindgen_test(async)]
async fn gradient_gpu_row_vector_matches_expected_output_without_webgpu() {
    let runtime = init_runmat(init_options(false))
        .await
        .expect("initialize wasm runtime");

    let gpu_status: GpuStatusPayload =
        serde_wasm_bindgen::from_value(runtime.gpu_status().expect("gpu status"))
            .expect("deserialize gpu status");
    assert!(
        !gpu_status.active,
        "expected CPU fallback test to disable GPU"
    );

    let payload: ExecPayload = serde_wasm_bindgen::from_value(
        runtime
            .execute(
                "G = gpuArray(single([1 4 9]));\nD = gradient(G, 2);\nout = gather(D)".to_string(),
            )
            .await
            .expect("execute gradient script"),
    )
    .expect("deserialize execution payload");

    if let Some(err) = payload.error {
        panic!(
            "gradient wasm CPU-fallback execution failed: {} (gpu requested={}, active={}, gpu error={:?})",
            err.message, gpu_status.requested, gpu_status.active, gpu_status.error
        );
    }

    let value_text = payload.value_text.unwrap_or_default();
    assert!(
        value_text.contains("1.5") && value_text.contains("2") && value_text.contains("2.5"),
        "unexpected CPU-fallback gradient output: {:?} (gpu requested={}, active={}, gpu error={:?})",
        value_text,
        gpu_status.requested,
        gpu_status.active,
        gpu_status.error
    );
}
