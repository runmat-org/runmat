#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::wasm_bindgen_test;

#[path = "support/symptom_regressions_shared.rs"]
mod shared;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

#[wasm_bindgen_test(async)]
async fn impedance_loop_executes_without_runtime_error() {
    shared::assert_impedance_loop_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn slice_end_arithmetic_executes_without_runtime_error() {
    shared::assert_slice_end_arithmetic_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn tic_toc_loop_executes_without_runtime_error() {
    shared::assert_tic_toc_loop_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn symbolic_limit_workflow_executes_without_runtime_error() {
    shared::assert_symbolic_limit_workflow_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn signal_compatibility_harness_executes_without_runtime_error() {
    shared::assert_signal_compatibility_harness_executes_without_runtime_error().await;
}
