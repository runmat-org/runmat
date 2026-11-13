use super::*;

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_folded_min_include_omit_nan_and_signed_zero_full_call() {
    runmat_runtime::builtins::wasm_registry::register_all();
    let provider = register_wgpu_provider_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU adapter required for NaN extrema qualification");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    for omit in [false, true] {
        special_cases::assert_borrowed_nan_min(provider, omit).await;
    }
    special_cases::assert_empty_min_ordinary_fallback().await;
}
