use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn selected_ieee_words_need_the_enabled_device_feature_not_adapter_intent() {
    let f64_only = wgpu::Features::SHADER_F64;
    let f64_with_words = f64_only | wgpu::Features::SHADER_INT64;
    assert!(required_features_enabled(wgpu::Features::empty(), f64_only));
    assert!(!required_features_enabled(
        wgpu::Features::SHADER_INT64,
        f64_only
    ));
    assert!(required_features_enabled(
        wgpu::Features::SHADER_INT64,
        f64_with_words
    ));
}
