use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dimensional_source_freezes_actual_shader_token_not_dispatch_divisor() {
    for source in [REDUCE_DIM_SHADER_F32, REDUCE_DIM_SHADER_F64] {
        let first = freeze_dim_reduction_source(source, 64);
        let second = freeze_dim_reduction_source(source, 128);
        assert_ne!(first, second);
        assert!(!first.contains("@WG@"));
        assert!(!second.contains("@WG@"));
        assert!(first.contains("@compute @workgroup_size(64)"));
        assert!(second.contains("@compute @workgroup_size(128)"));
    }
    for workgroup in [64, 128, 256] {
        for output_words in [1, 65, 129, 513] {
            let groups =
                crate::backend::wgpu::dispatch::common::dispatch_size(output_words, workgroup);
            assert!(groups * workgroup >= output_words);
            if output_words > 1 {
                assert!((groups - 1) * workgroup < output_words);
            }
        }
    }
    assert_eq!(
        crate::backend::wgpu::dispatch::common::dispatch_size(65, 64),
        2
    );
}
