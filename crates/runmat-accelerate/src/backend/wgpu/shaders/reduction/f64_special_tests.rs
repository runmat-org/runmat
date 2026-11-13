use super::*;
use crate::backend::wgpu::shaders::scan;

#[test]
fn f64_special_values_use_exact_word_halves_in_every_selected_reduction_and_scan() {
    assert!(REDUCE_GLOBAL_SHADER_F64.contains("runmat_f64_from_words(0u, 0x7ff00000u)"));
    assert!(REDUCE_GLOBAL_SHADER_F64.contains("runmat_f64_from_words(0u, 0xfff00000u)"));
    for source in [
        REDUCE_GLOBAL_SHADER_F64.to_owned(),
        REDUCE_DIM_SHADER_F64.to_owned(),
        REDUCE_DIM_MINMAX_SHADER_F64.to_owned(),
        scan::CUMSUM_SHADER_F64.to_owned(),
        scan::CUMPROD_SHADER_F64.to_owned(),
        scan::CUMMIN_SHADER_F64.to_owned(),
        scan::CUMMAX_SHADER_F64.to_owned(),
        ordered_mean::source(NumericPrecision::F64, NumericPrecision::F64),
    ] {
        assert!(!source.contains("var bits: u64 = 0x"));
        if source.contains("fn nan_f64()") {
            assert!(source.contains("runmat_f64_from_words(0u, 0x7ff80000u)"));
        }
    }
}
