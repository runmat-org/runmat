use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn global_minmax_selected_sources_validate_without_new_capability_tax() {
    for (precision, caps) in [
        (NumericPrecision::F32, RequiredCaps::None),
        (NumericPrecision::F64, RequiredCaps::Float64Int64),
    ] {
        let source = source(precision);
        assert_eq!(source.matches("fn runmat_real_max(").count(), 1);
        assert_eq!(source.matches("fn runmat_real_min(").count(), 1);
        assert!(source.contains("case 2u: { return runmat_real_min(a, b); }"));
        assert!(source.contains("case 3u: { return runmat_real_max(a, b); }"));
        assert!(source.contains("case 0u: { return a + b; }"));
        assert!(source.contains("case 1u: { return a * b; }"));
        assert!(source.contains("case 4u: { return a + b; }"));
        for workgroup in [64, 128, 256] {
            validate_generated(
                &format!("global-minmax-{precision:?}-wg{workgroup}"),
                &source,
                workgroup,
                8,
                caps,
            );
        }
    }
}
